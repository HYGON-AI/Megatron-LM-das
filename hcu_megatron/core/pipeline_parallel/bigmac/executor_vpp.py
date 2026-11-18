# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

import os
import pickle
import torch
import torch.distributed as dist
from typing import List
from .communication import (
    P2PHandleAdapter,
    configure_vpp_p2p,
    all_gather_along_pp,
    scatter_along_pp,
    release_model_grad_buffer,
    release_tensor,
    rebuild_model_grad_buffer,
)
from .schedules import (
    VPP1F1BSchedule,
    OpType,
    ScheduleOp,
    BigMacVPP,
    UnifiedBigMacVPP,
    FusedScheduleOp,
)
from . import visualize_vppschedule
from .executor import (
    LLMDataIterator,
    LLMPPExecutor,
    P2PCommunicator,
    is_pp_last_stage,
    is_pp_first_stage,
    _get_tensor_shapes_compat,
    _forward_step_compat,
    _backward_step_compat,
    _call_with_supported_kwargs,
    _new_total_num_tokens,
    _get_model_no_sync_context,
    _run_encoder_backward,
)
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import (
    forward_step,
    get_model_config,
    get_model_type,
    get_tensor_shapes,
    deallocate_output_tensor,
    backward_step,
)
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from torch.autograd.variable import Variable


try:
    from megatron.core.process_groups_config import ProcessGroupCollection
except ImportError:
    class ProcessGroupCollection:
        pass


def _create_fake_llm_input_grads(encoder_output_metas, device):
    # Fake VE slots still need shape-compatible gradient placeholders so the
    # backward reshard path can preserve the same structure as real microbatches.
    fake_grads = []
    for meta in encoder_output_metas:
        if not meta.requires_grad:
            fake_grads.append(None)
            continue
        fake_grads.append(
            torch.zeros(meta.shape, dtype=meta.dtype, device=device)
        )
    return fake_grads

class LLMVPPExecutor(LLMPPExecutor):
    def __init__(self, 
                forward_step_func, 
                data_iterator,
                model,
                num_microbatches,
                seq_length,
                micro_batch_size,
                forward_only,
                pp_size,
                vpp_size,
                pp_group,
                pp_rank,
                collect_non_loss_data: bool = False,
                get_tensor_shapes_fn: callable = None,
                trace_enabled: bool = False,
                trace_dir: str = None,
                trace_iter_id: int = None):
        assert not forward_only, "validation is not supported currently"

        assert isinstance(model, list), "model must be a list"
        assert len(model) == vpp_size, "model must have the same number of chunks as vpp_size"
        assert isinstance(data_iterator, list), "data_iterator must be a list"
        assert len(data_iterator) == vpp_size, "data_iterator must have the same number of chunks as vpp_size"
        
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.forward_only = forward_only
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.pp_rank = pp_rank
        self.trace_enabled = trace_enabled
        self.trace_dir = trace_dir
        self.trace_iter_id = trace_iter_id
        self.trace_output_path = None
        self._trace_origin_us = None
        self._trace_origin_perf_ns = None
        self._trace_base_event = None
        self.config = get_model_config(self.model[0])
        self.model_type = get_model_type(self.model[0])
        # assert os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] == '1'

        self.p2p_communicator = P2PHandleAdapter(P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=self.config
        ))
        configure_vpp_p2p(self.p2p_communicator.communicator)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        if cp_group.size() != 1:
            raise NotImplementedError("This open-source pp_executor path supports single context group execution only.")

        self.grad_finalize_pgs = ProcessGroupCollection()
        self.grad_finalize_pgs.tp = tp_group
        self.grad_finalize_pgs.pp = pp_group
        self.grad_finalize_pgs.embd = embd_group
        self.grad_finalize_pgs.pos_embd = pos_emb_group
        self.grad_finalize_pgs.cp = cp_group
        self.grad_finalize_pgs.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        if hasattr(parallel_state, "get_tensor_and_data_parallel_group"):
            self.grad_finalize_pgs.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
                with_context_parallel=True
            )
        self.collect_non_loss_data = collect_non_loss_data

        self.action_map = {}
        self.get_tensor_shapes_fn = get_tensor_shapes_fn
        if self.get_tensor_shapes_fn is None:
            self.get_tensor_shapes_fn = get_tensor_shapes
        self.tensor_shapes = _get_tensor_shapes_compat(
            self.get_tensor_shapes_fn,
            seq_length=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            decoder_seq_length=None,
            config=self.config,
            model_type=self.model_type,
        )

        self.register_actions_for_llm_pp()

        self.vpp_size = vpp_size
        self.scheduler = VPP1F1BSchedule(pp_size, vpp_size, num_microbatches)
        
        self.scheduler.generate_schedule()
        self.scheduler.add_communication_op()
        self.scheduler.optimize_communication_ops()
        self.scheduler.fuse_comm_ops()
        
        self.schedule_plan = self.scheduler.schedules_with_comm[self.pp_rank]
        self.debug = False
    
    
    def register_actions_for_llm_pp(self):
        # Drive the scheduler through a simple opcode -> handler dispatch table.
        for op_type in OpType:
            self.register_action(op_type, self.default_action)
        
        self.register_action(OpType.FORWARD, self.forward_action)
        self.register_action(OpType.BACKWARD, self.backward_action)
        self.register_action(OpType.SENDFORWARD, self.send_forward_action)
        self.register_action(OpType.RECVFORWARD, self.recv_forward_action)
        self.register_action(OpType.SENDBACKWARD, self.send_backward_action)
        self.register_action(OpType.RECVBACKWARD, self.recv_backward_action)
        self.register_action(OpType.SENDFORWARDRECVBACKWARD, self.send_forward_recv_backward_action)
        self.register_action(OpType.SENDBACKWARDRECVFORWARD, self.send_backward_recv_forward_action)
        self.register_action(OpType.SENDFORWARDRECVFORWARD, self.send_forward_recv_forward_action)
        self.register_action(OpType.SENDBACKWARDRECVBACKWARD, self.send_backward_recv_backward_action)
    
    
    def forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        parallel_state.set_virtual_pipeline_model_parallel_rank(chunk_id)
        if op.microbatch_id in self.recv_forward_handles[chunk_id]:
            # Overlapped fused comm stores the recv handle here; the forward of
            # this virtual stage must not read its input until the transfer lands.
            recv_handle = self.recv_forward_handles[chunk_id].pop(op.microbatch_id)
            if recv_handle is not None:
                recv_handle.wait()
        assert chunk_id is not None, "chunk_id is not set"
        if self.data_iterator[chunk_id]:
            llm_data_iterator = self.llm_data_iterators[chunk_id]
        else:
            llm_data_iterator = None
        if self.is_first_stage(op):
            input_tensor = [None]
        else:
            input_tensor = self.input_tensors[chunk_id][op.microbatch_id]
        
        with self._execute_compute_with_trace(op):
            output_tensor, num_tokens = _forward_step_compat(
                self.forward_step_func,
                llm_data_iterator,
                self.model[chunk_id],
                self.num_microbatches,
                input_tensor,
                self.forward_data_store,
                self.config,
                cp_group_size=self.grad_finalize_pgs.cp.size(),
                collect_non_loss_data=self.collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=op.microbatch_id == 0,
                current_microbatch=op.microbatch_id,
                vp_stage=chunk_id,
                is_last_stage=is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id == self.vpp_size - 1,
            )
        self._accumulate_num_tokens(num_tokens)
        self.output_tensors[chunk_id].append(output_tensor)
    
    def backward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        parallel_state.set_virtual_pipeline_model_parallel_rank(chunk_id)
        assert chunk_id is not None, "chunk_id is not set"
        # If the pending delayed-forward-send belongs to this exact
        # (chunk, microbatch), drain it before backward_step touches output_tensor.
        self._drain_pending_send_forward_for(chunk_id, op.microbatch_id)
        if op.microbatch_id in self.recv_backward_handles[chunk_id]:
            recv_handle = self.recv_backward_handles[chunk_id].pop(op.microbatch_id)
            if recv_handle is not None:
                recv_handle.wait()
        input_tensor = self.input_tensors[chunk_id][op.microbatch_id]
        output_tensor = self.output_tensors[chunk_id][op.microbatch_id]
        output_tensor_grad = self.output_tensor_grads[chunk_id][op.microbatch_id]
        # Megatron DDP overlap_grad_reduce expects us to disable grad synchronization for
        # all non-last microbatches (gradient accumulation), otherwise the same parameter
        # will be "registered grad-ready" multiple times and trip:
        #   AssertionError: Cannot set grad twice
        is_last_microbatch = op.microbatch_id == (self.num_microbatches - 1)
        model_for_chunk = self.model[chunk_id] if isinstance(self.model, (list, tuple)) else self.model
        with _get_model_no_sync_context(
            model_for_chunk,
            is_last_microbatch=is_last_microbatch,
        ), self._execute_compute_with_trace(op):
            input_tensor_grad = _backward_step_compat(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                self.model_type,
                self.config,
            )
        self.input_tensor_grads[chunk_id].append(input_tensor_grad)
        self.input_tensors[chunk_id][op.microbatch_id] = None
        self.output_tensors[chunk_id][op.microbatch_id] = None
        self.output_tensor_grads[chunk_id][op.microbatch_id] = None
    
    def send_forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        assert chunk_id is not None, "chunk_id is not set"
        output_tensor = self.output_tensors[chunk_id][op.microbatch_id]
        if not isinstance(output_tensor, list):
            assert isinstance(output_tensor, torch.Tensor), f"expect output_tensor to be a list or a tensor, but got {output_tensor}"
            output_tensor = [output_tensor]
        assert len(output_tensor) == 1, "output_tensor must contain a single tensor for VPP send_forward"
        # Any subsequent forward-send path should drain the previous pending one.
        self._drain_pending_send_forward()
        _, wait_handles = self.p2p_communicator.send_forward_recv_forward(
            output_tensor[0],
            recv_prev=False,
            tensor_shape=None,
            overlap_p2p_comm=True,
        )
        send_handle = wait_handles.get("send_next") if isinstance(wait_handles, dict) else None
        if send_handle is None:
            deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)
        else:
            self.send_forward_handles = (
                send_handle,
                output_tensor[0],
                chunk_id,
                op.microbatch_id,
            )
    
    def send_backward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        assert chunk_id is not None, "chunk_id is not set"
        input_tensor_grad = self.input_tensor_grads[chunk_id][op.microbatch_id]
        if isinstance(input_tensor_grad, list):
            assert len(input_tensor_grad) == 1, "input_tensor_grad must contain a single tensor for VPP send_backward"
            input_tensor_grad = input_tensor_grad[0]
        # Any subsequent backward-send path should drain the previous pending one.
        self._drain_pending_send_backward()
        _, wait_handles = self.p2p_communicator.send_backward_recv_backward(
            input_tensor_grad,
            recv_next=False,
            tensor_shape=None,
            overlap_p2p_comm=True,
        )
        send_handle = wait_handles.get("send_prev") if isinstance(wait_handles, dict) else None
        if send_handle is None:
            self.input_tensor_grads[chunk_id][op.microbatch_id] = None
        else:
            self.send_backward_handles = (send_handle, chunk_id, op.microbatch_id)
    
    def recv_forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        assert chunk_id is not None, "chunk_id is not set"
        data_iterator = self.data_iterator[chunk_id]
        if data_iterator:
            # Preload the payload that this virtual chunk will consume once the
            # pipeline input tensor becomes available.
            data = next(data_iterator)
            self.llm_data_iterators[chunk_id].put(data)
        else:
            data = None
        assert isinstance(self.tensor_shapes, list) and len(self.tensor_shapes) == 1, "tensor_shapes must be a list with length 1 for VPP recv_forward"
        recv_prev = not (is_pp_first_stage(self.p2p_communicator.pp_group) and chunk_id == 0)
        input_tensor, wait_handles = self.p2p_communicator.send_forward_recv_forward(
            None,
            recv_prev=recv_prev,
            tensor_shape=self.tensor_shapes[0],
            overlap_p2p_comm=True,
        )
        if recv_prev and isinstance(wait_handles, dict):
            self.recv_forward_handles[chunk_id][op.microbatch_id] = wait_handles.get("recv_prev")
        self.input_tensors[chunk_id].append([input_tensor])
    
    def recv_backward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        assert chunk_id is not None, "chunk_id is not set"
        assert isinstance(self.tensor_shapes, list) and len(self.tensor_shapes) == 1, "tensor_shapes must be a list with length 1 for VPP recv_backward"
        recv_next = not (
            is_pp_last_stage(self.p2p_communicator.pp_group)
            and chunk_id == self.vpp_size - 1
        )
        output_tensor_grad, wait_handles = self.p2p_communicator.send_backward_recv_backward(
            input_tensor_grad=None,
            recv_next=recv_next,
            tensor_shape=self.tensor_shapes[0],
            overlap_p2p_comm=True,
        )
        if recv_next and isinstance(wait_handles, dict):
            self.recv_backward_handles[chunk_id][op.microbatch_id] = wait_handles.get("recv_next")
        self.output_tensor_grads[chunk_id].append([output_tensor_grad])
    
    def send_forward_recv_backward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2
        chunk_id1 = op1.chunk_id
        assert chunk_id1 is not None, "chunk_id is not set"
        chunk_id2 = op2.chunk_id
        assert chunk_id2 is not None, "chunk_id is not set"
        
        output_tensor = self.output_tensors[chunk_id1][op1.microbatch_id]
        if isinstance(output_tensor, torch.Tensor):
            output_tensor = [output_tensor]

        # This op launches a forward send as well; it should also drain pending forward-send.
        self._drain_pending_send_forward()
        
        output_tensor_grad = self.p2p_communicator.send_forward_recv_backward(
            output_tensor, 
            self.tensor_shapes, 
            is_last_stage = is_pp_last_stage(self.p2p_communicator.pp_group) and op2.chunk_id == self.vpp_size - 1
        )
        deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs) 
        self.output_tensor_grads[chunk_id2].append(output_tensor_grad)
    
    def send_backward_recv_forward_action(self, op: FusedScheduleOp):
        raise NotImplementedError("send_backward_recv_forward_action should not be called for vpp executor")

    def _drain_pending_send_forward(self):
        # In overlap mode we defer send completion/deallocation by one send.
        pending = self.send_forward_handles
        if pending is None:
            return
        send_handle, send_tensor, _, _ = pending
        send_handle.wait()
        deallocate_output_tensor(send_tensor, self.config.deallocate_pipeline_outputs)
        self.send_forward_handles = None

    def _drain_pending_send_forward_for(self, chunk_id: int, microbatch_id: int):
        # Drain only if the pending delayed send corresponds to this exact
        # output tensor that is about to be consumed by backward.
        pending = self.send_forward_handles
        if pending is None:
            return
        send_handle, send_tensor, pending_chunk_id, pending_microbatch_id = pending
        if pending_chunk_id != chunk_id or pending_microbatch_id != microbatch_id:
            return
        send_handle.wait()
        deallocate_output_tensor(send_tensor, self.config.deallocate_pipeline_outputs)
        self.send_forward_handles = None

    def _drain_pending_send_backward(self):
        # Match forward overlap behavior: keep one in-flight send backward op.
        pending = self.send_backward_handles
        if pending is None:
            return
        send_handle, chunk_id, microbatch_id = pending
        send_handle.wait()
        self.input_tensor_grads[chunk_id][microbatch_id] = None
        self.send_backward_handles = None

    def send_forward_recv_forward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2

        chunk_id1 = op1.chunk_id
        assert chunk_id1 is not None, "chunk_id is not set"
        chunk_id2 = op2.chunk_id
        assert chunk_id2 is not None, "chunk_id is not set"
        output_tensor = self.output_tensors[chunk_id1][op1.microbatch_id]

        if not isinstance(output_tensor, list):
            assert isinstance(output_tensor, torch.Tensor), f"expect output_tensor to be a list or a tensor, but got {output_tensor}"
            output_tensor = [output_tensor]

        assert isinstance(output_tensor, list) and len(output_tensor) == 1, "output_tensor must be a list with length 1 for now"
        assert isinstance(self.tensor_shapes, list) and len(self.tensor_shapes) == 1, "tensor_shapes must be a list with length 1 for now"

        # Wait/deallocate the previous overlapped send before launching a new one.
        self._drain_pending_send_forward()
        
        input_tensor, wait_handles = self.p2p_communicator.send_forward_recv_forward(
            output_tensor[0], 
            recv_prev=not (is_pp_first_stage(self.p2p_communicator.pp_group) and chunk_id2 == 0),
            tensor_shape=self.tensor_shapes[0],
            overlap_p2p_comm=True
        )
        send_handle = wait_handles.get("send_next") if isinstance(wait_handles, dict) else None
        if send_handle is None:
            deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)
        else:
            self.send_forward_handles = (
                send_handle,
                output_tensor[0],
                chunk_id1,
                op1.microbatch_id,
            )
        if isinstance(wait_handles, dict) and 'recv_prev' in wait_handles:
            self.recv_forward_handles[chunk_id2][op2.microbatch_id] = wait_handles.get('recv_prev')

        data_iterator = self.data_iterator[chunk_id2]
        if data_iterator:
            data = next(data_iterator)
            self.llm_data_iterators[chunk_id2].put(data)
        else:
            data = None
        input_tensor = [input_tensor]
        self.input_tensors[chunk_id2].append(input_tensor)
        
    
    def send_backward_recv_backward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2
        chunk_id1 = op1.chunk_id
        assert chunk_id1 is not None, "chunk_id is not set"
        chunk_id2 = op2.chunk_id
        assert chunk_id2 is not None, "chunk_id is not set"

        input_tensor_grad = self.input_tensor_grads[chunk_id1][op1.microbatch_id]

        if isinstance(input_tensor_grad, list):
            assert len(input_tensor_grad) == 1, "input_tensor_grad must be a list with length 1 for now"
            input_tensor_grad = input_tensor_grad[0]

        # Wait/release the previous overlapped send before launching a new one.
        self._drain_pending_send_backward()
        
        output_tensor_grad, wait_handles = self.p2p_communicator.send_backward_recv_backward(
            input_tensor_grad,
            recv_next=not (is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id2 == self.vpp_size - 1),
            tensor_shape=self.tensor_shapes[0],
            overlap_p2p_comm=True
        )
        
        send_handle = wait_handles.get("send_prev") if isinstance(wait_handles, dict) else None
        if send_handle is None:
            self.input_tensor_grads[chunk_id1][op1.microbatch_id] = None
        else:
            self.send_backward_handles = (send_handle, chunk_id1, op1.microbatch_id)
        if isinstance(wait_handles, dict) and 'recv_next' in wait_handles:
            self.recv_backward_handles[chunk_id2][op2.microbatch_id] = wait_handles.get('recv_next')
        self.output_tensor_grads[chunk_id2].append(output_tensor_grad)
    
    def reset(self):
        # Runtime buffers are recreated per schedule run because the same executor
        # instance can be reused across steps with different in-flight microbatches.
        self.llm_data_iterators = [LLMDataIterator() for _ in range(self.vpp_size)]
        self.forward_data_store = []
        self.total_num_tokens = _new_total_num_tokens()
        self.input_tensors = [[] for _ in range(self.vpp_size)]
        self.output_tensors = [[] for _ in range(self.vpp_size)]
        self.output_tensor_grads = [[] for _ in range(self.vpp_size)]
        self.input_tensor_grads = [[] for _ in range(self.vpp_size)]
        self.send_forward_handles = None
        self.send_backward_handles = None
        self.recv_forward_handles = [{} for _ in range(self.vpp_size)]
        self.recv_backward_handles = [{} for _ in range(self.vpp_size)]
    
    def post_process(self):
        self._drain_pending_send_forward()
        self._drain_pending_send_backward()
        del self.llm_data_iterators
        del self.input_tensors
        del self.output_tensors
        del self.output_tensor_grads
        del self.input_tensor_grads
        del self.send_forward_handles
        del self.send_backward_handles
        del self.recv_forward_handles
        del self.recv_backward_handles
class LLMVPPExecutorWithVE(LLMVPPExecutor):
    def __init__(self, 
                forward_step_func, 
                data_iterator,
                model,
                num_microbatches,
                seq_length,
                micro_batch_size,
                forward_only,
                pp_size,
                vpp_size,
                pp_group,
                pp_rank,
                collect_non_loss_data: bool = False,
                get_tensor_shapes_fn: callable = None,
                encoder_models: List[torch.nn.Module] = [],
                encoder_forward_func: callable = None,
                get_encoder_output_meta_func: callable = None,
                trace_enabled: bool = False,
                trace_dir: str = None,
                trace_iter_id: int = None):
        
        assert not isinstance(data_iterator, list)
        data_iterator_for_init = [data_iterator] * vpp_size
        super().__init__(
            forward_step_func,
            data_iterator_for_init,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            forward_only,
            pp_size,
            vpp_size,
            pp_group,
            pp_rank,
            collect_non_loss_data,
            get_tensor_shapes_fn,
            trace_enabled=trace_enabled,
            trace_dir=trace_dir,
            trace_iter_id=trace_iter_id,
        )
        self.data_iterator = data_iterator
        self.encoder_models = encoder_models
        self.encoder_forward_func = encoder_forward_func
        owner = getattr(self.encoder_forward_func, "__self__", None)
        vl_args = getattr(owner, "vl_args", None)
        self.vl_args = vl_args
        self.rebuild_grad_buffer = bool(getattr(vl_args, "rebuild_grad_buffer", False))
        self.offload_llm_input = bool(getattr(vl_args, "offload_ve_output", False))
        self.accurate_attn_mask_with_cp = bool(
            getattr(vl_args, "accurate_attn_mask_with_cp", False)
        )
        self.accurate_attn_mask_eod_token = getattr(vl_args, "accurate_attn_mask_eod_token", None)

        self.scheduler = BigMacVPP(pp_size, vpp_size, num_microbatches)
        self.scheduler.generate_schedule()
        if self.rebuild_grad_buffer and hasattr(self.scheduler, "add_encoder_grad_buffer_ops"):
            self.scheduler.add_encoder_grad_buffer_ops()
        self.scheduler.add_communication_op()
        self.scheduler.optimize_communication_ops()
        self.scheduler.fuse_comm_ops()
        if self.offload_llm_input and hasattr(self.scheduler, "add_llm_input_offload_onload_ops"):
            self.scheduler.add_llm_input_offload_onload_ops()
        self.schedule_plan = self.scheduler.schedules_with_comm[self.pp_rank]

        if get_encoder_output_meta_func is not None:
            self.encoder_output_metas = get_encoder_output_meta_func()
        else:
            raise NotImplementedError("automatically generate encoder output meta is not supported currently")

        self.register_actions_for_ve()
        self.pp_comm_launch_stream = torch.cuda.Stream()
        # Dedicated stream for LLM input offload/onload memcpy so it can
        # overlap with compute on the default stream.
        self.offload_stream = torch.cuda.Stream()

    def reset(self):
        super().reset()
        self._last_ve_forward_contains_fake = False
        self._last_ve_forward_output_is_fake = False
        owner = getattr(self.encoder_forward_func, "__self__", None)
        if owner is not None and hasattr(owner, "reset_pp_executor_encoder_forward_state"):
            owner.reset_pp_executor_encoder_forward_state()
        # VE forward outputs queued in global scheduler order, before we know
        # whether the current send op is tagged real or fake.
        self.ve_pending_output_tensors = []
        self.ve_pending_forward_contains_fake = []
        self.ve_pending_forward_output_is_fake = []
        self.ve_pending_backward_tensors = []
        # Real/fake VE outputs after encoder_send_forward_action routes them.
        self.ve_output_tensors = []
        # The exact encoder outputs that participated in forward reshard and
        # must be released once first PP stage finishes consuming them.
        self.ve_forward_release_tensors = []
        # Keep only fake masks for backward reconstruction to avoid retaining
        # whole llm_input dicts (e.g. input_embed) until encoder_recv_backward.
        self.ve_llm_input_fake_masks_for_backward = []
        self.ve_fake_llm_input_fake_masks_for_backward = []
        # VE forward activations needed by the later VE backward pass.
        self.ve_backward_tensors = []
        self.ve_fake_backward_tensors = []
        self.ve_output_tensor_grads = []
        self.ve_fake_output_tensor_grads = []
        # Async communication handles for VE -> LLM forward and the reverse backward path.
        self.ve_send_forward_handles = []
        self.ve_send_forward_keepalives = []
        self.ve_send_forward_ready_events = []
        self.ve_send_backward_handles = []
        self.ve_send_backward_ready_events = []
        self.ve_fake_send_backward_handles = []
        self.ve_fake_send_backward_ready_events = []
        self.llm_input_offload_done_events = {}
        self.llm_input_onload_done_events = {}
        # First PP stage keeps the real LLM inputs/gradients so VE backward can
        # rebuild encoder_output_grads in the original reshard order.
        self.first_pp_inputs = []
        self.first_pp_input_grads = []
        self.first_pp_input_grads_cursor = 0
    
    def post_process(self):
        super().post_process()
        del self.ve_pending_output_tensors
        del self.ve_pending_forward_contains_fake
        del self.ve_pending_forward_output_is_fake
        del self.ve_pending_backward_tensors
        del self.ve_output_tensors
        del self.ve_forward_release_tensors
        del self.ve_llm_input_fake_masks_for_backward
        del self.ve_fake_llm_input_fake_masks_for_backward
        del self.ve_backward_tensors
        del self.ve_fake_backward_tensors
        del self.ve_output_tensor_grads
        del self.ve_send_forward_handles
        del self.ve_send_forward_keepalives
        del self.ve_send_forward_ready_events
        del self.ve_send_backward_handles
        del self.ve_send_backward_ready_events
        del self.ve_fake_output_tensor_grads
        del self.ve_fake_send_backward_handles
        del self.ve_fake_send_backward_ready_events
        del self.llm_input_offload_done_events
        del self.llm_input_onload_done_events
        del self.first_pp_inputs
        del self.first_pp_input_grads
        del self.first_pp_input_grads_cursor
    
    def register_actions_for_ve(self):
        self.register_action(OpType.VEFORWARD, self.encoder_forward_action)
        self.register_action(OpType.VEBACKWARD, self.encoder_backward_action)
        self.register_action(OpType.VESENDFORWARD, self.encoder_send_forward_action)
        self.register_action(OpType.VERECVFORWARD, self.encoder_recv_forward_action)
        self.register_action(OpType.VESENDBACKWARD, self.encoder_send_backward_action)
        self.register_action(OpType.VERECVBACKWARD, self.encoder_recv_backward_action)
        if hasattr(OpType, "VEGRADBUFFERRELEASE"):
            self.register_action(OpType.VEGRADBUFFERRELEASE, self.encoder_grad_buffer_release_action)
        if hasattr(OpType, "VEGRADBUFFERREBUILD"):
            self.register_action(OpType.VEGRADBUFFERREBUILD, self.encoder_grad_buffer_rebuild_action)
        if hasattr(OpType, "LLMINPUTOFFLOAD"):
            self.register_action(OpType.LLMINPUTOFFLOAD, self.llm_input_offload_action)
        if hasattr(OpType, "LLMINPUTONLOAD"):
            self.register_action(OpType.LLMINPUTONLOAD, self.llm_input_onload_action)

    def encoder_grad_buffer_release_action(self, op: ScheduleOp):
        if not self.rebuild_grad_buffer:
            return
        if not self.encoder_models:
            return
        release_model_grad_buffer(self.encoder_models)

    def encoder_grad_buffer_rebuild_action(self, op: ScheduleOp):
        if not self.rebuild_grad_buffer:
            return
        if not self.encoder_models:
            return
        rebuild_model_grad_buffer(self.encoder_models)

    def llm_input_offload_action(self, op: ScheduleOp):
        self._finish_ve_forward_handoff(op.microbatch_id)
        
        if not self.is_first_stage(op):
            return
        microbatch_id = op.microbatch_id
        if microbatch_id is None or microbatch_id < 0:
            return
        if microbatch_id >= len(self.first_pp_inputs):
            return
        llm_input = self.first_pp_inputs[microbatch_id]
        if llm_input is None:
            return

        # Launch D2H copies on dedicated offload stream.
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            for meta in self.encoder_output_metas:
                if not getattr(meta, "only_requires_in_first_stage", False):
                    continue
                name = meta.name
                value = llm_input.pop(name, None)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value = value.detach()
                    # Make it a leaf tensor before toggling requires_grad flags.
                    cpu_value = value.to("cpu", non_blocking=True)
                    cpu_value.requires_grad_(False)
                    llm_input[name] = cpu_value
                    del value
                else:
                    llm_input[name] = value

            offload_done = torch.cuda.Event()
            offload_done.record(self.offload_stream)
            self.llm_input_offload_done_events[microbatch_id] = offload_done

    def llm_input_onload_action(self, op: ScheduleOp):
        if not self.is_first_stage(op):
            return
        microbatch_id = op.microbatch_id
        if microbatch_id is None or microbatch_id < 0:
            return
        if microbatch_id >= len(self.first_pp_inputs):
            return
        llm_input = self.first_pp_inputs[microbatch_id]
        if llm_input is None:
            return

        target_device = torch.device("cuda", torch.cuda.current_device())
        offload_done = self.llm_input_offload_done_events.pop(microbatch_id, None)
        # Launch H2D copies on dedicated offload stream; if we have a matching
        # D2H offload event, make this stream wait for it first.
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        if offload_done is not None:
            self.offload_stream.wait_event(offload_done)
        with torch.cuda.stream(self.offload_stream):
            for meta in self.encoder_output_metas:
                if not getattr(meta, "only_requires_in_first_stage", False):
                    continue
                name = meta.name
                value = llm_input.pop(name, None)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                    # Keep the onloaded tensor as a leaf so requires_grad_ is valid.
                    cuda_value = value.to(target_device, non_blocking=True)
                    if meta.requires_grad:
                        cuda_value.requires_grad_(True)
                    llm_input[name] = cuda_value
                else:
                    llm_input[name] = value

            onload_done = torch.cuda.Event()
            onload_done.record(self.offload_stream)
            self.llm_input_onload_done_events[microbatch_id] = onload_done
    
    def check_internal(self, data, meta):
        def check_shape(output, output_meta):
            try:
                output_shape = output.shape
                output_meta_shape = output_meta.shape
                if len(output_shape) != len(output_meta_shape):
                    return False
                num_encoders = len(self.encoder_models)
                for i in range(len(output_shape)):
                    if i == 0 and num_encoders > 1:
                        if output_shape[i] != output_meta_shape[i] * num_encoders and output_shape[i] != output_meta_shape[i]:
                            return False
                    else:
                        if output_shape[i] != output_meta_shape[i]:
                            return False
                return True
            except Exception as e:
                print(f"catch error when checking shape of output {output_meta.name}: {e}")
                raise e
        
        def check_pickleable(output):
            try:
                pickle.dumps(output)
                return True
            except Exception:
                return False
        
        if meta.is_none:
            assert data is None, f"data name: {meta.name}, data is not None"
        elif meta.is_object:
            assert check_pickleable(data), f"data name: {meta.name}, data is not pickleable"
        else:
            assert isinstance(data, torch.Tensor), f"data name: {meta.name}, data is not a tensor"
            data_name = meta.name
            assert check_shape(data, meta), f"data name: {data_name}, output shape: {data.shape}, output_meta shape: {meta.shape}"
            assert data.dtype == meta.dtype, f"output name: {data_name}, output dtype: {data.dtype}, output_meta dtype: {meta.dtype}"

    def check_encoder_output_meta(self, encoder_output, microbatch_id, encoder_output_is_fake):
        assert len(encoder_output) == len(self.encoder_output_metas)
        for output, output_meta in zip(encoder_output, self.encoder_output_metas):
            self.check_internal(output, output_meta)
        actual_is_fake = encoder_output_is_fake
        expected_is_fake = microbatch_id < 0
        assert actual_is_fake == expected_is_fake, (
            f"rank {dist.get_rank()} got mismatched VE encoder output type: "
            f"microbatch_id={microbatch_id}, expected_is_fake={expected_is_fake}, "
            f"actual_is_fake={actual_is_fake}"
        )
   
    def encoder_forward_action(self, op: ScheduleOp):
        if hasattr(self.data_iterator, "_zzl_current_microbatch_id"):
            self.data_iterator._zzl_current_microbatch_id = op.microbatch_id
        result = _call_with_supported_kwargs(
            self.encoder_forward_func,
            self.encoder_models,
            self.data_iterator,
            num_microbatches=self.num_microbatches,
            pp_size=self.pp_size,
            encoder_output_metas=self.encoder_output_metas,
            microbatch_id=op.microbatch_id,
        )
        if isinstance(result, tuple) and len(result) == 2:
            encoder_output, ve_backward_outputs = result
        else:
            encoder_output, ve_backward_outputs = result, None
        forward_contains_fake = False
        encoder_output_is_fake = False
        self.check_encoder_output_meta(encoder_output, op.microbatch_id, encoder_output_is_fake)
        self.ve_pending_output_tensors.append(encoder_output)
        self.ve_pending_forward_contains_fake.append(forward_contains_fake)
        self.ve_pending_forward_output_is_fake.append(encoder_output_is_fake)
        self.ve_pending_backward_tensors.append(ve_backward_outputs)

    def encoder_backward_action(self, op: ScheduleOp):
        microbatch_id = op.microbatch_id
        if microbatch_id < 0:
            handles = self.ve_fake_send_backward_handles.pop(0)
            ready_event = self.ve_fake_send_backward_ready_events.pop(0)
            encoder_outputs = None
            ve_backward_outputs = self.ve_fake_backward_tensors.pop(0)
            encoder_output_grads = self.ve_fake_output_tensor_grads.pop(0)
        else:
            group_idx = microbatch_id // self.pp_size
            handles = self.ve_send_backward_handles[group_idx]
            self.ve_send_backward_handles[group_idx] = []
            ready_event = self.ve_send_backward_ready_events[group_idx]
            self.ve_send_backward_ready_events[group_idx] = None
            encoder_outputs = self.ve_output_tensors[group_idx]
            ve_backward_outputs = self.ve_backward_tensors[group_idx]
            encoder_output_grads = self.ve_output_tensor_grads[group_idx]
        if ready_event is not None:
            torch.cuda.current_stream().wait_event(ready_event)
        for handle in handles:
            handle.wait()
        total_ve_backwards = self.num_microbatches // self.pp_size
        current_ve_backward_id = microbatch_id // self.pp_size
        is_last_ve_microbatch = (current_ve_backward_id == total_ve_backwards - 1)
        
        # Set is_last_microbatch flag for the last VE microbatch
        if is_last_ve_microbatch:
            for model in self.encoder_models:
                model.module.is_last_microbatch = True

        self._current_encoder_backward_is_last_microbatch = is_last_ve_microbatch
        try:
            _run_encoder_backward(self, encoder_outputs, encoder_output_grads, ve_backward_outputs)
        finally:
            self._current_encoder_backward_is_last_microbatch = True
        
        # Reset is_last_microbatch flag after backward
        if is_last_ve_microbatch:
            for model in self.encoder_models:
                model.module.is_last_microbatch = False

        if microbatch_id >= 0:
            self.ve_output_tensors[group_idx] = None
            self.ve_backward_tensors[group_idx] = None
            self.ve_output_tensor_grads[group_idx] = None

    
    def encoder_forward_reshard(self, encoder_outputs):
        gathered, pp_handles = all_gather_along_pp(
            encoder_outputs, self.pp_group, self.encoder_output_metas
        )
        return gathered, pp_handles, None, None
    
    def encoder_send_forward_action(self, op: ScheduleOp):
        microbatch_id = op.microbatch_id
        if microbatch_id >= 0:
            # One VE forward fan-outs into a full PP wave. group_idx tracks that
            # VE wave so we can later match the backward reshard to the same data.
            group_idx = microbatch_id // self.pp_size
            assert microbatch_id % self.pp_size == self.pp_rank

        assert self.ve_pending_output_tensors, (
            f"rank {dist.get_rank()} has no pending VE encoder outputs to send for "
            f"microbatch_id={microbatch_id}"
        )
        encoder_outputs = self.ve_pending_output_tensors.pop(0)
        forward_contains_fake = self.ve_pending_forward_contains_fake.pop(0)
        encoder_output_is_fake = self.ve_pending_forward_output_is_fake.pop(0)
        ve_backward_outputs = self.ve_pending_backward_tensors.pop(0)

        llm_inputs, send_handles, send_keepalive, send_ready_event = self.encoder_forward_reshard(encoder_outputs)
        if forward_contains_fake:
            # Once fake slots are involved, we need the fully materialized
            # post-reshard layout to build an exact fake mask for llm_inputs.
            for handle in send_handles:
                handle.wait()
            if send_ready_event is not None:
                torch.cuda.current_stream().wait_event(send_ready_event)
            send_handles = []
            send_keepalive = None
            send_ready_event = None
            local_fake_flag = torch.tensor(
                [1 if encoder_output_is_fake else 0],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            gathered_pp_fake_flags = [torch.empty_like(local_fake_flag) for _ in range(self.pp_size)]
            dist.all_gather(gathered_pp_fake_flags, local_fake_flag, group=self.pp_group)
            llm_input_fake_flags = []
            for pp_idx in range(self.pp_size):
                llm_input_fake_flags.append(bool(gathered_pp_fake_flags[pp_idx].item()))
            assert len(llm_input_fake_flags) == len(llm_inputs), (
                f"rank {dist.get_rank()} built mismatched fake mask length: "
                f"mask_len={len(llm_input_fake_flags)}, llm_inputs_len={len(llm_inputs)}, "
                f"pp_size={self.pp_size}"
            )
        else:
            llm_input_fake_flags = [False] * len(llm_inputs)

        if microbatch_id < 0:
            self.ve_fake_backward_tensors.append(ve_backward_outputs)
            self.ve_fake_llm_input_fake_masks_for_backward.append(llm_input_fake_flags)
        else:
            assert group_idx == len(self.ve_output_tensors), (
                f"rank {dist.get_rank()} saw unexpected VE forward group order: "
                f"group_idx={group_idx}, cached_groups={len(self.ve_output_tensors)}"
            )
            self.ve_output_tensors.append(encoder_outputs)
            self.ve_backward_tensors.append(ve_backward_outputs)
            assert group_idx == len(self.ve_llm_input_fake_masks_for_backward), (
                f"rank {dist.get_rank()} saw unexpected VE forward group order: "
                f"group_idx={group_idx}, cached_groups={len(self.ve_llm_input_fake_masks_for_backward)}"
            )
            self.ve_llm_input_fake_masks_for_backward.append(llm_input_fake_flags)

        if forward_contains_fake:
            llm_inputs = [
                llm_input for llm_input, is_fake in zip(llm_inputs, llm_input_fake_flags)
                if not is_fake
            ]

        if len(llm_inputs) == 0:
            raise RuntimeError(f"rank {dist.get_rank()} got all-fake VE encoder outputs for microbatch_id={microbatch_id}, cannot proceed with empty LLM inputs")

        # llm_inputs are referenced twice: once by the per-chunk iterator that
        # feeds LLM forward, and once by first_pp_inputs for VE backward rebuild.
        # Both owners must release them to avoid keeping the graph alive.
        is_first_stage = self.is_first_stage(op)
        first_stage_only_fields = {
            meta.name
            for meta in self.encoder_output_metas
            if getattr(meta, "only_requires_in_first_stage", False)
        }
        for vpp_rank in range(self.vpp_size):
            for i in range(len(llm_inputs)):
                if is_first_stage:
                    # chunk 0 uses the original tensors — .grad will be read by
                    # encoder_send_backward_action for VE backward.
                    self.llm_data_iterators[vpp_rank].put(llm_inputs[i])
                else:
                    # Other VPP chunks get detached copies so their backward
                    # passes don't accumulate into the original tensor's .grad.
                    # On non-first PP stages, drop fields that are marked as
                    # only needed by first stage.
                    detached = {}
                    for name, val in llm_inputs[i].items():
                        if name in first_stage_only_fields: # do not add data tagged with only_requires_in_first_stage to iterator when we are not in the first stage
                            detached[name] = None
                        elif isinstance(val, torch.Tensor):
                            detached[name] = val.detach()
                        else:
                            detached[name] = val
                    self.llm_data_iterators[vpp_rank].put(detached)

        for i in range(len(llm_inputs)):
            if is_first_stage:
                self.first_pp_inputs.append(llm_inputs[i])
            else:
                self.first_pp_inputs.append(None)

        self.ve_send_forward_handles.append(send_handles)
        self.ve_send_forward_keepalives.append(send_keepalive)
        self.ve_send_forward_ready_events.append(send_ready_event)
        self.ve_forward_release_tensors.append(encoder_outputs)
    
    def encoder_send_backward_action(self, op: ScheduleOp):
        # Pull gradients from the original first-stage LLM inputs so VE backward
        # sees grads in encoder-output order instead of LLM-input dict order.
        microbatch_id = op.microbatch_id
        assert self.is_first_stage(op)
        encoder_outputs = self.first_pp_inputs[microbatch_id]
        encoder_output_grads = []
        for meta in self.encoder_output_metas:
            name = meta.name
            encoder_output = encoder_outputs[name]
            if meta.requires_grad:
                grad = encoder_output.grad
                encoder_output.grad = None
                assert grad is not None
            else:
                grad = None
            encoder_output_grads.append(grad)

        def _release_nested_tensors_inplace(value):
            if isinstance(value, torch.Tensor):
                release_tensor(value)
                return None
            if isinstance(value, dict):
                for k in list(value.keys()):
                    value[k] = _release_nested_tensors_inplace(value[k])
                return value
            if isinstance(value, list):
                for i in range(len(value)):
                    value[i] = _release_nested_tensors_inplace(value[i])
                return value
            if isinstance(value, tuple):
                # Tuples are immutable; release tensor storages recursively.
                for item in value:
                    _release_nested_tensors_inplace(item)
                return value
            return value

        # Explicitly release tensor storages to avoid accidental long-lived
        # references keeping full LLM input memory alive.
        _release_nested_tensors_inplace(encoder_outputs)
        self.first_pp_inputs[microbatch_id] = None  # Release cached LLM inputs once grads are harvested.
        self.first_pp_input_grads.append(encoder_output_grads)

    def encoder_recv_forward_action(self, op: ScheduleOp):
        pass
        

    def encoder_recv_backward_action(self, op: ScheduleOp):
        microbatch_id = op.microbatch_id
        if microbatch_id < 0:
            llm_input_fake_flags = self.ve_fake_llm_input_fake_masks_for_backward.pop(0)
        else:
            assert microbatch_id % self.pp_size == self.pp_rank
            group_idx = microbatch_id // self.pp_size
            llm_input_fake_flags = self.ve_llm_input_fake_masks_for_backward[group_idx]
            self.ve_llm_input_fake_masks_for_backward[group_idx] = None

        if self.is_first_stage(op):
            real_encoder_output_grads = []
            for is_fake in llm_input_fake_flags:
                if is_fake:
                    continue

                grad_idx = self.first_pp_input_grads_cursor
                assert grad_idx < len(self.first_pp_input_grads), (
                    f"rank {dist.get_rank()} ran out of cached LLM grads while rebuilding VE grads: "
                    f"microbatch_id={microbatch_id}, grad_idx={grad_idx}, "
                    f"cached_grad_count={len(self.first_pp_input_grads)}"
                )
                real_grads = self.first_pp_input_grads[grad_idx]
                assert real_grads is not None, (
                    f"rank {dist.get_rank()} expected LLM grads at index {grad_idx} "
                    f"for VE microbatch_id={microbatch_id}, but found None"
                )
                real_encoder_output_grads.append(real_grads)
                self.first_pp_input_grads[grad_idx] = None
                self.first_pp_input_grads_cursor += 1

            encoder_output_grads = []
            real_grad_idx = 0
            for is_fake in llm_input_fake_flags:
                if is_fake:
                    # Fake slots never entered the LLM, so we synthesize zero grads
                    # to keep the VE backward reshard layout perfectly aligned.
                    encoder_output_grads.append(
                        _create_fake_llm_input_grads(
                            self.encoder_output_metas,
                            torch.device("cuda", torch.cuda.current_device()),
                        )
                    )
                else:
                    assert real_grad_idx < len(real_encoder_output_grads), (
                        f"rank {dist.get_rank()} rebuilt fewer real grads than expected "
                        f"for VE microbatch_id={microbatch_id}"
                    )
                    encoder_output_grads.append(real_encoder_output_grads[real_grad_idx])
                    real_grad_idx += 1
        else:
            encoder_output_grads = [None] * len(llm_input_fake_flags)
        encoder_output_grads, scatter_handles, backward_ready_event = self.encoder_backward_reshard(encoder_output_grads, op)

        if microbatch_id < 0:
            self.ve_fake_send_backward_handles.append(scatter_handles)
            self.ve_fake_send_backward_ready_events.append(backward_ready_event)
            self.ve_fake_output_tensor_grads.append(encoder_output_grads)
            return

        self.ve_send_backward_handles.append(scatter_handles)
        self.ve_send_backward_ready_events.append(backward_ready_event)
        self.ve_output_tensor_grads.append(encoder_output_grads)
    

    def encoder_backward_reshard(self, encoder_output_grads, op):
        assert len(encoder_output_grads) == self.pp_size
        grads_len = len(self.encoder_output_metas)
        output_tensor_grads = []
        scatter_handles = []
        for i in range(grads_len):
            meta = self.encoder_output_metas[i]
            if meta.requires_grad:
                if self.is_first_stage(op):
                    scatter_list = [encoder_output_grads[j][i] for j in range(self.pp_size)]
                else:
                    scatter_list = None
                output_tensor_grad = torch.empty(meta.shape, device=torch.cuda.current_device(), dtype=meta.dtype)
                src_rank = dist.get_global_rank(self.pp_group, 0)
                handle = dist.scatter(output_tensor_grad, scatter_list, src=src_rank, group=self.pp_group, async_op=True)
                output_tensor_grads.append(output_tensor_grad)
                if handle is not None:
                    scatter_handles.append(handle)
            else:
                output_tensor_grads.append(None)
        return output_tensor_grads, scatter_handles, None
        
    def _finish_ve_forward_handoff(self, microbatch_id):
        if microbatch_id < 0:
            return
        group_idx = microbatch_id // self.pp_size
        if group_idx >= len(self.ve_send_forward_handles):
            return

        handles = self.ve_send_forward_handles[group_idx]
        for handle in handles:
            handle.wait()

        ready_event = self.ve_send_forward_ready_events[group_idx]
        if ready_event is not None:
            torch.cuda.current_stream().wait_event(ready_event)

        self.ve_send_forward_handles[group_idx] = []
        self.ve_send_forward_keepalives[group_idx] = None
        self.ve_send_forward_ready_events[group_idx] = None

        # Once the local PP stage reaches the point where this VE wave can be
        # consumed, the resharded tensors are owned by the local LLM path and
        # the comm keepalives / original encoder output storage can be released.
        encoder_outputs = self.ve_forward_release_tensors[group_idx]
        self.ve_forward_release_tensors[group_idx] = None
        if encoder_outputs is not None:
            for i, meta in enumerate(self.encoder_output_metas):
                encoder_output = encoder_outputs[i]
                if encoder_output is not None and meta.requires_grad:
                    release_tensor(encoder_output)

    def recv_forward_action(self, op: ScheduleOp):
        self._finish_ve_forward_handoff(op.microbatch_id)
        
        chunk_id = op.chunk_id
        assert isinstance(self.tensor_shapes, list) and len(self.tensor_shapes) == 1, "tensor_shapes must be a list with length 1 for VPP recv_forward"
        recv_prev = not self.is_first_stage(op)
        input_tensor, wait_handles = self.p2p_communicator.send_forward_recv_forward(
            None,
            recv_prev=recv_prev,
            tensor_shape=self.tensor_shapes[0],
            overlap_p2p_comm=True,
        )
        if recv_prev and isinstance(wait_handles, dict):
            self.recv_forward_handles[chunk_id][op.microbatch_id] = wait_handles.get("recv_prev")
        self.input_tensors[chunk_id].append([input_tensor])

    def forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        assert chunk_id is not None, "chunk_id is not set"
        if op.microbatch_id in self.recv_forward_handles[chunk_id]:
            recv_handle = self.recv_forward_handles[chunk_id].pop(op.microbatch_id)
            if recv_handle is not None:
                recv_handle.wait()
        if self.is_first_stage(op):
            onload_done = self.llm_input_onload_done_events.pop(op.microbatch_id, None)
            if onload_done is not None:
                torch.cuda.current_stream().wait_event(onload_done)
        llm_data_iterator = self.llm_data_iterators[chunk_id]
        if self.is_first_stage(op):
            input_tensor = [None]
        else:
            input_tensor = self.input_tensors[chunk_id][op.microbatch_id]
        
        with self._execute_compute_with_trace(op):
            output_tensor, num_tokens = _forward_step_compat(
                self.forward_step_func,
                llm_data_iterator,
                self.model[chunk_id],
                self.num_microbatches,
                input_tensor,
                self.forward_data_store,
                self.config,
                cp_group_size=self.grad_finalize_pgs.cp.size(),
                collect_non_loss_data=self.collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=op.microbatch_id == 0,
                current_microbatch=op.microbatch_id,
                vp_stage=chunk_id,
                is_last_stage=is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id == self.vpp_size - 1,
            )
        self._accumulate_num_tokens(num_tokens)
        self.output_tensors[chunk_id].append(output_tensor)

    def send_forward_recv_forward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2

        chunk_id1 = op1.chunk_id
        assert chunk_id1 is not None, "chunk_id is not set"
        chunk_id2 = op2.chunk_id
        assert chunk_id2 is not None, "chunk_id is not set"
        output_tensor = self.output_tensors[chunk_id1][op1.microbatch_id]

        if not isinstance(output_tensor, list):
            assert isinstance(output_tensor, torch.Tensor), f"expect output_tensor to be a list or a tensor, but got {output_tensor}"
            output_tensor = [output_tensor]

        assert isinstance(output_tensor, list) and len(output_tensor) == 1, f"output_tensor must be a list with length 1 for now, but got {output_tensor}"
        assert isinstance(self.tensor_shapes, list) and len(self.tensor_shapes) == 1, f"tensor_shapes must be a list with length 1 for now, but got {self.tensor_shapes}"

        # Keep one in-flight overlapped send; drain the previous one now.
        self._drain_pending_send_forward()
        
        input_tensor, wait_handles = self.p2p_communicator.send_forward_recv_forward(
            output_tensor[0], 
            recv_prev=not (is_pp_first_stage(self.p2p_communicator.pp_group) and chunk_id2 == 0),
            tensor_shape=self.tensor_shapes[0],
            overlap_p2p_comm=True
        )
        # Defer send completion/deallocation by one send to increase overlap.
        send_handle = wait_handles.get("send_next") if isinstance(wait_handles, dict) else None
        if send_handle is None:
            deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)
        else:
            self.send_forward_handles = (
                send_handle,
                output_tensor[0],
                chunk_id1,
                op1.microbatch_id,
            )
        if isinstance(wait_handles, dict) and 'recv_prev' in wait_handles:
            self.recv_forward_handles[chunk_id2][op2.microbatch_id] = wait_handles.get('recv_prev')

        # Mirror recv_forward_action for fused receive paths that do not
        # execute a standalone recv_forward_action before consuming the wave.
        self._finish_ve_forward_handoff(op2.microbatch_id)
        input_tensor = [input_tensor]
        self.input_tensors[chunk_id2].append(input_tensor)


class LLMVPPExecutorWithDecoder(LLMVPPExecutorWithVE):
    
    def __init__(self, 
                forward_step_func, 
                data_iterator,
                model,
                num_microbatches,
                seq_length,
                micro_batch_size,
                forward_only,
                pp_size,
                vpp_size,
                pp_group,
                pp_rank,
                collect_non_loss_data: bool = False,
                get_tensor_shapes_fn: callable = None,
                encoder_models: List[torch.nn.Module] = [],
                encoder_forward_func: callable = None,
                get_encoder_output_meta_func: callable = None,
                decoder_models: List[torch.nn.Module] = [],
                decoder_forward_func: callable = None,
                get_decoder_input_meta_func: callable = None,
                trace_enabled: bool = False,
                trace_dir: str = None,
                trace_iter_id: int = None):
        super().__init__(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            forward_only,
            pp_size,
            vpp_size,
            pp_group,
            pp_rank,
            collect_non_loss_data,
            get_tensor_shapes_fn,
            encoder_models,
            encoder_forward_func,
            get_encoder_output_meta_func,
            trace_enabled=trace_enabled,
            trace_dir=trace_dir,
            trace_iter_id=trace_iter_id,
        )
        
        self.decoder_models = decoder_models
        self.decoder_forward_func = decoder_forward_func
        self.get_decoder_input_meta_func = get_decoder_input_meta_func

        self.scheduler = UnifiedBigMacVPP(pp_size, vpp_size, num_microbatches)
        self.scheduler.generate_schedule()
        if self.rebuild_grad_buffer and hasattr(self.scheduler, "add_encoder_grad_buffer_ops"):
            self.scheduler.add_encoder_grad_buffer_ops()
        self.scheduler.add_communication_op()
        self.scheduler.optimize_communication_ops()
        self.schedule_plan = self.scheduler.schedules_with_comm[self.pp_rank]

        self.decoder_data_iterator = LLMDataIterator()

        self.register_actions_for_decoder()

        if get_decoder_input_meta_func is not None:
            self.decoder_input_metas = get_decoder_input_meta_func()
        else:
            raise NotImplementedError("automatically generate decoder input meta is not supported currently")
    
    def reset(self):
        super().reset()
        self.decoder_input_tensors = []
        self.decoder_output_tensors = []
        self.decoder_input_tensor_grads = []
        self.decoder_output_tensor_grads = []
        self.decoder_send_forward_handles = []
        self.decoder_send_backward_handles = []
        self.decoder_recv_forward_handles = []
        self.decoder_recv_backward_handles = []
    
    def register_actions_for_decoder(self):
        self.register_action(OpType.GENFORWARD, self.decoder_forward_action)
        self.register_action(OpType.GENBACKWARD, self.decoder_backward_action)
        self.register_action(OpType.GENSENDFORWARD, self.decoder_send_forward_action)
        self.register_action(OpType.GENRECVFORWARD, self.decoder_recv_forward_action)
        self.register_action(OpType.GENSENDBACKWARD, self.decoder_send_backward_action)
        self.register_action(OpType.GENRECVBACKWARD, self.decoder_recv_backward_action)
    
    
    def calc_num_grad_acc(self):
        return self.num_microbatches // self.pp_size
    
    def forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        parallel_state.set_virtual_pipeline_model_parallel_rank(chunk_id)
        assert chunk_id is not None, "chunk_id is not set"
        if op.microbatch_id in self.recv_forward_handles[chunk_id]:
            recv_handle = self.recv_forward_handles[chunk_id].pop(op.microbatch_id)
            if recv_handle is not None:
                recv_handle.wait()
        if self.is_first_stage(op):
            onload_done = self.llm_input_onload_done_events.pop(op.microbatch_id, None)
            if onload_done is not None:
                torch.cuda.current_stream().wait_event(onload_done)
        llm_data_iterator = self.llm_data_iterators[chunk_id]
        if self.is_first_stage(op):
            input_tensor = [None]
        else:
            input_tensor = self.input_tensors[chunk_id][op.microbatch_id]
        
        with self._execute_compute_with_trace(op):
            output_tensor, num_tokens = _forward_step_compat(
                self.forward_step_func,
                llm_data_iterator,
                self.model[chunk_id],
                self.num_microbatches,
                input_tensor,
                self.forward_data_store,
                self.config,
                cp_group_size=self.grad_finalize_pgs.cp.size(),
                collect_non_loss_data=self.collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=op.microbatch_id == 0,
                current_microbatch=op.microbatch_id,
                vp_stage=chunk_id,
                is_last_stage=False,  # Loss is produced by the decoder stage, never by the LLM stage.
            )
        self._accumulate_num_tokens(num_tokens)
        if isinstance(output_tensor, list) and len(output_tensor) == 1 and isinstance(output_tensor[0], list):
            output_tensor = output_tensor[0]
        self.output_tensors[chunk_id].append(output_tensor)
    
    def send_forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        assert chunk_id is not None, "chunk_id is not set"
        output_tensor = self.output_tensors[chunk_id][op.microbatch_id]
        self.p2p_communicator.send_forward(output_tensor, is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id == self.vpp_size - 1)
        if is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id == self.vpp_size - 1:
            return  # Keep final hidden states alive until the decoder-side scatter consumes them.
        deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs) 
    
    def decoder_forward_action(self, op: ScheduleOp):
        decoder_output, loss_dict = self.decoder_forward_func(self.decoder_models, self.decoder_data_iterator)
        assert torch.is_tensor(decoder_output) and decoder_output.ndim == 0, f"decoder output should be a scalar tensor, but got {decoder_output}"
        # Normalize loss by the number of accumulation steps because decoder
        # backward is launched once per VE wave, not once per token sample.
        decoder_output /= self.calc_num_grad_acc()
        self.forward_data_store.append(loss_dict)
        self.decoder_output_tensors.append(decoder_output)
    
    def recv_backward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        micro_batch_id = op.microbatch_id
        assert chunk_id is not None, "chunk_id is not set"
        output_tensor_grad = self.p2p_communicator.recv_backward(
            self.tensor_shapes, is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id == self.vpp_size - 1
        )
        if not (is_pp_last_stage(self.p2p_communicator.pp_group) and chunk_id == self.vpp_size - 1):
            self.output_tensor_grads[chunk_id].append(output_tensor_grad)

    def decoder_backward_action(self, op: ScheduleOp):
        microbatch_id = op.microbatch_id
        decoder_output = self.decoder_output_tensors[microbatch_id // self.pp_size]
        decoder_output_grad = torch.ones_like(decoder_output, memory_format=torch.preserve_format)
        
        # Calculate total number of VE backward operations
        # Each VE forward produces pp_size microbatches, so total VE backwards = num_microbatches / pp_size
        total_decoder_backwards = self.num_microbatches // self.pp_size
        current_decoder_backward_id = microbatch_id // self.pp_size
        is_last_decoder_microbatch = (current_decoder_backward_id == total_decoder_backwards - 1)
        
        # Set is_last_microbatch flag for the last VE microbatch
        if is_last_decoder_microbatch:
            for model in self.decoder_models:
                model.module.is_last_microbatch = True

        with _get_model_no_sync_context(
            self.decoder_models,
            is_last_microbatch=is_last_decoder_microbatch,
        ):
            Variable._execution_engine.run_backward(
                tensors=(decoder_output,),
                grad_tensors=(decoder_output_grad,),
                keep_graph=False,
                create_graph=False,
                inputs=tuple(),
                allow_unreachable=True,
                accumulate_grad=True,
            )
        
        # Reset is_last_microbatch flag after backward
        if is_last_decoder_microbatch:
            for model in self.decoder_models:
                model.module.is_last_microbatch = False

        self.decoder_output_tensors[microbatch_id // self.pp_size] = None

    def decoder_send_forward_action(self, op: ScheduleOp):
        pass

    def decoder_recv_forward_action(self, op: ScheduleOp):
        chunk_id = op.chunk_id
        microbatch_id = op.microbatch_id
        data_len = len(self.decoder_input_metas)

        if self.pp_rank == self.pp_size - 1:
            # The last PP rank has the full set of generator inputs for one VE
            # wave and scatters the per-rank slices to all PP ranks.
            start_idx = microbatch_id // self.pp_size * self.pp_size
            end_idx = start_idx + self.pp_size
            assert len(self.output_tensors[chunk_id][start_idx:end_idx]) == self.pp_size
            to_scatter = [[] for _ in range(data_len)]
            for i in range(data_len):  # Iterate over decoder input fields.
                meta = self.decoder_input_metas[i]
                for j in range(self.pp_size):  # Gather one field across PP ranks.
                    to_scatter[i].append(self.output_tensors[chunk_id][start_idx + j][i])
                    self.check_internal(to_scatter[i][j], meta)
        else:
            to_scatter = [[None] * self.pp_size for _ in range(data_len)]
        
        scattered, handles = scatter_along_pp(to_scatter, self.pp_group, self.pp_size - 1, self.decoder_input_metas)

        for handle in handles:
            handle.wait()
        # After scatter, the source rank no longer needs to keep the last-stage
        # LLM outputs that were only serving as decoder inputs.
        if self.pp_rank == self.pp_size - 1:
            for i in range(data_len):  # Iterate over decoder input fields.
                meta = self.decoder_input_metas[i]
                for j in range(self.pp_size):  # Release the per-rank slice for this field.
                    if isinstance(self.output_tensors[chunk_id][start_idx + j][i], torch.Tensor) and meta.requires_grad:
                        deallocate_output_tensor(self.output_tensors[chunk_id][start_idx + j][i], self.config.deallocate_pipeline_outputs)
        
        self.decoder_input_tensors.append(scattered)
        self.decoder_data_iterator.put(scattered)
        

    def decoder_send_backward_action(self, op: ScheduleOp):
        microbatch_id = op.microbatch_id
        idx = microbatch_id // self.pp_size
        input_tensors = list(self.decoder_input_tensors[idx].values())
        input_tensor_grads = []
        assert len(input_tensors) == len(self.decoder_input_metas)
        for i in range(len(self.decoder_input_metas)):
            meta = self.decoder_input_metas[i]
            if meta.requires_grad:
                assert input_tensors[i].grad is not None, f"input tensor grad is not set for {meta.name}"
                input_tensor_grads.append(input_tensors[i].grad)
            else:
                input_tensor_grads.append(None)
        self.decoder_input_tensors[idx] = None
        gathered_datas, handles = all_gather_along_pp(input_tensor_grads, self.pp_group, self.decoder_input_metas, comm_requires_grad_only=True)
        for handle in handles:
            handle.wait()
        if self.pp_rank == self.pp_size - 1:
            output_tensor_grads = [list(gathered_data.values()) for gathered_data in gathered_datas]
            output_tensors = self.output_tensors[self.vpp_size - 1]
            self.output_tensor_grads[self.vpp_size - 1].extend(output_tensor_grads)
    
    def decoder_recv_backward_action(self, op: ScheduleOp):
        pass
