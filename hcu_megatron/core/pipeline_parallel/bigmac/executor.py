# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

from dataclasses import dataclass
from typing import Optional, Iterator, List, Union
import torch
import contextlib
import gc
import inspect
import pickle
import time
import torch.distributed as dist
from collections import deque
from types import SimpleNamespace
from .schedules import (
    PPSchedule,
    Schedule1F1B,
    VPP1F1BSchedule,
    OpType,
    ScheduleOp,
    BigMac,
    FusedScheduleOp,
)
from . import visualize_schedule
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
from .communication import (
    all_gather_along_pp,
    scatter_along_pp,
    release_model_grad_buffer,
    release_tensor,
    rebuild_model_grad_buffer,
)
from .trace import (
    build_chrome_trace_events,
    export_chrome_trace,
    export_chrome_trace_events,
    mark_op_trace,
    mark_op_trace_events,
)


@dataclass(frozen=True)
class EncoderOutputMeta:
    # Pipeline handoff metadata. A -1 shape dimension is exchanged per encoder unit.

    name: str
    is_none: bool = False
    is_object: bool = False
    requires_grad: Optional[bool] = False
    seq_dim: Optional[int] = 0
    shape: Optional[tuple] = None
    dtype: Optional[torch.dtype] = None
    only_requires_in_first_stage: bool = False


class LLMDataIterator:
    # Small FIFO iterator used to feed microbatch payloads into Megatron steps.

    def __init__(self):
        self.queue = deque()

    def put(self, data):
        self.queue.append(data)

    def get(self):
        return self.queue.popleft()

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()


try:
    from megatron.core.process_groups_config import ProcessGroupCollection
except ImportError:
    class ProcessGroupCollection:
        pass


try:
    from megatron.core.pipeline_parallel.schedules import (
        P2PCommunicator,
        is_pp_last_stage,
        is_pp_first_stage,
    )
except ImportError:
    from megatron.core.pipeline_parallel import p2p_communication

    class P2PCommunicator:
        def __init__(self, pp_group, config):
            self.pp_group = pp_group
            self.config = config

        @staticmethod
        def _as_list(value):
            if isinstance(value, list):
                return value
            return [value]

        def send_forward(self, output_tensor, is_last_stage):
            for tensor in self._as_list(output_tensor):
                p2p_communication.send_forward(tensor, self.config, is_last_stage)

        def send_backward(self, input_tensor_grad, is_first_stage):
            for tensor_grad in self._as_list(input_tensor_grad):
                p2p_communication.send_backward(tensor_grad, self.config, is_first_stage)

        def recv_forward(self, tensor_shapes, is_first_stage):
            input_tensors = []
            for tensor_shape in self._as_list(tensor_shapes):
                input_tensors.append(
                    None
                    if tensor_shape is None
                    else p2p_communication.recv_forward(
                        tensor_shape,
                        self.config,
                        is_first_stage,
                    )
                )
            return input_tensors

        def recv_backward(self, tensor_shapes, is_last_stage):
            output_tensor_grads = []
            for tensor_shape in self._as_list(tensor_shapes):
                output_tensor_grads.append(
                    None
                    if tensor_shape is None
                    else p2p_communication.recv_backward(
                        tensor_shape,
                        self.config,
                        is_last_stage,
                    )
                )
            return output_tensor_grads

        def send_forward_recv_backward(self, output_tensor, tensor_shapes, is_last_stage):
            output_tensor_grads = []
            for tensor, tensor_shape in zip(
                self._as_list(output_tensor),
                self._as_list(tensor_shapes),
            ):
                output_tensor_grads.append(
                    None
                    if tensor_shape is None
                    else p2p_communication.send_forward_recv_backward(
                        tensor, tensor_shape, self.config, is_last_stage
                    )
                )
            return output_tensor_grads

        def send_backward_recv_forward(self, input_tensor_grad, tensor_shapes, is_first_stage):
            input_tensors = []
            for tensor_grad, tensor_shape in zip(
                self._as_list(input_tensor_grad), self._as_list(tensor_shapes)
            ):
                input_tensors.append(
                    None
                    if tensor_shape is None
                    else p2p_communication.send_backward_recv_forward(
                        tensor_grad, tensor_shape, self.config, is_first_stage
                    )
                )
            return input_tensors

    def is_pp_last_stage(pp_group=None):
        return parallel_state.is_pipeline_last_stage()

    def is_pp_first_stage(pp_group=None):
        return parallel_state.is_pipeline_first_stage()


def _finalize_model_grads_compat(model, num_tokens, pg_collection):
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads as fallback
    config = get_model_config(model[0])
    finalizer = getattr(config, "finalize_model_grads_func", fallback)
    if finalizer is None:
        return
    signature = inspect.signature(finalizer)
    if "pg_collection" in signature.parameters:
        return finalizer(model, num_tokens=num_tokens, pg_collection=pg_collection)
    return finalizer(model, num_tokens=num_tokens)


def _call_with_supported_kwargs(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(*args, **kwargs)
    supported_kwargs = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return fn(*args, **supported_kwargs)


def _forward_step_compat(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    **kwargs,
):
    from megatron.core.pipeline_parallel import schedules
    return _call_with_supported_kwargs(
        schedules.forward_step,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        **kwargs,
    )


def _backward_step_compat(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    from megatron.core.pipeline_parallel.schedules import backward_step
    signature = inspect.signature(backward_step)
    if "model_type" in signature.parameters:
        return backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
    return backward_step(input_tensor, output_tensor, output_tensor_grad, config)


def _new_total_num_tokens():
    return torch.zeros([], dtype=torch.int, device=torch.cuda.current_device())


def _get_tensor_shapes_compat(
    get_tensor_shapes_fn,
    *,
    seq_length,
    micro_batch_size,
    decoder_seq_length,
    config,
    model_type,
):
    return _call_with_supported_kwargs(
        get_tensor_shapes_fn,
        rank=parallel_state.get_pipeline_model_parallel_rank(),
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=False,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        cp_group=parallel_state.get_context_parallel_group(),
    )


def _run_encoder_backward(executor, encoder_outputs, encoder_output_grads, ve_backward_outputs=None):
    del ve_backward_outputs
    tensors = []
    grad_tensors = []
    for encoder_output, encoder_output_grad in zip(encoder_outputs, encoder_output_grads):
        if isinstance(encoder_output, torch.Tensor) and encoder_output.requires_grad:
            tensors.append(encoder_output)
            grad_tensors.append(encoder_output_grad)
    if not tensors:
        return
    with _get_model_no_sync_context(
        getattr(executor, "encoder_models", None),
        is_last_microbatch=getattr(executor, "_current_encoder_backward_is_last_microbatch", True),
    ):
        Variable._execution_engine.run_backward(
            tensors=tuple(tensors),
            grad_tensors=tuple(grad_tensors),
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )


def _get_model_no_sync_context(models, is_last_microbatch: bool):
    if is_last_microbatch or models is None:
        return contextlib.nullcontext()

    if not isinstance(models, (list, tuple)):
        models = [models]

    no_sync_contexts = []
    for model in models:
        if (
            hasattr(model, "no_sync")
            and hasattr(model, "ddp_config")
            and getattr(model.ddp_config, "overlap_grad_reduce", False)
        ):
            no_sync_contexts.append(model.no_sync())

    if not no_sync_contexts:
        return contextlib.nullcontext()

    stack = contextlib.ExitStack()
    for no_sync_context in no_sync_contexts:
        stack.enter_context(no_sync_context)
    return stack


def _should_empty_cache_before_grad_finalize(executor) -> bool:
    owner = getattr(getattr(executor, "encoder_forward_func", None), "__self__", None)
    if owner is None or not hasattr(owner, "debug_index"):
        return False
    return getattr(owner, "debug_index") == 1


def _create_fake_llm_input_grads(encoder_output_metas, device):
    fake_grads = []
    for meta in encoder_output_metas:
        if not meta.requires_grad:
            fake_grads.append(None)
            continue
        fake_grads.append(
            torch.zeros(meta.shape, dtype=meta.dtype, device=device)
        )
    return fake_grads


class _ExecuteComputeWithTrace:
    def __init__(self, executor, op):
        self.executor = executor
        self.op = op
        self.start_event = None
        self.end_event = None
        self.start_us = None
        self.trace_enabled = False
        self.use_cuda_events = False

    def __enter__(self):
        self.trace_enabled = self.executor.trace_enabled
        if not self.trace_enabled:
            return self

        self.use_cuda_events = self.executor._trace_base_event is not None
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record(torch.cuda.current_stream())
        else:
            self.start_us = self.executor._trace_now_us()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.trace_enabled:
            return False

        if self.use_cuda_events:
            self.end_event.record(torch.cuda.current_stream())
            mark_op_trace_events(
                self.op,
                self.executor._trace_base_event,
                self.start_event,
                self.end_event,
                self.executor._trace_origin_us,
            )
        else:
            end_us = self.executor._trace_now_us()
            mark_op_trace(self.op, self.start_us, end_us)
        return False


class LLMPPExecutor:
    def __init__(self, 
                forward_step_func, 
                data_iterator,
                model,
                num_microbatches,
                seq_length,
                micro_batch_size,
                forward_only,
                pp_size,
                pp_group,
                pp_rank,
                collect_non_loss_data: bool = False,
                get_tensor_shapes_fn: callable = None,
                trace_enabled: bool = False,
                trace_dir: str = None,
                trace_iter_id: int = None):
        # Args:
        #     forward_step_func: The function to execute the forward step.
        #     data_iterator: The data iterator.
        #     model: model.
        #     num_microbatches: number of microbatches.
        #     seq_length: llm sequence length.
        #     micro_batch_size: micro batch size.
        #     forward_only: Whether to execute the forward only.
        #     pp_size: pipeline model parallel size.
        assert not forward_only, "validation is not supported currently"

        if isinstance(model, list):
            assert (
                len(model) == 1
            ), "LLM PP Executor does not support model chunking"
            model = model[0]
        if isinstance(data_iterator, list):
            assert (
                len(data_iterator) == 1
            ), "LLM PP Executor does not support model chunking"
            data_iterator = data_iterator[0]
        
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
        self.config = get_model_config(self.model)
        self.model_type = get_model_type(model)
        # assert os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] == '1'

        self.p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=self.config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        if cp_group.size() != 1:
            raise NotImplementedError("This open-source pp_executor path supports single context group execution only.")
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

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

        self.scheduler = Schedule1F1B(pp_size, num_microbatches)
        self.scheduler.generate_schedule()
        self.scheduler.add_communication_op()
        self.scheduler.optimize_communication_ops()
        self.scheduler.fuse_comm_ops()
        self.schedule_plan = self.scheduler.schedules_with_comm[self.pp_rank]

        self.action_map = {}
        self.forward_data_store = []
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
        self.debug = False

    
    def reset(self):
        self.llm_data_iterator = LLMDataIterator()
        self.forward_data_store = []
        self.total_num_tokens = _new_total_num_tokens()
        self.input_tensors = []
        self.output_tensors = []
        self.output_tensor_grads = []
        self.input_tensor_grads = []

    def _accumulate_num_tokens(self, num_tokens):
        if num_tokens is not None:
            self.total_num_tokens += num_tokens

    def print_dp_0(self, msg):
        if not self.debug:
            return
        from megatron.core import mpu
        if mpu.get_data_parallel_rank() == 0:
            print(f"[pp_rank {self.pp_rank}] {msg}")

    def _trace_now_us(self):
        if self._trace_origin_us is None or self._trace_origin_perf_ns is None:
            return time.time_ns() // 1000
        elapsed_us = (time.perf_counter_ns() - self._trace_origin_perf_ns) // 1000
        return self._trace_origin_us + elapsed_us

    def _init_trace_base_event(self):
        if torch.cuda.is_available():
            self._trace_base_event = torch.cuda.Event(enable_timing=True)
            self._trace_base_event.record(torch.cuda.current_stream())

    def _sync_trace_clock(self):
        if not self.trace_enabled:
            return
        if not (dist.is_available() and dist.is_initialized()):
            self._trace_origin_us = time.time_ns() // 1000
            self._trace_origin_perf_ns = time.perf_counter_ns()
            self._init_trace_base_event()
            return

        dist.barrier(group=self.pp_group)
        origin = [time.time_ns() // 1000 if self.pp_rank == 0 else 0]
        try:
            src_rank = dist.get_global_rank(self.pp_group, 0)
        except AttributeError:
            src_rank = 0
        dist.broadcast_object_list(origin, src=src_rank, group=self.pp_group)
        dist.barrier(group=self.pp_group)
        self._trace_origin_us = origin[0]
        self._trace_origin_perf_ns = time.perf_counter_ns()
        self._init_trace_base_event()

    def _execute_op_with_trace(self, op):
        action = self.get_action(op)
        if getattr(op, "op_type", None) in (OpType.FORWARD, OpType.BACKWARD):
            action(op)
            return

        if self._trace_base_event is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            try:
                action(op)
            finally:
                end_event.record(torch.cuda.current_stream())
                mark_op_trace_events(
                    op,
                    self._trace_base_event,
                    start_event,
                    end_event,
                    self._trace_origin_us,
                )
            return

        start_us = self._trace_now_us()
        try:
            action(op)
        finally:
            end_us = self._trace_now_us()
            mark_op_trace(op, start_us, end_us)

    def _execute_compute_with_trace(self, op):
        return _ExecuteComputeWithTrace(self, op)

    def _trace_phase(self, name):
        if not self.trace_enabled:
            return contextlib.nullcontext()
        op = SimpleNamespace(op_type=name, microbatch_id=None)
        self._trace_tail_ops.append(op)
        return self._execute_compute_with_trace(op)

    @staticmethod
    def _trace_rank_width(world_size: int) -> int:
        return max(1, len(str(max(0, world_size - 1))))

    def _trace_filename(self):
        iter_suffix = "" if self.trace_iter_id is None else f"_iter_{self.trace_iter_id}"
        if dist.is_available() and dist.is_initialized():
            global_rank_width = self._trace_rank_width(dist.get_world_size())
            return f"pp_group_rank_{dist.get_rank():0{global_rank_width}d}{iter_suffix}_trace.json"
        pp_rank_width = self._trace_rank_width(self.pp_size)
        return f"pp_rank_{self.pp_rank:0{pp_rank_width}d}{iter_suffix}_trace.json"

    def _export_trace(self):
        if not self.trace_enabled:
            return
        trace_ops = [*self.schedule_plan, *self._trace_tail_ops]
        pp_rank_width = self._trace_rank_width(self.pp_size)
        if dist.is_available() and dist.is_initialized():
            local_events = build_chrome_trace_events(
                trace_ops,
                self.pp_rank,
                pp_rank_width=pp_rank_width,
            )
            gathered_events = [None for _ in range(dist.get_world_size(self.pp_group))]
            dist.all_gather_object(gathered_events, local_events, group=self.pp_group)
            if self.pp_rank != 0:
                return
            merged_events = []
            for rank_events in gathered_events:
                merged_events.extend(rank_events or [])
            self.trace_output_path = export_chrome_trace_events(
                merged_events,
                self.trace_dir,
                self._trace_filename(),
            )
            print(f"[pp_executor] wrote trace: {self.trace_output_path}", flush=True)
            return
        self.trace_output_path = export_chrome_trace(
            trace_ops,
            self.trace_dir,
            self.pp_rank,
            filename=self._trace_filename(),
            pp_rank_width=pp_rank_width,
        )
        print(f"[pp_executor] wrote trace: {self.trace_output_path}", flush=True)
    
    def post_process(self):
        # post process to release memory
        del self.input_tensors
        del self.output_tensors
        del self.output_tensor_grads
        del self.input_tensor_grads
        del self.llm_data_iterator
    
    def get_action(self, op: ScheduleOp):
        return self.action_map[op.op_type]
    
    def register_action(self, op, action_func: callable):
        self.action_map[op] = action_func
    
    def register_actions_for_llm_pp(self):
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
    
    def default_action(self, op: ScheduleOp):
        if op.op_type not in (OpType.IDLE, OpType.NULLCOMMUNICATION):
            raise NotImplementedError(f"No BigMac executor action registered for {op.op_type}")
    
    def forward_action(self, op: ScheduleOp):
        if self.is_first_stage():
            input_tensor = [None]
        else:
            input_tensor = self.input_tensors[op.microbatch_id]
        
        with self._execute_compute_with_trace(op):
            output_tensor, num_tokens = _forward_step_compat(
                self.forward_step_func,
                self.llm_data_iterator,
                self.model,
                self.num_microbatches,
                input_tensor,
                self.forward_data_store,
                self.config,
                cp_group_size=self.grad_finalize_pgs.cp.size(),
                collect_non_loss_data=self.collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=op.microbatch_id == 0,
                current_microbatch=op.microbatch_id,
                is_last_stage=is_pp_last_stage(self.p2p_communicator.pp_group),
            )
        self._accumulate_num_tokens(num_tokens)
        self.output_tensors.append(output_tensor)
    
    def backward_action(self, op: ScheduleOp):
        input_tensor = self.input_tensors[op.microbatch_id]
        output_tensor = self.output_tensors[op.microbatch_id]
        output_tensor_grad = self.output_tensor_grads[op.microbatch_id]
        # Megatron DDP overlap_grad_reduce expects us to disable grad synchronization for
        # all non-last microbatches (gradient accumulation), otherwise the same parameter
        # will be "registered grad-ready" multiple times and trip:
        #   AssertionError: Cannot set grad twice
        is_last_microbatch = op.microbatch_id == (self.num_microbatches - 1)
        with _get_model_no_sync_context(
            self.model,
            is_last_microbatch=is_last_microbatch,
        ), self._execute_compute_with_trace(op):
            input_tensor_grad = _backward_step_compat(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                self.model_type,
                self.config,
            )
        self.input_tensor_grads.append(input_tensor_grad)
        self.input_tensors[op.microbatch_id] = None
        self.output_tensors[op.microbatch_id] = None
        self.output_tensor_grads[op.microbatch_id] = None
    
    def send_forward_action(self, op: ScheduleOp):
        output_tensor = self.output_tensors[op.microbatch_id]
        self.p2p_communicator.send_forward(output_tensor, is_pp_last_stage(self.p2p_communicator.pp_group))
        deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs) 
    
    def send_backward_action(self, op: ScheduleOp):
        input_tensor_grad = self.input_tensor_grads[op.microbatch_id]
        self.p2p_communicator.send_backward(
            input_tensor_grad, is_pp_first_stage(self.p2p_communicator.pp_group)
        )
        self.input_tensor_grads[op.microbatch_id] = None
    
    def recv_forward_action(self, op: ScheduleOp):
        if self.data_iterator:
            data = next(self.data_iterator)
        else:
            data = None
        self.llm_data_iterator.put(data)
        input_tensor = self.p2p_communicator.recv_forward(
            self.tensor_shapes, is_pp_first_stage(self.p2p_communicator.pp_group)
        )
        self.input_tensors.append(input_tensor)
    
    def recv_backward_action(self, op: ScheduleOp):
        output_tensor_grad = self.p2p_communicator.recv_backward(
            self.tensor_shapes, is_pp_last_stage(self.p2p_communicator.pp_group)
        )
        self.output_tensor_grads.append(output_tensor_grad)
    
    def send_forward_recv_backward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2        
        output_tensor = self.output_tensors[op1.microbatch_id]
        output_tensor_grad = self.p2p_communicator.send_forward_recv_backward(output_tensor, self.tensor_shapes, is_pp_last_stage(self.p2p_communicator.pp_group))
        deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs) 
        self.output_tensor_grads.append(output_tensor_grad)
    
    def send_backward_recv_forward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2
        
        input_tensor_grad = self.input_tensor_grads[op1.microbatch_id]
        input_tensor = self.p2p_communicator.send_backward_recv_forward(input_tensor_grad, self.tensor_shapes, is_pp_first_stage(self.p2p_communicator.pp_group))
        self.input_tensor_grads[op1.microbatch_id] = None
        if self.data_iterator:
            data = next(self.data_iterator)
        else:
            data = None
        self.llm_data_iterator.put(data)
        self.input_tensors.append(input_tensor)

    
    def is_first_stage(self, op: ScheduleOp = None):
        if op is not None and getattr(op, 'chunk_id', None) is not None:
            return self.pp_rank == 0 and op.chunk_id == 0
        else:
            return self.pp_rank == 0
    
    def execute(self, ):
        self.reset()
        self._trace_tail_ops = []
        self._sync_trace_clock()
        
        for op in self.schedule_plan:
            self.print_dp_0(f"executing op: {op}")
            if self.trace_enabled:
                self._execute_op_with_trace(op)
            else:
                action = self.get_action(op)
                action(op)

        self.post_process()

        # Megatron DDP launches async grad reduce-scatter/all-reduce when
        # `overlap_grad_reduce=True` and clears the outstanding handle in
        # `finish_grad_sync()`. Our custom pipeline executor (`LLMPPExecutor`)
        # bypasses Megatron's standard forward-backward utilities, so we must
        # explicitly finalize grads (wait + extra reductions such as embeddings).
        if isinstance(self.model, list):
            model = self.model
        else:
            model = [self.model]
        if _should_empty_cache_before_grad_finalize(self):
            print(
                f"rank {dist.get_rank()} emptying CUDA cache before grad finalize "
                "as requested by encoder forward owner"
            )
            torch.cuda.empty_cache()
        owner = getattr(getattr(self, "encoder_forward_func", None), "__self__", None)
        if owner is not None and hasattr(owner, "prepare_encoder_grads"):
            with self._trace_phase("PP_VISION_GRAD_SUM"):
                owner.prepare_encoder_grads(self)
        num_tokens = self.total_num_tokens if self.config.calculate_per_token_loss else None
        with self._trace_phase("NATIVE_GRAD_FINALIZE"):
            _finalize_model_grads_compat(model, num_tokens=num_tokens, pg_collection=self.grad_finalize_pgs)
        self._export_trace()

        return self.forward_data_store


class LLMPPExecutorWithVE(LLMPPExecutor):
    def __init__(self, 
                forward_step_func, 
                data_iterator,
                model,
                num_microbatches,
                seq_length,
                micro_batch_size,
                forward_only,
                pp_size,
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
        super().__init__(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            forward_only,
            pp_size,
            pp_group,
            pp_rank,
            collect_non_loss_data,
            get_tensor_shapes_fn,
            trace_enabled=trace_enabled,
            trace_dir=trace_dir,
            trace_iter_id=trace_iter_id,
        )
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

        self.scheduler = BigMac(pp_size, num_microbatches)
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
        # Dedicated stream for LLM input offload/onload memcpy.
        self.offload_stream = torch.cuda.Stream()

    def reset(self):
        super().reset()
        self._last_ve_forward_contains_fake = False
        self._last_ve_forward_output_is_fake = False
        owner = getattr(self.encoder_forward_func, "__self__", None)
        if owner is not None and hasattr(owner, "reset_pp_executor_encoder_forward_state"):
            owner.reset_pp_executor_encoder_forward_state()
        self.ve_input_tensors = []
        self.ve_pending_output_tensors = []
        self.ve_pending_forward_contains_fake = []
        self.ve_pending_forward_output_is_fake = []
        self.ve_pending_backward_tensors = []
        self.ve_output_tensors = []
        self.ve_forward_release_tensors = []
        # Keep only fake masks for backward reconstruction to avoid retaining
        # whole llm_input dicts (e.g. input_embed) until encoder_recv_backward.
        self.ve_llm_input_fake_masks_for_backward = []
        self.ve_fake_llm_input_fake_masks_for_backward = []
        self.ve_backward_shapes = []
        self.ve_fake_backward_shapes = []
        self.ve_backward_tensors = []
        self.ve_fake_backward_tensors = []
        self.ve_input_tensor_grads = []
        self.ve_output_tensor_grads = []
        self.ve_fake_output_tensor_grads = []
        self.ve_send_forward_handles = []
        self.ve_send_forward_keepalives = []
        self.ve_send_forward_ready_events = []
        self.ve_send_backward_handles = []
        self.ve_send_backward_ready_events = []
        self.ve_fake_send_backward_ready_events = []
        self.ve_fake_send_backward_handles = []
        self.llm_input_offload_done_events = {}
        self.llm_input_onload_done_events = {}
        self.first_pp_inputs = []
        self.first_pp_input_grads = []
        self.first_pp_input_grads_cursor = 0
    
    def post_process(self):
        super().post_process()
        del self.ve_input_tensors
        del self.ve_pending_output_tensors
        del self.ve_pending_forward_contains_fake
        del self.ve_pending_forward_output_is_fake
        del self.ve_pending_backward_tensors
        del self.ve_output_tensors
        del self.ve_forward_release_tensors
        del self.ve_llm_input_fake_masks_for_backward
        del self.ve_fake_llm_input_fake_masks_for_backward
        del self.ve_backward_shapes
        del self.ve_fake_backward_shapes
        del self.ve_backward_tensors
        del self.ve_fake_backward_tensors
        del self.ve_input_tensor_grads
        del self.ve_output_tensor_grads
        del self.ve_fake_output_tensor_grads
        del self.ve_send_forward_handles
        del self.ve_send_forward_keepalives
        del self.ve_send_forward_ready_events
        del self.ve_send_backward_handles
        del self.ve_send_backward_ready_events
        del self.ve_fake_send_backward_ready_events
        del self.ve_fake_send_backward_handles
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

        if not self.is_first_stage():
            return
        microbatch_id = op.microbatch_id
        if microbatch_id is None or microbatch_id < 0:
            return
        if microbatch_id >= len(self.first_pp_inputs):
            return
        llm_input = self.first_pp_inputs[microbatch_id]
        if llm_input is None:
            return

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
                    cpu_value = value.detach().to("cpu", non_blocking=True)
                    cpu_value.requires_grad_(False)
                    llm_input[name] = cpu_value
                else:
                    llm_input[name] = value
            offload_done = torch.cuda.Event()
            offload_done.record(self.offload_stream)
            self.llm_input_offload_done_events[microbatch_id] = offload_done

    def llm_input_onload_action(self, op: ScheduleOp):
        if not self.is_first_stage():
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
                    cuda_value = value.detach().to(target_device, non_blocking=True)
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
                    if output_meta_shape[i] == -1:
                        continue
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
        self._current_ve_microbatch_id = op.microbatch_id
        try:
            setattr(self.data_iterator, "_zzl_current_microbatch_id", op.microbatch_id)
        except Exception:
            pass
        result = _call_with_supported_kwargs(
            self.encoder_forward_func,
            self.encoder_models,
            self.data_iterator,
            num_microbatches=self.num_microbatches,
            pp_size=self.pp_size,
            encoder_output_metas=self.encoder_output_metas,
            microbatch_id=getattr(self, "_current_ve_microbatch_id", None),
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
        # Shapes are available before tensor communication completes. Keep only
        # metadata, before filtering/releasing first-stage-only tensor fields.
        backward_shapes = {
            meta.name: [tuple(inputs[meta.name].shape) for inputs in llm_inputs]
            for meta in self.encoder_output_metas
            if meta.requires_grad and not meta.is_none and not meta.is_object
        }
        if forward_contains_fake:
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
            self.ve_fake_backward_shapes.append(backward_shapes)
            self.ve_fake_llm_input_fake_masks_for_backward.append(llm_input_fake_flags)
        else:
            assert group_idx == len(self.ve_output_tensors), (
                f"rank {dist.get_rank()} saw unexpected VE forward group order: "
                f"group_idx={group_idx}, cached_groups={len(self.ve_output_tensors)}"
            )
            self.ve_output_tensors.append(encoder_outputs)
            self.ve_backward_shapes.append(backward_shapes)
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
        
        # First PP stage also keeps references for VE backward; release them
        # explicitly after encoder_send_backward_action consumes their grads.
        is_first_stage = self.is_first_stage()
        first_stage_only_fields = {
            meta.name
            for meta in self.encoder_output_metas
            if getattr(meta, "only_requires_in_first_stage", False)
        }
        for i in range(len(llm_inputs)):
            llm_input = llm_inputs[i]
            if not is_first_stage:
                llm_input = {
                    name: val
                    for name, val in llm_input.items()
                    if name not in first_stage_only_fields
                }
            self.llm_data_iterator.put(llm_input)
            if is_first_stage:
                self.first_pp_inputs.append(llm_inputs[i])
            else:
                self.first_pp_inputs.append(None)

        self.ve_send_forward_handles.append(send_handles)
        self.ve_send_forward_keepalives.append(send_keepalive)
        self.ve_send_forward_ready_events.append(send_ready_event)
        self.ve_forward_release_tensors.append(encoder_outputs)

    def encoder_send_backward_action(self, op: ScheduleOp):
        # 构造ve output tensor grad
        microbatch_id = op.microbatch_id
        assert self.is_first_stage()
        encoder_outputs = self.first_pp_inputs[microbatch_id]
        encoder_output_grads = []
        for meta in self.encoder_output_metas:
            name = meta.name
            encoder_output = encoder_outputs[name]
            if meta.requires_grad:
                grad = encoder_output.grad
                encoder_output.grad = None
                if grad is None:
                    grad = torch.zeros_like(encoder_output)
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
        self.first_pp_inputs[microbatch_id] = None
        self.first_pp_input_grads.append(encoder_output_grads)

    def encoder_recv_forward_action(self, op: ScheduleOp):
        pass
        

    def encoder_recv_backward_action(self, op: ScheduleOp):
        microbatch_id = op.microbatch_id
        if microbatch_id < 0:
            llm_input_fake_flags = self.ve_fake_llm_input_fake_masks_for_backward.pop(0)
            backward_shapes = self.ve_fake_backward_shapes.pop(0)
        else:
            assert microbatch_id % self.pp_size == self.pp_rank
            group_idx = microbatch_id // self.pp_size
            llm_input_fake_flags = self.ve_llm_input_fake_masks_for_backward[group_idx]
            self.ve_llm_input_fake_masks_for_backward[group_idx] = None
            backward_shapes = self.ve_backward_shapes[group_idx]
            self.ve_backward_shapes[group_idx] = None

        if self.is_first_stage():
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
        
        encoder_output_grads, scatter_handles, backward_ready_event = self.encoder_backward_reshard(
            encoder_output_grads, tensor_shapes=backward_shapes,
        )

        if microbatch_id < 0:
            self.ve_fake_send_backward_handles.append(scatter_handles)
            self.ve_fake_send_backward_ready_events.append(backward_ready_event)
            self.ve_fake_output_tensor_grads.append(encoder_output_grads)
            return

        self.ve_send_backward_handles.append(scatter_handles)
        self.ve_send_backward_ready_events.append(backward_ready_event)
        self.ve_output_tensor_grads.append(encoder_output_grads)
    

    def encoder_backward_reshard(self, encoder_output_grads, tensor_shapes=None):
        # scatter along pp
        assert len(encoder_output_grads) == self.pp_size
        by_field = [
            [encoder_output_grads[j][i] for j in range(self.pp_size)]
            if self.is_first_stage() and meta.requires_grad else [None] * self.pp_size
            for i, meta in enumerate(self.encoder_output_metas)
        ]
        grads, handles = scatter_along_pp(
            by_field, self.pp_group, 0, self.encoder_output_metas, comm_requires_grad_only=True,
            tensor_shapes=tensor_shapes,
        )
        return [grads[meta.name] for meta in self.encoder_output_metas], handles, None

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

        # Once this VE wave reaches the local PP stage that will consume it,
        # the comm keepalives and original encoder output storage can be
        # released while preserving the autograd graph for VE backward.
        encoder_outputs = self.ve_forward_release_tensors[group_idx]
        self.ve_forward_release_tensors[group_idx] = None
        if encoder_outputs is not None:
            for i, meta in enumerate(self.encoder_output_metas):
                encoder_output = encoder_outputs[i]
                if encoder_output is not None and meta.requires_grad:
                    release_tensor(encoder_output)
        

    def recv_forward_action(self, op: ScheduleOp):
        self._finish_ve_forward_handoff(op.microbatch_id)
        
        input_tensor = self.p2p_communicator.recv_forward(
            self.tensor_shapes, is_pp_first_stage(self.p2p_communicator.pp_group)
        )
        self.input_tensors.append(input_tensor)
    
    def send_backward_recv_forward_action(self, op: FusedScheduleOp):
        op1 = op.op1
        op2 = op.op2
        
        input_tensor_grad = self.input_tensor_grads[op1.microbatch_id]
        input_tensor = self.p2p_communicator.send_backward_recv_forward(input_tensor_grad, self.tensor_shapes, is_pp_first_stage(self.p2p_communicator.pp_group))
        self.input_tensor_grads[op1.microbatch_id] = None
        self._finish_ve_forward_handoff(op2.microbatch_id)
        self.input_tensors.append(input_tensor)

    def forward_action(self, op: ScheduleOp):
        if self.is_first_stage():
            onload_done = self.llm_input_onload_done_events.pop(op.microbatch_id, None)
            if onload_done is not None:
                torch.cuda.current_stream().wait_event(onload_done)
        super().forward_action(op)
