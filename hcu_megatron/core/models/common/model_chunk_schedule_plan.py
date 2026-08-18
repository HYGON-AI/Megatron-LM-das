# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from contextlib import nullcontext

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context

from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan as MegatronTransformerLayerSchedulePlan
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan as MegatronTransformerModelChunkSchedulePlan
from megatron.core.pipeline_parallel.utils import (
    NoopScheduleNode,
    get_comm_stream,
)
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from hcu_megatron.training.arguments import get_adaptor_args


class TransformerLayerSchedulePlanWithoutSplitAttn(MegatronTransformerLayerSchedulePlan):

    @staticmethod
    def run(f_layer, b_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False, block_level_wgrad_compute=False):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        This function interleaves forward and backward operations, overlapping the communications
        (dispatch or combine) of one with the computations (att or mlp) of the other
        to maximize parallelism and efficiency.

        When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
        comm_stream: combine_bwd | dispatch_fwd->dispatch_bwd  | combine_fwd
        comp_stream: attn_fwd    | mlp_bwd->mlp_bwd_dw->mlp_fwd| attn_bwd
        For MTP, mtp_post_process_fwd is executed after the combine_fwd in the comp_stream,
        and mtp_post_process_bwd is executed before the combine_bwd in the comp_stream.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.attn.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if b_layer is not None and b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn.backward_dw()

        return f_input, b_grad


F_ATTN_PRE_B_COMBINE_SYNC_EVENT = torch.cuda.Event()
B_ATTN_POST_F_COMBINE_SYNC_EVENT = torch.cuda.Event()


class TransformerLayerSchedulePlanWithSplitAttn:
    """Schedule the executing plan of the nodes in a transformer layer.
    """

    attn_qkv = None
    core_attn = None
    attn_proj = None
    moe_dispatch = None
    mlp = None
    moe_combine = None
    mtp_post_process = None

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args={}):
        """Initializes a transformer layer schedule plan.

        Args:
            layer (TransformerLayer):
                split a transformer layer into multiple nodes for fine-grained scheduling.
            event (torch.cuda.Event):
                record CUDA event across multiple nodes on different streams for synchronization.
            chunk_state (ModelChunkState): model state shared in the model chunk.
            comp_stream (Callable): Func that returns CUDA stream for computation.
            comm_stream (Callable): Func that returns CUDA stream for communication.
            extra_args (dict): extra arguments for the layer.

        The event and chunk_state are binded to the TransformerModelChunkSchedulePlan
        and shared across all layers in the model chunk.
        """
        from megatron.core.models.gpt.fine_grained_callables import TransformerLayerState

        self.config = layer.config
        self.layer_state = TransformerLayerState()
        self.chunk_state = chunk_state
        self.layer = layer
        self.event = event
        self.comp_stream = comp_stream
        self.comm_stream = comm_stream

        # get callable nodes for transformer/mtp layer
        self._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        if hasattr(self, 'attn_qkv') and self.attn_qkv is not None:
            del self.attn_qkv
            self.attn_qkv = None
        if hasattr(self, 'core_attn') and self.core_attn is not None:
            del self.core_attn
            self.core_attn = None
        if hasattr(self, 'attn_proj') and self.attn_proj is not None:
            del self.attn_proj
            self.attn_proj = None
        if hasattr(self, 'moe_dispatch') and self.moe_dispatch is not None:
            del self.moe_dispatch
            self.moe_dispatch = None
        if hasattr(self, 'mlp') and self.mlp is not None:
            del self.mlp
            self.mlp = None
        if hasattr(self, 'moe_combine') and self.moe_combine is not None:
            del self.moe_combine
            self.moe_combine = None
        if hasattr(self, 'mtp_post_process') and self.mtp_post_process is not None:
            del self.mtp_post_process
            self.mtp_post_process = None
        if hasattr(self, 'layer_state') and self.layer_state is not None:
            del self.layer_state
            self.layer_state = None
        if hasattr(self, 'layer'):
            del self.layer

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            attn_qkv, core_attn, attn_proj, mlp, moe_dispatch and moe_combine, and mtp_post_process.
        """
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        from hcu_megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode
        from hcu_megatron.core.models.gpt.fine_grained_callables import build_layer_callables_with_split_attn

        # build the forward and backward callables for the transformer/mtp layer
        fwd_callables, bwd_dw_callable_map = build_layer_callables_with_split_attn(self.layer)

        # get flags for latter use
        is_mtp = isinstance(self.layer, MultiTokenPredictionLayer)
        transformer_layer = self.layer.mtp_model_layer if is_mtp else self.layer
        is_moe = isinstance(transformer_layer.mlp, MoELayer)
        num_local_experts = transformer_layer.mlp.num_local_experts if is_moe else None

        extra_args["config"] = self.layer.config
        extra_args["is_moe"] = is_moe
        extra_args["num_local_experts"] = num_local_experts
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = is_mtp

        # wrapper to help create TransformerLayerNode
        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            return TransformerLayerNode(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=extra_args,
            )

        (
            attn_qkv_module,
            core_attn_module,
            attn_proj_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module
        ) = fwd_callables

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.attn_qkv = create_node(comp_stream, attn_qkv_module, "attn_qkv")
        self.core_attn = create_node(comp_stream, core_attn_module, "core_attn")
        self.attn_proj = create_node(comp_stream, attn_proj_module, "attn_proj")
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        if is_mtp:
            self.mtp_post_process = create_node(
                comp_stream, mtp_post_process_module, "mtp_post_process"
            )
        else:
            self.mtp_post_process = NoopScheduleNode()

    def set_fsdp_reshard_hooks(self, post_forward_hook, post_backward_hook):
        """Wire FSDP parameter release callbacks for the fine-grained overlap schedule.

        The EP overlap schedule bypasses the normal FSDP forward/backward hooks
        (registered on the FSDP unit module) because it calls sub-modules directly
        instead of going through TransformerLayer.forward(). This method attaches
        explicit release hooks to individual schedule nodes so that all-gathered
        parameters are freed at the right time.

        Args:
            post_forward_hook: Callable(module) that releases forward-pass params
                (bwd=False). Typically ``fsdp_wrapper.post_forward_release_module``.
            post_backward_hook: Callable(module) that releases backward-pass params
                (bwd=True). Typically ``fsdp_wrapper.post_backward_release_module``.
        """
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
        from megatron.core.transformer.transformer_layer import TransformerLayer

        assert isinstance(self.layer, (TransformerLayer, MultiTokenPredictionLayer)), (
            f"Megatron FSDP with EP Overlap only supports TransformerLayer, "
            f"but got {type(self.layer).__name__}."
        )

        if isinstance(self.layer, TransformerLayer):
            hook_module = self.layer
        else:
            hook_module = self.layer.mtp_model_layer

        # After the last backward op (attn), release backward-pass params.
        self.attn.set_post_backward_hook(lambda: post_backward_hook(hook_module))

        # Determine the last node in forward order.
        if isinstance(self.moe_combine, NoopScheduleNode):
            last_fwd_node = self.mlp
        else:
            last_fwd_node = self.moe_combine

        # After the last forward op, release forward-pass params.
        last_fwd_node.set_post_forward_hook(lambda: post_forward_hook(hook_module))

    def get_fp8_context(self):
        """
        Get the fp8 context for the transformer layer.
        """
        use_inner_fp8_context = (
            self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        )
        return (
            get_fp8_context(self.layer.config, self.layer.layer_number - 1)
            if use_inner_fp8_context
            else nullcontext()
        )

    @staticmethod
    def run(
        f_layer,
        b_layer,
        f_input=None,
        b_grad=None,
        is_last_layer_in_bwd=False,
        block_level_wgrad_compute=False,
    ):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        This function interleaves forward and backward operations, overlapping the communications
        (dispatch or combine) of one with the computations (att or mlp) of the other
        to maximize parallelism and efficiency.

        When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
        comm_stream: combine_bwd            | dispatch_fwd->dispatch_bwd  | combine_fwd
        comp_stream: attn_fwd->post_attn_fwd| mlp_bwd->mlp_bwd_dw->mlp_fwd| post_attn_bwd->attn_bwd
        For MTP, mtp_post_process_fwd is executed after the combine_fwd in the comp_stream,
        and mtp_post_process_bwd is executed before the combine_bwd in the comp_stream.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """
        is_sync_1f1b = f_layer is not None and b_layer is not None

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)

        f_attn_pre_b_combine_sync_event = F_ATTN_PRE_B_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.attn_qkv.forward(
                    f_input,
                    stream_record_event=f_attn_pre_b_combine_sync_event,
                )

        if b_layer is not None:
            b_grad = b_layer.moe_combine.backward(
                b_grad,
                stream_wait_event=f_attn_pre_b_combine_sync_event,
            )

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.core_attn.forward(f_input)
                f_input = f_layer.attn_proj.forward(
                    f_input,
                )

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input,)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        b_attn_post_f_combine_sync_event = B_ATTN_POST_F_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if b_layer is not None:
            b_grad = b_layer.attn_proj.backward(
                b_grad,
                stream_record_event=b_attn_post_f_combine_sync_event,
            )

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(
                    f_input,
                    stream_wait_event=b_attn_post_f_combine_sync_event,
                )

        if b_layer is not None:
            b_grad = b_layer.core_attn.backward(b_grad)
            b_grad = b_layer.attn_qkv.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn_qkv.backward_dw()
                b_layer.attn_proj.backward_dw()

        return f_input, b_grad


if get_adaptor_args().integrate_recompute_to_ep_comm_overlap:
    from .model_chunk_schedule_plan_with_recompute import TransformerLayerSchedulePlanWithRecompute
    layer_schedule_plan_cls = TransformerLayerSchedulePlanWithRecompute
elif get_adaptor_args().overlap_ep_comm_with_split_attn:
    layer_schedule_plan_cls = TransformerLayerSchedulePlanWithSplitAttn
else:
    layer_schedule_plan_cls = TransformerLayerSchedulePlanWithoutSplitAttn

class TransformerModelChunkSchedulePlan(MegatronTransformerModelChunkSchedulePlan):
    """Schedule the executing plan of the sub-modules in a model chunk sub-modules.

    This class organizes the computation nodes for a model chunk,
    including preprocessing, transformer layers, and postprocessing.

    TransformerModelChunkSchedulePlan
    ├── pre_process: PreProcessNode
    ├── layers: List[TransformerLayerSchedulePlan]
    │   ├── layer[0]: TransformerLayerSchedulePlan
    │   ├── layer[1]: TransformerLayerSchedulePlan
    │   └── ...
    └── post_process: PostProcessNode
    """

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream):
        if module is None:
            return
        num_layers = len(module.layers)
        for layer_idx in range(num_layers):
            extra_args = {
                "is_first_layer": layer_idx == 0,
                "is_last_layer": layer_idx == num_layers - 1,
            }
            layer_plan = layer_schedule_plan_cls(
                module.layers[layer_idx],
                self.event,
                self.state,
                comp_stream,
                comm_stream,
                extra_args,
            )
            self._transformer_layers.append(layer_plan)

    @staticmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        block_level_wgrad_compute=False,
    ):
        args = get_adaptor_args()
        if args.integrate_recompute_to_ep_comm_overlap and args.ep_overlap_early_recompute:
            run_func = TransformerModelChunkSchedulePlan.run_recompute_with_overlap_three_layers
            return run_func(
                f_schedule_plan,
                b_schedule_plan,
                b_grad=b_grad,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
                block_level_wgrad_compute=block_level_wgrad_compute,
            )

        f_input = None
        if f_schedule_plan:
            # pp output send/receive sync
            if pre_forward is not None:
                pre_forward(None if args.schedule_method == "dualpipev" else f_schedule_plan.vp_stage)
            f_schedule_plan.record_current_stream()
            f_input = f_schedule_plan.pre_process.forward()

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None
            if pre_backward is not None:
                pre_backward(None if args.schedule_method == "dualpipev" else b_schedule_plan.vp_stage)
                b_schedule_plan.record_current_stream()

            if b_schedule_plan.post_process is not None:
                b_grad = b_schedule_plan.post_process.backward(b_grad)

        f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
        b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
        overlapped_layers = min(f_num_layers, b_num_layers)

        f_layer = b_layer = None
        # combined forward and backward pass for overlapped layers
        for i in range(overlapped_layers):
            f_layer = f_schedule_plan.get_layer(i)
            b_layer = b_schedule_plan.pop_layer() if not block_level_wgrad_compute else b_schedule_plan.get_layer(b_num_layers - 1 - i)
            nvtx_msg = f"layer_{i}f-layer_{b_num_layers - 1 - i}b"
            nvtx_range_push(nvtx_msg)
            f_input, b_grad = layer_schedule_plan_cls.run(
                f_layer,
                b_layer,
                f_input=f_input,
                b_grad=b_grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
                block_level_wgrad_compute=block_level_wgrad_compute,
            )
            if not block_level_wgrad_compute and i < b_num_layers - 1:
                b_layer.release_state()
            nvtx_range_pop(nvtx_msg)

        # backward pass for the remaining layers
        for i in range(overlapped_layers, b_num_layers):
            b_layer = b_schedule_plan.pop_layer() if not block_level_wgrad_compute else b_schedule_plan.get_layer(b_num_layers - 1 - i)
            nvtx_msg = f"layer_{b_num_layers - 1 - i}b"
            nvtx_range_push(nvtx_msg)
            _, b_grad = layer_schedule_plan_cls.run(
                None,
                b_layer,
                b_grad=b_grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
                block_level_wgrad_compute=block_level_wgrad_compute,
            )
            if not block_level_wgrad_compute and i < b_num_layers - 1:
                b_layer.release_state()
            nvtx_range_pop(nvtx_msg)

        # forward pass for the remaining layers
        for i in range(overlapped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            nvtx_msg = f"layer_{i}f"
            nvtx_range_push(nvtx_msg)
            f_input, _ = layer_schedule_plan_cls.run(f_layer, None, f_input=f_input)
            nvtx_range_pop(nvtx_msg)

        if f_schedule_plan is not None and post_forward is not None:
            # post_forward()/send_forward_recv_forward() is running in the communication stream,
            # so the p2p comm could be overlapped with the attn backward
            with torch.cuda.stream(get_comm_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input, None if args.schedule_method == "dualpipev" else f_schedule_plan.vp_stage)

        # post_backward()/send_backward_recv_backward() is running in the computation stream,
        # so the p2p comm could be overlapped with the wgrad of attn backward
        if b_schedule_plan is not None and post_backward is not None:
            b_schedule_plan.wait_current_stream()
            post_backward(b_grad, None if args.schedule_method == "dualpipev" else b_schedule_plan.vp_stage)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute and b_num_layers > 0:
            assert b_layer is not None
            if get_adaptor_args().overlap_ep_comm_with_split_attn:
                b_layer.attn_qkv.backward_dw()
                b_layer.attn_proj.backward_dw()
            else:
                b_layer.attn.backward_dw()
            b_layer.release_state()

        # post process forward
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()
            # Release reference as early as possible, this helps avoid memory leak.
            b_schedule_plan.release_state()

        if get_adaptor_args().schedule_method != "dualpipev":
            assert not block_level_wgrad_compute, "block_level_wgrad_compute should be False when not using dualpipev"
            return f_input

        if b_num_layers and block_level_wgrad_compute:
            def chunk_backward_dw():
                for _ in range(b_num_layers):
                    b_layer = b_schedule_plan.pop_layer()
                    if get_adaptor_args().overlap_ep_comm_with_split_attn:
                        b_layer.attn_qkv.backward_dw()
                        b_layer.attn_proj.backward_dw()
                    else:
                        b_layer.attn.backward_dw()
                    b_layer.mlp.backward_dw()
                    b_layer.release_state()
            return f_input, chunk_backward_dw

        return f_input, None

    @staticmethod
    def run_recompute_with_overlap_three_layers(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        block_level_wgrad_compute=False,
    ):
        """Model Chunk level 1f1b fine-grained scheduler.

        This function schedules the forward, backward and recomputation passes for a model chunk,
        which interleaves forward, backward and recomputation function of multiple Transformer layers
        within a model chunk, and this is needed to overlap the submodules between the individual
        forward and backward functions.

        Assume there are 4 layers in the given model chunk:
        Phase 0: p2p_comm_sync -> forward_preprocess -> p2p_comm_sync -> backward_postprocess
        Phase 1: recompute_layer[3]
        Phase 2: forward_layer[0] + backward_layer[3] + recompute_layer[2], overlapped execution by schedule_layer_1f1b
        Phase 3: forward_layer[1] + backward_layer[2] + recompute_layer[1], overlapped execution by schedule_layer_1f1b
        Phase 4: forward_layer[2] + backward_layer[1] + recompute_layer[0], overlapped execution by schedule_layer_1f1b
        Phase 5: forward_layer[3] + backward_layer[0], overlapped execution by schedule_layer_1f1b
        Phase 5: send_forward_recv_backward -> send_backward_recv_forward
        Phase 6: backward_dw of the first layer -> forward_postprocess -> backward_preprocess

        Args:
            f_schedule_plan (TransformerModelChunkSchedulePlan): The forward schedule plan
            b_schedule_plan (TransformerModelChunkSchedulePlan): The backward schedule plan
            b_grad (Tensor or None): The gradient of the loss function
            pre_forward (callable or None): The function to call before the forward pass
            pre_backward (callable or None): The function to call before the backward pass
            post_forward (callable or None): The function to call after the forward pass
            post_backward (callable or None): The function to call after the backward pass
        Returns:
            The output of the forward pass.
        """
        args = get_adaptor_args()

        f_input = None
        if f_schedule_plan:
            # pp output send/receive sync
            if pre_forward is not None:
                pre_forward(None if args.schedule_method == "dualpipev" else f_schedule_plan.vp_stage)
            f_schedule_plan.record_current_stream()
            f_input = f_schedule_plan.pre_process.forward()

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None
            if pre_backward is not None:
                pre_backward(None if args.schedule_method == "dualpipev" else b_schedule_plan.vp_stage)
                b_schedule_plan.record_current_stream()

            if b_schedule_plan.post_process is not None:
                b_grad = b_schedule_plan.post_process.backward(b_grad)

        f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
        b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
        overlapped_layers = min(f_num_layers, b_num_layers - 1) if b_num_layers > 0 else 0

        f_layer = b_layer = r_layer = None

        # S1: recompute last layer
        if b_schedule_plan is not None:
            r_layer = b_schedule_plan.pop_layer() if not block_level_wgrad_compute else b_schedule_plan.get_layer(b_num_layers - 1)
            nvtx_msg = f"layer_{b_num_layers - 1}r"
            nvtx_range_push(nvtx_msg)
            _, _ = layer_schedule_plan_cls.run_early_recompute(
                None,
                None,
                r_layer,
            )
            b_layer = r_layer
            nvtx_range_pop(nvtx_msg)

        # S2: combined forward, backward and recomputation pass for overlapped layers
        for i in range(overlapped_layers):
            f_layer = f_schedule_plan.get_layer(i)
            r_layer = b_schedule_plan.pop_layer() if not block_level_wgrad_compute else b_schedule_plan.get_layer(b_num_layers - 1 - i - 1)
            nvtx_msg = f"layer_{i}f-layer_{b_num_layers - 1 - i}b-layer_{b_num_layers - 2 - i}r"
            nvtx_range_push(nvtx_msg)
            f_input, b_grad = layer_schedule_plan_cls.run_early_recompute(
                f_layer,
                b_layer,
                r_layer,
                f_input=f_input,
                b_grad=b_grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
                block_level_wgrad_compute=block_level_wgrad_compute,
            )
            if not block_level_wgrad_compute:
                b_layer.release_state()
            b_layer = r_layer
            nvtx_range_pop(nvtx_msg)

        if overlapped_layers == f_num_layers:
            # S3: combined backward and recomputation pass for overlapped layers
            for i in range(overlapped_layers, b_num_layers - 1):
                r_layer = b_schedule_plan.pop_layer() if not block_level_wgrad_compute else b_schedule_plan.get_layer(b_num_layers - 1 - i - 1)
                nvtx_msg = f"layer_{b_num_layers - 1 - i}b-layer_{b_num_layers - 2 - i}r"
                nvtx_range_push(nvtx_msg)
                _, b_grad = layer_schedule_plan_cls.run_early_recompute(
                    None,
                    b_layer,
                    r_layer,
                    b_grad=b_grad,
                    is_last_layer_in_bwd=(i == b_num_layers - 1),
                    block_level_wgrad_compute=block_level_wgrad_compute,
                )
                if not block_level_wgrad_compute:
                    b_layer.release_state()
                b_layer = r_layer
                nvtx_range_pop(nvtx_msg)

            # S4: backward pass for the first layer
            nvtx_msg = f"layer_0b"
            nvtx_range_push(nvtx_msg)
            _, b_grad = layer_schedule_plan_cls.run_early_recompute(
                None,
                b_layer,
                None,
                b_grad=b_grad,
                is_last_layer_in_bwd=True,
                block_level_wgrad_compute=block_level_wgrad_compute,
            )
            nvtx_range_pop(nvtx_msg)
        else:
            # S3: combined forward and backward pass for overlapped layers
            f_layer = f_schedule_plan.get_layer(overlapped_layers)
            nvtx_msg = f"layer_{overlapped_layers}f-layer_0b"
            nvtx_range_push(nvtx_msg)
            f_input, b_grad = layer_schedule_plan_cls.run_early_recompute(
                f_layer,
                b_layer,
                None,
                f_input=f_input,
                b_grad=b_grad,
                is_last_layer_in_bwd=True,
                block_level_wgrad_compute=block_level_wgrad_compute,
            )
            nvtx_range_pop(nvtx_msg)

            # S4: forward pass for the remaining layers
            for i in range(overlapped_layers + 1, f_num_layers):
                f_layer = f_schedule_plan.get_layer(i)
                nvtx_msg = f"layer_{i}f"
                nvtx_range_push(nvtx_msg)
                f_input, _ = layer_schedule_plan_cls.run_early_recompute(f_layer, None, None, f_input=f_input)
                nvtx_range_pop(nvtx_msg)

        if f_schedule_plan is not None and post_forward is not None:
            # post_forward()/send_forward_recv_forward() is running in the communication stream,
            # so the p2p comm could be overlapped with the attn backward
            with torch.cuda.stream(get_comm_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input, None if args.schedule_method == "dualpipev" else f_schedule_plan.vp_stage)

        # post_backward()/send_backward_recv_backward() is running in the computation stream,
        # so the p2p comm could be overlapped with the wgrad of attn backward
        if b_schedule_plan is not None and post_backward is not None:
            b_schedule_plan.wait_current_stream()
            post_backward(b_grad, None if args.schedule_method == "dualpipev" else b_schedule_plan.vp_stage)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute and b_num_layers > 0:
            assert b_layer is not None
            if get_adaptor_args().overlap_ep_comm_with_split_attn:
                b_layer.attn_qkv.backward_dw()
                b_layer.attn_proj.backward_dw()
            else:
                b_layer.attn.backward_dw()
            b_layer.release_state()

        # post process forward
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()
            # Release reference as early as possible, this helps avoid memory leak.
            b_schedule_plan.release_state()

        if get_adaptor_args().schedule_method != "dualpipev":
            assert not block_level_wgrad_compute, "block_level_wgrad_compute should be False when not using dualpipev"
            return f_input

        if b_num_layers and block_level_wgrad_compute:
            def chunk_backward_dw():
                for _ in range(b_num_layers):
                    b_layer = b_schedule_plan.pop_layer()
                    if get_adaptor_args().overlap_ep_comm_with_split_attn:
                        b_layer.attn_qkv.backward_dw()
                        b_layer.attn_proj.backward_dw()
                    else:
                        b_layer.attn.backward_dw()
                    b_layer.mlp.backward_dw()
                    b_layer.release_state()
            return f_input, chunk_backward_dw

        return f_input, None
