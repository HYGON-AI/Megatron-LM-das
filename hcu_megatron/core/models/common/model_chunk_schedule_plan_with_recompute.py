# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import contextlib
from contextlib import nullcontext

import torch

from megatron.core.tensor_parallel.random import _get_all_rng_states, _set_all_rng_states
from megatron.core.pipeline_parallel.utils import NoopScheduleNode
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan as MegatronTransformerLayerSchedulePlan

from .model_chunk_schedule_plan import TransformerLayerSchedulePlanWithSplitAttn as _TransformerLayerSchedulePlanWithSplitAttn
from hcu_megatron.training.arguments import get_adaptor_args


@contextlib.contextmanager
def _fork_recompute_rng(origin_states):
    """Fork the rng state."""
    # Store the current states.
    current_states = _get_all_rng_states()
    _set_all_rng_states(*origin_states)
    try:
        yield
    finally:
        # Set the states back to what it was at the start of this function.
        _set_all_rng_states(*current_states)


def fork_recompute_rng(origin_states):
    if origin_states is None:
        return nullcontext()

    return _fork_recompute_rng(origin_states)


def get_grad_context(is_recompute=False):
    return torch.enable_grad() if is_recompute else torch.no_grad()


class TransformerLayerSchedulePlanWithoutSplitAttn(MegatronTransformerLayerSchedulePlan):

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

        # for recomputing
        self.saved_tensors = None
        self.rng_states = None

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            attn, mlp, moe_dispatch and moe_combine, and mtp_post_process.
        """
        from megatron.core.models.gpt.fine_grained_callables import build_layer_callables
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        from hcu_megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode

        # build the forward and backward callables for the transformer/mtp layer
        fwd_callables, bwd_dw_callable_map = build_layer_callables(self.layer)

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
            attn_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.attn = create_node(comp_stream, attn_module, "attn")
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

        if f_layer is not None:
            f_layer.saved_tensors = (f_input,)
            f_layer.rng_states = _get_all_rng_states()

        if b_layer is not None:
            b_input_recompute = b_layer.saved_tensors
            b_layer_rng_states = b_layer.rng_states

        if b_layer is not None:
            with _fork_recompute_rng(b_layer_rng_states):
                with torch.enable_grad(), b_layer.get_fp8_context():
                    b_input_recompute = b_layer.attn.forward(b_input_recompute, is_recompute=True)
                    b_input_recompute = b_layer.moe_dispatch.forward(b_input_recompute, is_recompute=True)
                    b_input_recompute = b_layer.mlp.forward(b_input_recompute, is_recompute=True)
                    b_input_recompute = b_layer.moe_combine.forward(b_input_recompute, is_recompute=True)
                    b_input_recompute = b_layer.mtp_post_process.forward(b_input_recompute, is_recompute=True)

            b_layer.saved_tensors = None
            b_layer.rng_states = None

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.attn.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if b_layer is not None and b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)
                f_input = f_layer.mtp_post_process.forward(f_input)

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn.backward_dw()

        return f_input, b_grad

    @staticmethod
    def run_early_recompute(
        f_layer,
        b_layer,
        r_layer,
        f_input=None,
        b_grad=None,
        is_last_layer_in_bwd=False,
        block_level_wgrad_compute=False,
    ):
        if f_layer is None or r_layer is None:
            overlap_func = TransformerLayerSchedulePlanWithoutSplitAttn.run_overlap_fb_or_rb_layers
        else:
            overlap_func = TransformerLayerSchedulePlanWithoutSplitAttn.run_overlap_fbr_layers

        return overlap_func(
            f_layer,
            b_layer,
            r_layer,
            f_input=f_input,
            b_grad=b_grad,
            is_last_layer_in_bwd=is_last_layer_in_bwd,
            block_level_wgrad_compute=block_level_wgrad_compute
        )

    @staticmethod
    def run_overlap_fbr_layers(f_layer, b_layer, r_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False, block_level_wgrad_compute=False):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            r_layer (TransformerLayerSchedulePlan): recomputation layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """
        if f_layer is not None:
            f_layer.saved_tensors = (f_input,)
            f_layer.rng_states = _get_all_rng_states()

        if r_layer is not None:
            r_input = r_layer.saved_tensors
            r_layer_rng_states = r_layer.rng_states

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.attn.forward(r_input, is_recompute=True)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.attn.forward(f_input)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.moe_dispatch.forward(r_input, is_recompute=True)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.mlp.forward(r_input, is_recompute=True)

        if b_layer is not None and b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.moe_combine.forward(r_input, is_recompute=True)
                    r_input = r_layer.mtp_post_process.forward(r_input, is_recompute=True)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)
                f_input = f_layer.mtp_post_process.forward(f_input)

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn.backward_dw()

        if r_layer is not None:
            r_layer.saved_tensors = None
            r_layer.rng_states = None

        return f_input, b_grad

    @staticmethod
    def run_overlap_fb_or_rb_layers(f_layer, b_layer, r_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False, block_level_wgrad_compute=False):
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
            r_layer (TransformerLayerSchedulePlan): Recomputation layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """

        assert f_layer is None or r_layer is None, "Either f_layer or r_layer (or both) is None"

        if f_layer is not None:
            f_layer.saved_tensors = (f_input,)
            f_layer.rng_states = _get_all_rng_states()

        r_input = None
        r_layer_rng_states = None
        if r_layer is not None:
            r_input = r_layer.saved_tensors
            r_layer_rng_states = r_layer.rng_states

        r_or_f_layer = f_layer if f_layer is not None else r_layer
        r_or_f_input = f_input if f_layer is not None else r_input

        if b_layer is None:
            # can help improve performance
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.attn.forward(r_or_f_input, is_recompute=r_layer is not None,)
                    r_or_f_input = r_or_f_layer.moe_dispatch.forward(r_or_f_input, is_recompute=r_layer is not None,)
                    r_or_f_input = r_or_f_layer.mlp.forward(r_or_f_input, is_recompute=r_layer is not None,)
                    r_or_f_input = r_or_f_layer.moe_combine.forward(r_or_f_input, is_recompute=r_layer is not None,)
                    r_or_f_input = r_or_f_layer.mtp_post_process.forward(r_or_f_input, is_recompute=r_layer is not None,)
            if r_layer is not None:
                r_layer.saved_tensors = None
                r_layer.rng_states = None
            return r_or_f_input if f_layer is not None else None, None

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.attn.forward(r_or_f_input, is_recompute=r_layer is not None,)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.moe_dispatch.forward(r_or_f_input, is_recompute=r_layer is not None,)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if b_layer is not None and b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.mlp.forward(r_or_f_input, is_recompute=r_layer is not None,)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.moe_combine.forward(r_or_f_input, is_recompute=r_layer is not None,)
                    r_or_f_input = r_or_f_layer.mtp_post_process.forward(r_or_f_input, is_recompute=r_layer is not None,)

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn.backward_dw()

        if r_layer is not None:
            r_layer.saved_tensors = None
            r_layer.rng_states = None

        return r_or_f_input if f_layer is not None else None, b_grad


R_ATTN_PRE_B_COMBINE_SYNC_EVENT = torch.cuda.Event()
F_ATTN_PRE_R_DISPATCH_SYNC_EVENT = torch.cuda.Event()
B_ATTN_POST_F_COMBINE_SYNC_EVENT = torch.cuda.Event()
F_ATTN_PRE_B_COMBINE_SYNC_EVENT = torch.cuda.Event()


class TransformerLayerSchedulePlanWithSplitAttn(_TransformerLayerSchedulePlanWithSplitAttn):

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

        super(TransformerLayerSchedulePlanWithSplitAttn, self).__init__(
            layer,
            event,
            chunk_state,
            comp_stream,
            comm_stream,
            extra_args=extra_args,
        )

        # for recomputing
        self.saved_tensors = None
        self.rng_states = None

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

    @staticmethod
    def run(
        f_layer,
        b_layer,
        r_layer,
        f_input=None,
        b_grad=None,
        is_last_layer_in_bwd=False,
        block_level_wgrad_compute=False,
    ):
        if f_layer is None or r_layer is None:
            overlap_func = TransformerLayerSchedulePlanWithSplitAttn.run_overlap_fb_or_rb_layers
        else:
            overlap_func = TransformerLayerSchedulePlanWithSplitAttn.run_overlap_fbr_layers

        return overlap_func(
            f_layer,
            b_layer,
            r_layer,
            f_input=f_input,
            b_grad=b_grad,
            is_last_layer_in_bwd=is_last_layer_in_bwd,
            block_level_wgrad_compute=block_level_wgrad_compute
        )

    @staticmethod
    def run_overlap_fbr_layers(
        f_layer,
        b_layer,
        r_layer,
        f_input=None,
        b_grad=None,
        is_last_layer_in_bwd=False,
        block_level_wgrad_compute=False,
    ):
        """Schedule one-forward-one-backward-one-recomputation operations for a single transformer layer.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            r_layer (TransformerLayerSchedulePlan): Recomputation layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """
        if f_layer is not None:
            f_layer.saved_tensors = (f_input,)
            f_layer.rng_states = _get_all_rng_states()

        if r_layer is not None:
            r_input = r_layer.saved_tensors
            r_layer_rng_states = r_layer.rng_states

        is_sync_1f1b = f_layer is not None and b_layer is not None

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)

        r_attn_pre_b_combine_sync_event = R_ATTN_PRE_B_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.attn_qkv.forward(
                        r_input,
                        stream_record_event=r_attn_pre_b_combine_sync_event,
                        is_recompute=True,
                    )

        if b_layer is not None:
            b_grad = b_layer.moe_combine.backward(
                b_grad,
                stream_wait_event=r_attn_pre_b_combine_sync_event,
            )

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.core_attn.forward(r_input, is_recompute=True)
                    r_input = r_layer.attn_proj.forward(r_input, is_recompute=True)

        f_attn_pre_r_dispatch_sync_event = F_ATTN_PRE_R_DISPATCH_SYNC_EVENT if is_sync_1f1b else None
        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.attn_qkv.forward(
                    f_input,
                    stream_record_event=f_attn_pre_r_dispatch_sync_event,
                )

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.moe_dispatch.forward(
                        r_input,
                        stream_wait_event=f_attn_pre_r_dispatch_sync_event,
                        is_recompute=True,
                    )

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.core_attn.forward(f_input)
                f_input = f_layer.attn_proj.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input,)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.mlp.forward(r_input, is_recompute=True)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.moe_combine.forward(r_input, is_recompute=True)
                    r_input = r_layer.mtp_post_process.forward(r_input, is_recompute=True)

        b_attn_post_f_combine_sync_event = B_ATTN_POST_F_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if b_layer is not None:
            b_grad = b_layer.attn_proj.backward(
                b_grad,
                stream_record_event=b_attn_post_f_combine_sync_event,
            )

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(
                    f_input,
                    stream_wait_event=b_attn_post_f_combine_sync_event,
                )

        if b_layer is not None:
            b_grad = b_layer.core_attn.backward(b_grad)
            b_grad = b_layer.attn_qkv.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn_qkv.backward_dw()
                b_layer.attn_proj.backward_dw()

        return f_input, b_grad

    @staticmethod
    def run_overlap_fb_or_rb_layers(
        f_layer,
        b_layer,
        r_layer,
        f_input=None,
        b_grad=None,
        is_last_layer_in_bwd=False,
        block_level_wgrad_compute=False,
    ):
        """Schedule one-forward (or recompuation)-one-backward operations for a single transformer layer.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            r_layer (TransformerLayerSchedulePlan): Recomputation layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """
        assert f_layer is None or r_layer is None, "Either f_layer or r_layer (or both) is None"

        if f_layer is not None:
            f_layer.saved_tensors = (f_input,)
            f_layer.rng_states = _get_all_rng_states()

        r_input = None
        r_layer_rng_states = None
        if r_layer is not None:
            r_input = r_layer.saved_tensors
            r_layer_rng_states = r_layer.rng_states

        r_or_f_layer = f_layer or r_layer
        r_or_f_input = f_input or r_input
        is_sync_1f1b = r_or_f_layer is not None and b_layer is not None

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)

        f_attn_pre_b_combine_sync_event = F_ATTN_PRE_B_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.attn_qkv.forward(
                        r_or_f_input,
                        stream_record_event=f_attn_pre_b_combine_sync_event,
                        is_recompute=r_layer is not None,
                    )

        if b_layer is not None:
            b_grad = b_layer.moe_combine.backward(
                b_grad,
                stream_wait_event=f_attn_pre_b_combine_sync_event,
            )

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.core_attn.forward(r_or_f_input, is_recompute=r_layer is not None,)
                    r_or_f_input = r_or_f_layer.attn_proj.forward(r_or_f_input, is_recompute=r_layer is not None,)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.moe_dispatch.forward(r_or_f_input, is_recompute=r_layer is not None,)

        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.mlp.forward(r_or_f_input, is_recompute=r_layer is not None,)

        b_attn_post_f_combine_sync_event = B_ATTN_POST_F_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if b_layer is not None:
            b_grad = b_layer.attn_proj.backward(
                b_grad,
                stream_record_event=b_attn_post_f_combine_sync_event,
            )

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.moe_combine.forward(
                        r_or_f_input,
                        stream_wait_event=b_attn_post_f_combine_sync_event,
                        is_recompute=r_layer is not None,
                    )

        if b_layer is not None:
            b_grad = b_layer.core_attn.backward(b_grad)
            b_grad = b_layer.attn_qkv.backward(b_grad)

        if r_or_f_layer is not None:
            with fork_recompute_rng(r_layer_rng_states):
                with get_grad_context(r_layer is not None), r_or_f_layer.get_fp8_context():
                    r_or_f_input = r_or_f_layer.mtp_post_process.forward(r_or_f_input, is_recompute=r_layer is not None,)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn_qkv.backward_dw()
                b_layer.attn_proj.backward_dw()

        return r_or_f_input if f_layer is not None else None, b_grad

    @staticmethod
    def run_overlap_three_layers_bak(f_layer, b_layer, r_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False, block_level_wgrad_compute=False):
        """Schedule one-forward-one-backward-one-recomputation operations for a single transformer layer.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            r_layer (TransformerLayerSchedulePlan): Recomputation layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """
        if f_layer is not None:
            f_layer.saved_tensors = (f_input,)
            f_layer.rng_states = _get_all_rng_states()

        if r_layer is not None:
            r_input = r_layer.saved_tensors
            r_layer_rng_states = r_layer.rng_states

        is_sync_1f1b = f_layer is not None and b_layer is not None

        if b_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.attn_qkv.forward(r_input, is_recompute=True)
                    r_input = r_layer.core_attn.forward(r_input, is_recompute=True)
                    r_input = r_layer.attn_proj.forward(r_input, is_recompute=True)

        f_attn_pre_r_dipatch_sync_event = F_ATTN_PRE_R_DISPATCH_SYNC_EVENT if is_sync_1f1b else None
        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.attn_qkv.forward(
                    f_input,
                    stream_record_event=f_attn_pre_r_dipatch_sync_event,
                )

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.moe_dispatch.forward(
                        r_input,
                        stream_wait_event=f_attn_pre_r_dipatch_sync_event,
                        is_recompute=True,
                    )

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.core_attn.forward(f_input)
                f_input = f_layer.attn_proj.forward(f_input)
                f_input = f_layer.moe_dispatch.forward(f_input)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.mlp.forward(r_input, is_recompute=True)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = r_layer.moe_combine.forward(r_input, is_recompute=True)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if r_layer is not None:
            with _fork_recompute_rng(r_layer_rng_states):
                with torch.enable_grad(), r_layer.get_fp8_context():
                    r_input = b_layer.mtp_post_process.forward(r_input, is_recompute=True)

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)
            b_grad = b_layer.mlp.backward(b_grad)

        b_attn_post_f_combine_sync_event = B_ATTN_POST_F_COMBINE_SYNC_EVENT if is_sync_1f1b else None
        if b_layer is not None:
            if not block_level_wgrad_compute:
                b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)
            b_grad = b_layer.attn_proj.backward(
                b_grad,
                stream_record_event=b_attn_post_f_combine_sync_event,
            )

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(
                    f_input,
                    stream_wait_event=b_attn_post_f_combine_sync_event
                )

        if b_layer is not None:
            b_grad = b_layer.core_attn.backward(b_grad)
            b_grad = b_layer.attn_qkv.backward(b_grad)

        if f_layer is not None:
            with torch.no_grad(), f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if not block_level_wgrad_compute:
            if b_layer is not None and not is_last_layer_in_bwd:
                b_layer.attn_qkv.backward_dw()
                b_layer.attn_proj.backward_dw()

        if b_layer is not None:
            b_layer.saved_tensors = None
            b_layer.rng_states = None

        return f_input, b_grad


if get_adaptor_args().overlap_ep_comm_with_split_attn:
    TransformerLayerSchedulePlanWithRecompute = TransformerLayerSchedulePlanWithSplitAttn
else:
    TransformerLayerSchedulePlanWithRecompute = TransformerLayerSchedulePlanWithoutSplitAttn
