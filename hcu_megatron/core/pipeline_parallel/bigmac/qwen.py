# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

import inspect
import copy
import warnings
import torch
import torch.distributed as dist
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps, partial
from .communication import all_reduce_grad_buckets, configure_qwen_p2p
from .executor import EncoderOutputMeta, LLMDataIterator


_FORWARD_ROPE = ContextVar("bigmac_forward_rope", default=None)


def _rope_constants(model):
    return (model.config.spatial_merge_size, model.image_token_id,
            model.video_token_id, model.vision_start_token_id)


def encoder_rope_positions(model, input_ids, image_grid_thw):
    # Compute image positions on CPU once; keep text-only positions on device.
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import get_rope_index

    has_image = image_grid_thw is not None and image_grid_thw.numel() > 0
    tokens = input_ids.cpu() if has_image else input_ids
    grid = image_grid_thw.cpu() if has_image else None
    positions, deltas = get_rope_index(
        *_rope_constants(model), tokens,
        image_grid_thw=grid, video_grid_thw=None, attention_mask=None,
    )
    return (positions.to(device=input_ids.device, dtype=input_ids.dtype),
            deltas.to(device=input_ids.device, dtype=input_ids.dtype))


@contextmanager
def precomputed_rope(model, input_ids, image_grid_thw, positions, deltas):
    # Never leave cached data visible to validation, another batch or thread.
    if tuple(positions.shape) != (3, *input_ids.shape):
        raise ValueError("BigMac mRoPE positions do not match the current microbatch")
    if tuple(deltas.shape) != (input_ids.shape[0], 1):
        raise ValueError("BigMac mRoPE deltas do not match the current microbatch")
    for tensor in (positions, deltas):
        if tensor.dtype != input_ids.dtype or tensor.device != input_ids.device:
            raise ValueError("BigMac mRoPE device/dtype differs from input_ids")
    token = _FORWARD_ROPE.set((
        _rope_constants(model), input_ids, image_grid_thw, positions, deltas,
    ))
    try:
        yield
    finally:
        _FORWARD_ROPE.reset(token)


def get_rope_index_wrapper(original):
    signature = inspect.signature(original)

    @wraps(original)
    def get_rope_index(*args, **kwargs):
        cached = _FORWARD_ROPE.get()
        if cached is not None:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            constants, tokens, grid, positions, deltas = cached
            names = ("spatial_merge_size", "image_token_id", "video_token_id", "vision_start_token_id")
            if (tuple(values.get(name) for name in names) == constants
                    and values.get("input_ids") is tokens
                    and values.get("image_grid_thw") is grid
                    and values.get("video_grid_thw") is None
                    and values.get("attention_mask") is None
                    and values.get("packed_seq_params") is None):
                return positions, deltas
        return original(*args, **kwargs)

    return get_rope_index


_HOST_GRID = ContextVar("bigmac_host_vision_grid", default=None)


@contextmanager
def host_vision_grid(vision, grid, device):
    if grid.device.type != "cpu":
        raise ValueError("BigMac vision geometry must be a CPU tensor")
    token = _HOST_GRID.set((vision, grid, device))
    try:
        yield
    finally:
        _HOST_GRID.reset(token)


def build_packed_seq_params_wrapper(original):
    @wraps(original)
    def build_packed_seq_params(self, grid_thw):
        packed = original(self, grid_thw)
        active = _HOST_GRID.get()
        if active is not None and active[0] is self and active[1] is grid_thw:
            # Attention requires device cu_seqlens. Compute their exact native
            # values on CPU, then transfer once; max lengths are host integers.
            # The native method returns a new PackedSeqParams on each call.
            packed.cu_seqlens_q = packed.cu_seqlens_q.to(device=active[2], non_blocking=True)
            packed.cu_seqlens_kv = packed.cu_seqlens_q
            packed.max_seqlen_q = int(packed.max_seqlen_q)
            packed.max_seqlen_kv = int(packed.max_seqlen_kv)
        return packed

    return build_packed_seq_params


def unwrap(model):
    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError("BigMac Qwen3-VL requires a single model chunk")
        model = model[0]
    while hasattr(model, "module") and model.module is not model:
        model = model.module
    return model


def qwen3_model_init_wrapper(original):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()
        language_config = bound.arguments["language_transformer_config"]
        vision_config = bound.arguments["vision_transformer_config"]
        if language_config.hidden_size != vision_config.out_hidden_size:
            raise ValueError(
                "BigMac Qwen3-VL model configuration mismatch: "
                f"language hidden_size={language_config.hidden_size}, "
                f"vision out_hidden_size={vision_config.out_hidden_size}. "
                "Read the language architecture arguments from the selected HF config.json; "
                "do not reuse architecture dimensions from another model size."
            )
        original(self, *args, **kwargs)
        if self.pp_group.size() == 1:
            return
        if self.vision_model is None:
            from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as native
            config = native.get_vision_model_config(
                bound.arguments["vision_transformer_config"], megatron_config=self.config,
            )
            spec = native.get_vit_layer_with_transformer_engine_spec()
            spec.submodules.self_attention.module = native.Qwen3VLSelfAttention
            merger = native.PatchMergerSubmodules(
                patch_norm=native.TENorm,
                linear_fc1=native.TEColumnParallelLinear,
                linear_fc2=native.TERowParallelLinear,
            )
            vision_pgs = copy.copy(self.pg_collection)
            # Each vision tower is complete. In particular, its layer offsets
            # must not be derived from the language model's physical PP rank.
            vision_pgs.pp = dist.new_group(
                ranks=[dist.get_rank()], use_local_synchronization=True,
            )
            self.vision_model = native.Qwen3VLVisionModel(
                config, spec, merger, pre_process=True, post_process=True,
                pg_collection=vision_pgs,
            )
        self._hcu_bigmac_vision = True
        for param in self.vision_model.parameters():
            # Native optimizer master/shard construction propagates this flag.
            # Count a PP-replicated vision parameter once in the global norm.
            if self.pp_group.rank() != 0:
                param.shared = True
    return initialize


def ddp_init_wrapper(original):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()
        raw = unwrap(bound.arguments["module"])
        if getattr(raw, "_hcu_bigmac_vision", False):
            config = copy.copy(bound.arguments["ddp_config"])
            # Sum vision gradients over PP before native DP reduce-scatter.
            # Launching a mixed LLM/vision bucket early would race this sum.
            if config.overlap_grad_reduce and dist.get_rank() == 0:
                warnings.warn(
                    "BigMac Qwen3-VL disables DDP gradient-reduction overlap for "
                    "its model so vision gradients are summed over PP before DP reduction.",
                    stacklevel=2,
                )
            config.overlap_grad_reduce = False
            bound.arguments["ddp_config"] = config
        return original(*bound.args, **bound.kwargs)
    return initialize


def optimizer_init_wrapper(original):
    @wraps(original)
    def initialize(config, model_chunks, *args, **kwargs):
        for chunk in model_chunks:
            model = unwrap(chunk)
            if not getattr(model, "_hcu_bigmac_vision", False):
                continue
            src = dist.get_global_rank(model.pp_group, 0)
            # Synchronize after HF weight loading and before FP32 optimizer
            # masters are constructed, including randomly initialized runs.
            for tensor in list(model.vision_model.parameters()) + list(model.vision_model.buffers()):
                dist.broadcast(tensor.data, src=src, group=model.pp_group)
        return original(config, model_chunks, *args, **kwargs)
    return initialize


def _mark_vision_replicas(state, pp_rank):
    if isinstance(state, dict):
        for value in state.values():
            _mark_vision_replicas(value, pp_rank)
    elif isinstance(state, (list, tuple)):
        for value in state:
            _mark_vision_replicas(value, pp_rank)
    elif hasattr(state, "replica_id") and "vision_model." in getattr(state, "key", ""):
        replica = state.replica_id
        state.replica_id = (*replica, pp_rank) if isinstance(replica, tuple) else (replica, pp_rank)


def sharded_state_dict_wrapper(original):
    @wraps(original)
    def state_dict(self, *args, **kwargs):
        state = original(self, *args, **kwargs)
        if getattr(self, "_hcu_bigmac_vision", False):
            _mark_vision_replicas(state, self.pp_group.rank())
        return state
    return state_dict


def pretrain_wrapper(original):
    signature = inspect.signature(original)

    @wraps(original)
    def pretrain(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        forward_step = bound.arguments["forward_step_func"]
        if not isinstance(getattr(forward_step, "__self__", None), Qwen3VLBigMacBridge):
            namespace = getattr(inspect.unwrap(forward_step), "__globals__", {})
            get_batch = namespace.get("get_batch")
            loss_func = namespace.get("loss_func")
            if not callable(get_batch) or not callable(loss_func):
                raise ValueError("BigMac requires the native Qwen3-VL forward_step, get_batch and loss_func")
            bound.arguments["forward_step_func"] = Qwen3VLBigMacBridge(
                forward_step, get_batch, loss_func,
            ).forward_step
        return original(*bound.args, **bound.kwargs)

    return pretrain


def register_patches(manager):
    manager.register_patch(
        "megatron.training.training.pretrain", pretrain_wrapper, apply_wrapper=True,
    )
    model = "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.Qwen3VLModel"
    manager.register_patch(model + ".__init__", qwen3_model_init_wrapper, apply_wrapper=True)
    manager.register_patch(model + ".sharded_state_dict", sharded_state_dict_wrapper, apply_wrapper=True)
    manager.register_patch(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.get_rope_index",
        get_rope_index_wrapper, apply_wrapper=True,
    )
    manager.register_patch(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model."
        "Qwen3VLVisionModel.build_packed_seq_params",
        build_packed_seq_params_wrapper, apply_wrapper=True,
    )
    manager.register_patch(
        "megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__",
        ddp_init_wrapper, apply_wrapper=True,
    )
    manager.register_patch(
        "megatron.core.optimizer.get_megatron_optimizer", optimizer_init_wrapper, apply_wrapper=True,
    )


class VisionProxy(torch.nn.Module):
    # Expose the existing tower to the executor without registering it twice.
    def __init__(self, parent, vision):
        super().__init__()
        object.__setattr__(self, "parent", parent)
        self.vision_model = vision

    @property
    def module(self):
        return self.vision_model

    @property
    def ddp_config(self):
        return getattr(self.parent, "ddp_config", None)

    def no_sync(self):
        return self.parent.no_sync()

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)


@contextmanager
def cached_vision_output(vision, embeds, deepstack):
    # Limit the handoff to one call; validation retains the native forward.
    sentinel = object()
    previous = vision.__dict__.get("forward", sentinel)
    vision.forward = lambda *args, **kwargs: (embeds, deepstack)
    try:
        yield
    finally:
        if previous is sentinel:
            del vision.forward
        else:
            vision.forward = previous


class Qwen3VLBigMacBridge:
    def __init__(self, native_forward_step, get_batch, loss_func):
        self.native_forward_step = native_forward_step
        self.get_batch = get_batch
        self.loss_func = loss_func
        self.model = None
        self._runtime_reported = False
        self.reset_pp_executor_encoder_forward_state()

    def reset_pp_executor_encoder_forward_state(self):
        self.loaded = 0
        self.num_microbatches = None
        self.data_iterator = None

    def get_encoder_models(self, model):
        parent = model[0] if isinstance(model, list) else model
        self.model = unwrap(parent)
        if not getattr(self.model, "_hcu_bigmac_vision", False):
            raise RuntimeError("Qwen3-VL was built without the BigMac feature enabled")
        return [VisionProxy(parent, self.model.vision_model)]

    def get_encoder_output_meta(self):
        hidden = self.model.config.hidden_size
        self.dtype = next(self.model.vision_model.parameters()).dtype
        self.deepstack_count = len(self.model.vision_model.config.deepstack_visual_indexes)
        metas = [
            EncoderOutputMeta("input_ids", shape=(-1, -1), dtype=torch.int64, seq_dim=1),
            EncoderOutputMeta("labels", shape=(-1, -1), dtype=torch.int64, seq_dim=1),
            EncoderOutputMeta("loss_mask", shape=(-1, -1), dtype=torch.float32, seq_dim=1),
            EncoderOutputMeta("image_grid_thw", shape=(-1, 3), dtype=torch.int64),
            EncoderOutputMeta("image_input_mask", shape=(-1, -1), dtype=torch.bool, seq_dim=1),
            EncoderOutputMeta("position_ids", shape=(3, -1, -1), dtype=torch.int64, seq_dim=2),
            EncoderOutputMeta("mrope_position_deltas", shape=(-1, 1), dtype=torch.int64),
            EncoderOutputMeta("vision_embeds", shape=(-1, hidden), dtype=self.dtype,
                              requires_grad=True, only_requires_in_first_stage=True),
        ]
        metas.extend(
            EncoderOutputMeta(f"deepstack_{i}", shape=(-1, hidden), dtype=self.dtype,
                              requires_grad=True, only_requires_in_first_stage=True)
            for i in range(self.deepstack_count)
        )
        return metas

    def configure_pp_executor(self, executor):
        transport = configure_qwen_p2p(executor.p2p_communicator)
        if not self._runtime_reported:
            if dist.get_rank() == 0:
                print(
                    f"[BigMac Qwen3-VL] source={__file__}; "
                    f"PP={executor.pp_size}, microbatches={executor.num_microbatches}, "
                    f"VE/rank={executor.num_microbatches // executor.pp_size}; "
                    f"p2p={transport}, vision_grid=cpu, mrope=encoder, "
                    "vision_grad=bucketed_pp_sum", flush=True,
                )
            self._runtime_reported = True

    def encoder_forward_func(self, encoder_models, data_iterator, *, num_microbatches, microbatch_id, **kwargs):
        self.num_microbatches = num_microbatches
        self.data_iterator = data_iterator
        # Every PP stage's native loader has the same DP stream. Advance past
        # samples owned by other encoder ranks without retaining their pixels.
        batch = self._get_encoder_batch(data_iterator, microbatch_id)
        tokens, labels, mask, attention, positions, pixels, grid, image_mask, _, _ = batch
        if attention is not None or positions is not None:
            raise ValueError("BigMac Qwen3-VL expects the native un-packed dataset contract")
        if image_mask is None:
            image_mask = tokens == self.model.image_token_id
        has_image = pixels is not None and grid is not None and grid.numel() > 0
        # Reuse the same post-TP-broadcast host grid for positions and vision.
        # Copy before launching the tower, so its Python shape reads cannot
        # synchronize patch embedding, attention, or the merger mid-forward.
        host_grid = grid.cpu() if has_image else None
        positions, deltas = encoder_rope_positions(self.model, tokens, host_grid)
        if has_image:
            with host_vision_grid(self.model.vision_model, host_grid, pixels.device):
                embeds, deepstack = encoder_models[0](
                    hidden_states=pixels.to(self.dtype), grid_thw=host_grid,
                )
            deepstack = list(deepstack or [])
        else:
            grid = tokens.new_empty((0, 3))
            embeds = torch.empty((0, self.model.config.hidden_size), dtype=self.dtype, device=tokens.device)
            deepstack = [torch.empty_like(embeds) for _ in range(self.deepstack_count)]
        if len(deepstack) != self.deepstack_count:
            raise ValueError("Qwen3-VL DeepStack output count differs from the vision configuration")
        return [tokens, labels, mask, grid, image_mask, positions, deltas, embeds, *deepstack]

    def _get_encoder_batch(self, data_iterator, microbatch_id):
        if microbatch_id < self.loaded:
            raise ValueError("BigMac encoder microbatches must be consumed in order")
        # Native DAS get_batch moves tensors to the device and broadcasts over
        # TP. All TP ranks here own the same VE microbatch; unowned batches only
        # need to advance the native CPU iterator to preserve sample/RNG order.
        while self.loaded < microbatch_id:
            next(data_iterator)
            self.loaded += 1
        batch = self.get_batch(data_iterator)
        self.loaded += 1
        return batch

    def forward_step(self, data_iterator, model, **kwargs):
        if not isinstance(data_iterator, LLMDataIterator):
            return self.native_forward_step(data_iterator, model, **kwargs)
        batch = next(data_iterator)
        raw = unwrap(model)
        grid = batch["image_grid_thw"]
        has_image = grid.numel() > 0
        model_kwargs = dict(
            input_ids=batch["input_ids"], labels=batch["labels"], loss_mask=batch["loss_mask"],
            attention_mask=None, position_ids=None,
            image_grid_thw=grid if has_image else None,
            image_input_mask=batch["image_input_mask"],
            # CP=1 and image-only DAS batches: Bridge only uses this nonempty
            # placeholder to reach its vision call, whose outputs are cached.
            pixel_values=batch["loss_mask"].new_empty((1, 1)) if has_image else None,
        )
        # Native Bridge resets its position_ids argument internally. Substitute
        # only its matching get_rope_index call while retaining native forward,
        # including embeddings, DeepStack, SP and loss behavior.
        with precomputed_rope(
            raw, batch["input_ids"], grid if has_image else None,
            batch["position_ids"], batch["mrope_position_deltas"],
        ):
            if raw.pre_process:
                with cached_vision_output(
                    raw.vision_model, batch["vision_embeds"],
                    [batch[f"deepstack_{i}"] for i in range(self.deepstack_count)],
                ):
                    output = model(**model_kwargs)
            else:
                output = model(**model_kwargs)
        return output, partial(self.loss_func, batch["loss_mask"], model=model)

    def prepare_encoder_grads(self, executor):
        # All ranks finish their scheduled encoder backward before entering
        # native DP reduction. Sum (do not average) the disjoint PP units.
        grads = []
        for param in reversed(list(self.model.vision_model.parameters())):
            if not param.requires_grad:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                grad = param.grad
            grads.append(grad)
        all_reduce_grad_buckets(grads, executor.pp_group)

    def finish_pp_executor_encoder_iteration(self, executor):
        if self.num_microbatches is not None:
            while self.loaded < self.num_microbatches:
                next(self.data_iterator)
                self.loaded += 1
        self.data_iterator = None
