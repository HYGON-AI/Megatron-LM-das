###############################################################################
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE for license information.
###############################################################################

"""EP8 four-way matrix test for the MegaMoE x UltraEP feature combination.

One Qwen3-30B-A3B MoE layer (hidden 2048, moe-ffn 768, 128 experts, top-8) on a
single eight-HCU node.  Four variants share identical router weights, expert
weights and inputs, so their outputs and gradients are directly comparable:

    baseline           native Megatron ``MoELayer`` (TEGroupedMLP experts)
    megamoe            ``PrimusTurboMegaMoELayer``
    ultraep            native ``MoELayer`` + UltraEP redundant experts
    megamoe_ultraep    ``PrimusTurboMegaMoELayer`` + UltraEP (the fused path)

Accuracy: every variant is compared against ``baseline`` on the forward output,
the input gradient, the router weight gradient and the expert weight gradients
accumulated over several gradient-accumulation micro-batches.

Performance: forward-only and forward+backward wall time per micro-batch is
measured for each variant and printed as a table.  Timings are reported, not
asserted -- they depend on the machine.

Run with::

    bash tests/unit_tests/moe/run_mega_moe_ultraep_matrix.sh

or, with the DTK / rocSHMEM environment already sourced::

    python -m torch.distributed.run --nproc-per-node=8 -m pytest -s -vv \
        tests/unit_tests/moe/test_mega_moe_ultraep_matrix.py

Knobs (environment variables): ``MEGAMOE_MATRIX_SEQ_LEN``,
``MEGAMOE_MATRIX_NUM_GA``, ``MEGAMOE_MATRIX_PERF_ITERS``,
``MEGAMOE_MATRIX_PERF_WARMUP``, ``MEGAMOE_MATRIX_COSINE_FLOOR``,
``MEGAMOE_MATRIX_VARIANTS`` (comma separated subset).
"""

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed

from hcu_megatron.core.extensions.mega_moe import HAVE_TURBO, PrimusTurboMegaMoELayer
from hcu_megatron.core.transformer.moe.eplb_manager import (
    HAVE_EPLB,
    clear_eplb_manager_registry,
)
from hcu_megatron.core.transformer.moe.moe_layer_ultraep import (
    moe_layer_ultraep_forward_wrapper,
    moe_layer_ultraep_init_wrapper,
)

# ---------------------------------------------------------------------------
# Qwen3-30B-A3B MoE shapes (examples/qwen3/train_qwen3_30B_A3B_ultraep.sh)
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 2048
MOE_FFN_HIDDEN_SIZE = 768
NUM_MOE_EXPERTS = 128
MOE_ROUTER_TOPK = 8
NUM_ATTENTION_HEADS = 32
MICRO_BATCH_SIZE = 1
REDUNDANT_EXPERTS_PER_RANK = 2
# The UltraEP manager is sized by config.num_layers; the two UltraEP variants
# register master pointers under distinct layer ids of one shared manager.
NUM_LAYERS = 4

SEQ_LEN = int(os.environ.get("MEGAMOE_MATRIX_SEQ_LEN", 4096))
NUM_GA = int(os.environ.get("MEGAMOE_MATRIX_NUM_GA", 2))
PERF_WARMUP = int(os.environ.get("MEGAMOE_MATRIX_PERF_WARMUP", 3))
PERF_ITERS = int(os.environ.get("MEGAMOE_MATRIX_PERF_ITERS", 10))
COSINE_FLOOR = float(os.environ.get("MEGAMOE_MATRIX_COSINE_FLOOR", 0.95))

WEIGHT_SEED = 20261123
ROUTER_SEED = 777
INPUT_SEED = 4242
INIT_SEED = 123

# rocSHMEM / IPC settings UltraEP needs for a single-node EP group.  They have
# to be in place before the first ultra_ep.Manager initializes the runtime.
ULTRAEP_RUNTIME_ENV = {
    "HSA_USE_SVM": "0",
    "MAX_NUM_NVL_PEERS": "8",
    "ROCSHMEM_BACKEND": "ipc",
    "ROCSHMEM_HEAP_SIZE": os.environ.get(
        "MEGAMOE_MATRIX_ROCSHMEM_HEAP_SIZE", str(2 * 1024**3)
    ),
}


@dataclass(frozen=True)
class Variant:
    """One cell of the feature matrix."""

    name: str
    mega_moe: bool
    ultraep: bool
    layer_number: int


ALL_VARIANTS = (
    Variant("baseline", mega_moe=False, ultraep=False, layer_number=1),
    Variant("megamoe", mega_moe=True, ultraep=False, layer_number=1),
    Variant("ultraep", mega_moe=False, ultraep=True, layer_number=1),
    Variant("megamoe_ultraep", mega_moe=True, ultraep=True, layer_number=2),
)
BASELINE = ALL_VARIANTS[0].name


def _selected_variants():
    requested = os.environ.get("MEGAMOE_MATRIX_VARIANTS")
    if not requested:
        return ALL_VARIANTS
    wanted = [name.strip() for name in requested.split(",") if name.strip()]
    known = {variant.name for variant in ALL_VARIANTS}
    unknown = sorted(set(wanted) - known)
    assert not unknown, f"unknown variants {unknown}; known: {sorted(known)}"
    assert BASELINE in wanted, f"'{BASELINE}' is the accuracy reference and must be selected"
    return tuple(variant for variant in ALL_VARIANTS if variant.name in wanted)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().float() - b.detach().float()).abs().max().item()


def _compare(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    return {
        "cos": _cosine(actual, reference),
        "rel": _relative_error(actual, reference),
        "max_abs": _max_abs_diff(actual, reference),
    }


def _all_reduce_scalar(value: float, op) -> float:
    """Reduce a rank-local metric so every rank asserts on the same number."""
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return tensor.item()


# ---------------------------------------------------------------------------
# Shared reference parameters
#
# Every variant is loaded from the same deterministic CPU draw, so differences
# in Megatron's init order / RNG forking cannot leak into the comparison.
# ---------------------------------------------------------------------------


def _reference_expert_weights(logical_expert_id: int):
    generator = torch.Generator().manual_seed(WEIGHT_SEED + logical_expert_id)
    # fc1 is the fused gate+up projection: [2 * ffn, hidden].
    fc1 = torch.empty(2 * MOE_FFN_HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float32)
    fc1.normal_(0.0, 0.02, generator=generator)
    fc2 = torch.empty(HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE, dtype=torch.float32)
    fc2.normal_(0.0, 0.02, generator=generator)
    return fc1.to(torch.bfloat16), fc2.to(torch.bfloat16)


def _reference_router_weight():
    generator = torch.Generator().manual_seed(ROUTER_SEED)
    weight = torch.empty(NUM_MOE_EXPERTS, HIDDEN_SIZE, dtype=torch.float32)
    weight.normal_(0.0, 0.02, generator=generator)
    return weight


def _create_args(variant: Variant):
    """Minimal ``megatron.training`` args touched by the MoE code paths."""
    args = SimpleNamespace()
    args.sequence_parallel = False
    args.context_parallel_size = 1
    args.micro_batch_size = MICRO_BATCH_SIZE
    args.moe_router_force_load_balancing = False
    args.moe_use_legacy_grouped_gemm = True
    args.use_turbo_grouped_gemm = False
    args.enable_primus_turbo = False
    args.moe_use_fused_router_with_aux_score = False
    args.router_logit_softcapping = None
    args.mega_moe = variant.mega_moe
    args.turbo_mega_moe_precision = "bf16"
    args.moe_enable_ultraep = variant.ultraep
    args.moe_num_redundant_experts_per_rank = (
        REDUNDANT_EXPERTS_PER_RANK if variant.ultraep else 0
    )
    return args


def _build_config(variant: Variant, ep_size: int) -> TransformerConfig:
    """One-layer Qwen3-30B-A3B MoE config for the requested variant."""
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        pipeline_model_parallel_size=1,
        num_moe_experts=NUM_MOE_EXPERTS,
        moe_ffn_hidden_size=MOE_FFN_HIDDEN_SIZE,
        moe_router_topk=MOE_ROUTER_TOPK,
        # Qwen3 uses plain softmax top-k routing without expert grouping.
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        # aux loss off: keeps every weight's gradient a clean parity check.
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        # Qwen3-30B-A3B has routed experts only (no shared expert).
        moe_shared_expert_intermediate_size=None,
        moe_token_dispatcher_type="alltoall",
        # fp64 avoids a broken TE general_gemm(workspace=) path in this environment.
        moe_router_dtype="fp64",
        # MegaMoE packs the experts itself; the native layer needs TEGroupedMLP
        # because UltraEP's redundant-expert registration requires it.
        moe_grouped_gemm=not variant.mega_moe,
        gated_linear_unit=True,
        activation_func=F.silu,  # MegaMoE hardcodes SwiGLU/SiLU
        add_bias_linear=False,
        bias_activation_fusion=False,
        use_cpu_initialization=False,
        # Expert wgrad is routed into main_grad by this test's hooks, not by TE.
        gradient_accumulation_fusion=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    # Fields that hcu_megatron injects into TransformerConfig from the adaptor
    # args at runtime; the standalone harness sets them directly.
    config.moe_enable_ultraep = variant.ultraep
    config.moe_num_redundant_experts_per_rank = (
        REDUNDANT_EXPERTS_PER_RANK if variant.ultraep else 0
    )
    config.moe_ultraep_autotune = False
    config.moe_ultraep_autotune_start_iteration = 3
    config.moe_ultraep_autotune_grad_reduce_max_sms = None
    return config


_ULTRAEP_MOE_LAYER_CLASS = None


def _ultraep_moe_layer_class():
    """``MoELayer`` with the UltraEP init/forward wrappers applied.

    A subclass keeps the wrappers off the global ``MoELayer``, so the baseline
    variant in the same process stays untouched.
    """
    global _ULTRAEP_MOE_LAYER_CLASS
    if _ULTRAEP_MOE_LAYER_CLASS is None:

        class UltraEPMoELayer(MoELayer):
            """Native MoE layer running on UltraEP physical experts."""

        UltraEPMoELayer.__init__ = moe_layer_ultraep_init_wrapper(MoELayer.__init__)
        UltraEPMoELayer.forward = moe_layer_ultraep_forward_wrapper(MoELayer.forward)
        _ULTRAEP_MOE_LAYER_CLASS = UltraEPMoELayer
    return _ULTRAEP_MOE_LAYER_CLASS


def _move_to_cuda(layer: torch.nn.Module) -> None:
    """Place parameters on the local device without detaching UltraEP buffers.

    ``Module.cuda()`` would re-allocate the replica parameters that UltraEP
    points at its own shared buffers, silently breaking weight sync.
    """
    for param in layer.parameters():
        if getattr(param, "is_eplb_replica", False):
            assert param.is_cuda, "UltraEP replica parameters must already be on device"
            continue
        if not param.is_cuda:
            param.data = param.data.cuda()
    for buffer_name, buffer in list(layer.named_buffers()):
        if buffer is not None and not buffer.is_cuda:
            module_path, _, attribute = buffer_name.rpartition(".")
            module = layer.get_submodule(module_path) if module_path else layer
            setattr(module, attribute, buffer.cuda())


def _expert_master_weights(layer, variant: Variant, experts_per_rank: int):
    """Return (fc1, fc2) master weight parameter lists for the logical experts."""
    if variant.mega_moe:
        # MegaMoE keeps all local experts in one packed [g, ...] parameter.
        return [layer.experts.fc1_weight.weight], [layer.experts.fc2_weight.weight]
    fc1 = [
        getattr(layer.experts.linear_fc1, f"weight{index}")
        for index in range(experts_per_rank)
    ]
    fc2 = [
        getattr(layer.experts.linear_fc2, f"weight{index}")
        for index in range(experts_per_rank)
    ]
    return fc1, fc2


def _load_reference_weights(layer, variant: Variant, ep_rank: int, experts_per_rank: int):
    first_expert = ep_rank * experts_per_rank
    with torch.no_grad():
        layer.router.weight.copy_(_reference_router_weight())
        for attribute in ("bias", "expert_bias"):
            tensor = getattr(layer.router, attribute, None)
            if tensor is not None:
                tensor.zero_()

        fc1_params, fc2_params = _expert_master_weights(layer, variant, experts_per_rank)
        for local_index in range(experts_per_rank):
            fc1, fc2 = _reference_expert_weights(first_expert + local_index)
            if variant.mega_moe:
                fc1_params[0][local_index].copy_(fc1)
                fc2_params[0][local_index].copy_(fc2)
            else:
                fc1_params[local_index].copy_(fc1)
                fc2_params[local_index].copy_(fc2)


def _accumulate_into_main_grad(param: torch.nn.Parameter) -> None:
    """Mirror Megatron's DDP hook: fold ``.grad`` into the fp32 ``main_grad``.

    UltraEP's grad reduce reads the replica ``main_grad`` buffers during
    backward, so autograd's ``.grad`` has to land there before it runs.  Doing
    the same for every variant also makes the accumulated expert gradients
    comparable: all of them accumulate over micro-batches in fp32.
    """
    if param.grad is None:
        return
    param.main_grad.add_(param.grad.to(param.main_grad.dtype))
    param.grad = None


def _prepare_grad_buffers(layer: torch.nn.Module, variant: Variant, experts_per_rank: int) -> None:
    """Give every trainable parameter an fp32 ``main_grad`` plus the hook.

    UltraEP replica parameters already carry a ``main_grad`` view into the
    runtime's shared grad buffer; those must be kept as-is.
    """
    mega_expert_params = set()
    if variant.mega_moe:
        fc1_params, fc2_params = _expert_master_weights(layer, variant, experts_per_rank)
        mega_expert_params = {id(param) for param in (*fc1_params, *fc2_params)}

    for param in layer.parameters():
        if not param.requires_grad:
            continue
        if getattr(param, "main_grad", None) is None:
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)
        if id(param) in mega_expert_params:
            # primus_turbo writes MegaMoE wgrad straight into main_grad, but only
            # when Megatron's companion flag is present as well.
            param.grad_added_to_main_grad = False
        param.register_post_accumulate_grad_hook(_accumulate_into_main_grad)


def _zero_grads(layer: torch.nn.Module) -> None:
    for param in layer.parameters():
        param.grad = None
        main_grad = getattr(param, "main_grad", None)
        if main_grad is not None:
            main_grad.zero_()


def _stacked_expert_grads(layer, variant: Variant, experts_per_rank: int):
    """Accumulated expert wgrads as [local_experts, ...] tensors."""
    fc1_params, fc2_params = _expert_master_weights(layer, variant, experts_per_rank)
    if variant.mega_moe:
        return fc1_params[0].main_grad.clone(), fc2_params[0].main_grad.clone()
    return (
        torch.stack([param.main_grad for param in fc1_params]).clone(),
        torch.stack([param.main_grad for param in fc2_params]).clone(),
    )


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------


def _build_layer(variant: Variant, ep_size: int, pg_collection, ep_rank: int):
    # The router and MegaMoE read per-variant switches through get_args().
    set_args(_create_args(variant))
    config = _build_config(variant, ep_size)
    experts_per_rank = NUM_MOE_EXPERTS // ep_size

    # Identical seed for every variant; the explicit weight load below makes the
    # starting point identical regardless, this only keeps any auxiliary buffer
    # (e.g. router expert bias) in the same state.
    _set_random_seed(seed_=INIT_SEED, data_parallel_random_init=False)

    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=NUM_MOE_EXPERTS, moe_grouped_gemm=not variant.mega_moe
    )
    submodules = get_submodules(spec.submodules.mlp)

    if variant.mega_moe:
        layer = PrimusTurboMegaMoELayer(
            config,
            submodules,
            layer_number=variant.layer_number,
            pg_collection=pg_collection,
        )
    else:
        layer_class = _ultraep_moe_layer_class() if variant.ultraep else MoELayer
        layer = layer_class(
            config,
            submodules,
            layer_number=variant.layer_number,
            pg_collection=pg_collection,
        )

    _move_to_cuda(layer)
    _load_reference_weights(layer, variant, ep_rank, experts_per_rank)
    _prepare_grad_buffers(layer, variant, experts_per_rank)
    layer.train(True)
    return layer, experts_per_rank


def _forward_backward(layer, x: torch.Tensor, grad_out: torch.Tensor):
    x = x.clone().requires_grad_(True)
    output, _ = layer(x)
    output.backward(grad_out)
    return output.detach(), x.grad.detach()


def _measure(layer, x: torch.Tensor, grad_out: torch.Tensor) -> dict:
    """Forward-only and forward+backward time per micro-batch, in ms."""

    def forward_only():
        output, _ = layer(x.clone().requires_grad_(True))
        return output

    for _ in range(PERF_WARMUP):
        _forward_backward(layer, x, grad_out)
        _zero_grads(layer)
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(PERF_ITERS):
        forward_only()
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - start) * 1000.0 / PERF_ITERS

    _zero_grads(layer)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(PERF_ITERS):
        _forward_backward(layer, x, grad_out)
    torch.cuda.synchronize()
    forward_backward_ms = (time.perf_counter() - start) * 1000.0 / PERF_ITERS

    _zero_grads(layer)
    # The slowest rank sets the step time in real training.
    return {
        "forward_ms": _all_reduce_scalar(forward_ms, dist.ReduceOp.MAX),
        "forward_backward_ms": _all_reduce_scalar(
            forward_backward_ms, dist.ReduceOp.MAX
        ),
    }


def _run_variant(variant: Variant, ep_size: int, pg_collection, ep_rank, inputs, grads):
    layer, experts_per_rank = _build_layer(variant, ep_size, pg_collection, ep_rank)
    try:
        outputs, input_grads = [], []
        for step in range(NUM_GA):
            output, input_grad = _forward_backward(layer, inputs[step], grads[step])
            outputs.append(output.clone())
            input_grads.append(input_grad.clone())

        expert_dw1, expert_dw2 = _stacked_expert_grads(layer, variant, experts_per_rank)
        tensors = {
            "router_dw": layer.router.weight.main_grad.clone(),
            "expert_dw1": expert_dw1,
            "expert_dw2": expert_dw2,
        }
        for step in range(NUM_GA):
            tensors[f"fwd[{step}]"] = outputs[step]
            tensors[f"dx[{step}]"] = input_grads[step]

        timings = _measure(layer, inputs[0], grads[0])
        return tensors, timings
    finally:
        del layer
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def distributed_environment():
    """Initialize the process group supplied by ``torchrun`` once per worker."""
    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(name in os.environ for name in required_env):
        pytest.skip("run this EP8 test with torchrun --nproc-per-node=8")

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        pytest.skip(f"this test requires exactly 8 processes, got {world_size}")
    if torch.cuda.device_count() < 8:
        pytest.skip(f"this test requires 8 HCUs, got {torch.cuda.device_count()}")
    if not HAVE_TURBO:
        pytest.skip("primus_turbo is required for the MegaMoE variants")
    if not HAVE_EPLB:
        pytest.skip("ultra_ep is required for the UltraEP variants")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with patch.dict(os.environ, ULTRAEP_RUNTIME_ENV, clear=False):
        os.environ.pop("ROCSHMEM_GDA_PROVIDER", None)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=world_size,
        )
        try:
            yield SimpleNamespace(
                world_size=world_size,
                rank=dist.get_rank(),
                local_rank=local_rank,
            )
        finally:
            # The two UltraEP variants share one runtime; drop it only at the
            # very end so no second symmetric heap is ever allocated.
            clear_eplb_manager_registry()
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def matrix_results(distributed_environment):
    """Run every selected variant once; reused by the accuracy and perf tests."""
    environment = distributed_environment
    variants = _selected_variants()
    ep_size = environment.world_size
    rank = environment.rank

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    assert pg_collection.ep is parallel_state.get_expert_model_parallel_group(), (
        "MegaMoE and the native UltraEP layer must share one EP group, otherwise "
        "each would build its own UltraEP runtime"
    )
    ep_rank = pg_collection.ep.rank()

    # Inputs differ per rank (realistic routing imbalance) but are identical
    # across variants.
    inputs, grads = [], []
    for step in range(max(NUM_GA, 1)):
        generator = torch.Generator(device="cuda").manual_seed(
            INPUT_SEED + rank * 1000 + step
        )
        shape = (SEQ_LEN, MICRO_BATCH_SIZE, HIDDEN_SIZE)
        inputs.append(
            torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
        )
        grads.append(
            torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
        )

    tensors, timings = {}, {}
    for variant in variants:
        if rank == 0:
            print(f"\n[matrix] running variant '{variant.name}'", flush=True)
        tensors[variant.name], timings[variant.name] = _run_variant(
            variant, ep_size, pg_collection, ep_rank, inputs, grads
        )
        dist.barrier()

    # Accuracy is measured against the baseline on each rank, then reduced so
    # every rank asserts on the same (worst) numbers.
    accuracy = {}
    for variant in variants:
        if variant.name == BASELINE:
            continue
        accuracy[variant.name] = {
            key: _compare(tensors[variant.name][key], tensors[BASELINE][key])
            for key in tensors[BASELINE]
        }

    # Free the per-variant tensors before the tests run.
    tensors.clear()
    torch.cuda.empty_cache()

    reduced = {}
    for name, metrics in accuracy.items():
        reduced[name] = {
            key: {
                "cos": _all_reduce_scalar(value["cos"], dist.ReduceOp.MIN),
                "rel": _all_reduce_scalar(value["rel"], dist.ReduceOp.MAX),
                "max_abs": _all_reduce_scalar(value["max_abs"], dist.ReduceOp.MAX),
            }
            for key, value in metrics.items()
        }

    return SimpleNamespace(
        variants=variants,
        accuracy=reduced,
        timings=timings,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMegaMoEUltraEPMatrix:
    @staticmethod
    def _print(rank, message):
        if rank == 0:
            print(message, flush=True)

    def test_accuracy(self, matrix_results):
        """Every variant must match the native MoE layer within tolerance."""
        rank = matrix_results.rank
        self._print(
            rank,
            f"\n=== accuracy vs '{BASELINE}' "
            f"(seq={SEQ_LEN}, ga={NUM_GA}, floor={COSINE_FLOOR}) ===\n"
            f"{'variant':<18}{'tensor':<14}{'cosine':>12}{'rel_err':>12}{'max_abs':>12}",
        )

        failures = []
        for name, metrics in matrix_results.accuracy.items():
            for key in sorted(metrics):
                value = metrics[key]
                self._print(
                    rank,
                    f"{name:<18}{key:<14}{value['cos']:>12.6f}"
                    f"{value['rel']:>12.3e}{value['max_abs']:>12.3e}",
                )
                # Negated compare so a NaN cosine fails instead of slipping through.
                if not value["cos"] > COSINE_FLOOR:
                    failures.append(
                        f"{name}/{key} cosine {value['cos']:.6f} <= {COSINE_FLOOR}"
                    )

        assert failures == [], "accuracy parity failures:\n" + "\n".join(failures)

    def test_performance(self, matrix_results):
        """Report per-variant timings; speedups are informational."""
        rank = matrix_results.rank
        timings = matrix_results.timings
        reference = timings.get(BASELINE)

        self._print(
            rank,
            f"\n=== performance "
            f"(seq={SEQ_LEN}, mbs={MICRO_BATCH_SIZE}, iters={PERF_ITERS}, "
            f"slowest rank) ===\n"
            f"{'variant':<18}{'fwd ms':>12}{'fwd+bwd ms':>14}"
            f"{'fwd speedup':>14}{'fwd+bwd speedup':>18}",
        )
        for variant in matrix_results.variants:
            value = timings[variant.name]
            forward_speedup = reference["forward_ms"] / value["forward_ms"]
            total_speedup = (
                reference["forward_backward_ms"] / value["forward_backward_ms"]
            )
            self._print(
                rank,
                f"{variant.name:<18}{value['forward_ms']:>12.3f}"
                f"{value['forward_backward_ms']:>14.3f}"
                f"{forward_speedup:>14.3f}x{total_speedup:>17.3f}x",
            )

        for name, value in timings.items():
            assert value["forward_ms"] > 0.0, f"{name}: no forward time measured"
            assert value["forward_backward_ms"] >= value["forward_ms"], (
                f"{name}: forward+backward ({value['forward_backward_ms']:.3f} ms) "
                f"is faster than forward alone ({value['forward_ms']:.3f} ms)"
            )
