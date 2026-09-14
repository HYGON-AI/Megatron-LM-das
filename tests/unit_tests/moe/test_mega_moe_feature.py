###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE for license information.
###############################################################################

"""EP8 parity test: fused ``MegaMoE`` vs Megatron ``MoELayer``.

Run with::

    python -m torch.distributed.run --nproc-per-node=8 -m pytest -s -vv \
        tests/unit_tests/moe/test_mega_moe_feature.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed

from hcu_megatron.core.extensions.mega_moe import (
    PrimusTurboMegaMoELayer,
    HAVE_TURBO,
)


def _create_args():
    args = SimpleNamespace()
    args.sequence_parallel = False
    args.context_parallel_size = 1
    args.micro_batch_size = 1
    args.moe_router_force_load_balancing = False
    args.moe_use_legacy_grouped_gemm = True
    args.use_turbo_grouped_gemm = False
    args.enable_primus_turbo = False
    args.moe_use_fused_router_with_aux_score = False
    args.router_logit_softcapping = None
    return args


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


@pytest.fixture(scope="module", autouse=True)
def distributed_test_environment():
    """Initialize the process group supplied by ``torchrun`` once per worker."""
    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(name in os.environ for name in required_env):
        pytest.skip("run this EP8 test with torchrun --nproc-per-node=8")

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        pytest.skip(f"this test requires exactly 8 processes, got {world_size}")
    if torch.cuda.device_count() < 8:
        pytest.skip(f"this test requires 8 GPUs, got {torch.cuda.device_count()}")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _build_config(ep_size, num_moe_experts, moe_router_topk):
    """One-layer Qwen3-30B-A3B MoE config."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=32,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        pipeline_model_parallel_size=1,
        num_moe_experts=num_moe_experts,
        moe_ffn_hidden_size=768,
        moe_router_topk=moe_router_topk,
        # Qwen3 uses standard softmax top-k routing without expert grouping.
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        # aux loss off: keeps every weight's grad a clean fwd/bwd parity check
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        # Qwen3-30B-A3B has routed experts only (no shared expert).
        moe_shared_expert_intermediate_size=None,
        moe_token_dispatcher_type="alltoall",
        # fp64 avoids a broken TE general_gemm(workspace=) path in this test environment
        moe_router_dtype="fp64",
        moe_grouped_gemm=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,  # MegaMoE hardcodes SwiGLU/SiLU
        add_bias_linear=False,
        bias_activation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _build_layers(config, num_moe_experts, num_layers):
    """Build ``num_layers`` matched (Megatron MoELayer, MegaMoE) pairs.

    Each layer gets its own seed -> distinct-but-matched weights; ref weights
    are copied into the mega layer so per-layer outputs are directly comparable.
    """
    from megatron.core.transformer.spec_utils import get_submodules

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    moe_layers, mega_layers = [], []
    for layer_idx in range(num_layers):
        _set_random_seed(seed_=123 + layer_idx, data_parallel_random_init=False)

        spec = get_gpt_layer_local_spec(num_experts=num_moe_experts, moe_grouped_gemm=False)
        # submodules = spec.submodules.mlp.submodules
        spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts,
            moe_grouped_gemm=False,
        )
        submodules = get_submodules(spec.submodules.mlp)

        moe_layer = MoELayer(config, submodules).cuda().to(torch.bfloat16)
        moe_layer.set_layer_number(layer_idx)

        # MegaMoE builds the same router type as the reference layer.
        mega_layer = PrimusTurboMegaMoELayer(
            config, submodules, layer_number=layer_idx, pg_collection=pg_collection
        ).cuda()

        experts = moe_layer.experts.local_experts
        mega_w1 = mega_layer.experts.fc1_weight.weight
        mega_w2 = mega_layer.experts.fc2_weight.weight
        assert len(experts) == mega_w1.shape[0]
        with torch.no_grad():
            # match routing: copy router gate weight into mega's own router
            mega_layer.router.weight.copy_(moe_layer.router.weight)
            for r in (moe_layer.router, mega_layer.router):
                if getattr(r, "bias", None) is not None:
                    r.bias.zero_()
                # Keep optional router biases identical for reusable configurations.
                if getattr(r, "expert_bias", None) is not None:
                    r.expert_bias.zero_()
            # routed experts: grouped fc1/fc2 -> packed w1/w2
            for i, expert in enumerate(experts):
                assert expert.linear_fc1.weight.shape == mega_w1[i].shape
                assert expert.linear_fc2.weight.shape == mega_w2[i].shape
                mega_w1[i].copy_(expert.linear_fc1.weight.to(torch.bfloat16))
                mega_w2[i].copy_(expert.linear_fc2.weight.to(torch.bfloat16))
            # Copy a shared expert when a tested model configuration has one.
            if mega_layer.shared_experts is not None:
                se_ref, se_mega = moe_layer.shared_experts, mega_layer.shared_experts
                se_mega.linear_fc1.weight.copy_(se_ref.linear_fc1.weight)
                se_mega.linear_fc2.weight.copy_(se_ref.linear_fc2.weight)
        moe_layers.append(moe_layer)
        mega_layers.append(mega_layer)
    return moe_layers, mega_layers


class TestMegaMoEAccuracy:
    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    def _init_process(self):
        torch.cuda.set_device(self.device)
        set_args(_create_args())
        # standalone harness has no Primus logger; attach a plain one so any
        # log_rank_0 call in the code path doesn't crash
        # import logging

        # from primus.core.utils import logger as _primus_logger

        # if _primus_logger._logger is None:
        #     _primus_logger._logger = logging.getLogger("mega_moe_test")

    def _setup_ep(self):
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=self.world_size,
        )

    def _rank0(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    @pytest.mark.skipif(not HAVE_TURBO, reason="primus_turbo is required")
    @pytest.mark.parametrize("moe_router_topk", [8])
    @pytest.mark.parametrize("num_ga", [4])
    def test_forward_backward(self, moe_router_topk, num_ga):
        # One-layer Qwen3-30B-A3B EP8 shapes: 128 experts, top-8,
        # hidden size 2048, and expert FFN hidden size 768.
        # Single MoE layer, num_ga gradient-accumulation microbatches: forward +
        # backward num_ga times without zeroing grads (mirrors real GA training).
        # One layer -> no cross-layer backward accumulation, so grads stay clean.
        num_moe_experts, hidden_size = 128, 2048
        self._init_process()
        self._setup_ep()
        try:
            config = _build_config(self.world_size, num_moe_experts, moe_router_topk)
            moe_layers, mega_layers = _build_layers(config, num_moe_experts, 1)
            moe_layer, mega_layer = moe_layers[0], mega_layers[0]
            assert moe_layer.shared_experts is None
            assert mega_layer.shared_experts is None
            moe_layer.train(True)
            mega_layer.train(True)
            experts = moe_layer.experts.local_experts
            seq, batch = 8192, 1

            fwd_cos, dx_cos = [], []
            # GA loop: accumulate grads over num_ga microbatches, compare each fwd/dx
            for step in range(num_ga):
                torch.manual_seed(1000 + self.rank + step * 97)
                x = torch.randn((seq, batch, hidden_size), dtype=torch.bfloat16, device=self.device)
                g = torch.randn((seq, batch, hidden_size), dtype=torch.bfloat16, device=self.device)

                x_ref = x.clone().requires_grad_(True)
                ref, _ = moe_layer(x_ref)
                ref.backward(g)
                x_meg = x.clone().requires_grad_(True)
                out, _ = mega_layer(x_meg)
                out.backward(g)

                fwd_cos.append(_cosine(out, ref))
                dx_cos.append(_cosine(x_meg.grad, x_ref.grad))
                self._rank0(
                    f"[ga {step}] fwd cos={fwd_cos[-1]:.6f} dx cos={dx_cos[-1]:.6f} "
                    f"max_abs_diff={(out.float()-ref.float()).abs().max().item():.3e}"
                )

            GRAD_FLOOR = 0.95
            failures = []

            def check_accuracy(tag, cos):
                # negated compare so a NaN cosine fails instead of slipping through
                if not cos > GRAD_FLOOR:
                    failures.append(f"{tag} cosine {cos:.6f} < {GRAD_FLOOR}")

            # forward + dx parity: every microbatch
            for step in range(num_ga):
                check_accuracy(f"ga{step} fwd", fwd_cos[step])
                check_accuracy(f"ga{step} dx", dx_cos[step])

            # accumulated per-weight grad parity: every trainable weight
            dgate = _cosine(mega_layer.router.weight.grad, moe_layer.router.weight.grad)
            check_accuracy("router", dgate)
            # routed experts: full w1/w2 grad tensors (norm-weighted over all local
            # experts); per-expert min is thin-token bf16 noise, informational only.
            ref_dw1 = torch.stack([e.linear_fc1.weight.grad for e in experts])
            ref_dw2 = torch.stack([e.linear_fc2.weight.grad for e in experts])
            mega_w1_grad = mega_layer.experts.fc1_weight.weight.grad
            mega_w2_grad = mega_layer.experts.fc2_weight.weight.grad
            dW1 = _cosine(mega_w1_grad, ref_dw1)
            dW2 = _cosine(mega_w2_grad, ref_dw2)
            check_accuracy("experts dW1", dW1)
            check_accuracy("experts dW2", dW2)
            dw1_min = min(_cosine(mega_w1_grad[i], e.linear_fc1.weight.grad) for i, e in enumerate(experts))
            dw2_min = min(_cosine(mega_w2_grad[i], e.linear_fc2.weight.grad) for i, e in enumerate(experts))
            self._rank0(
                f"[grad accum over {num_ga} ga] dgate={dgate:.6f} "
                f"experts dW1={dW1:.6f} dW2={dW2:.6f} "
                f"(per-expert min {dw1_min:.4f}/{dw2_min:.4f})"
            )
            assert failures == [], f"parity below {GRAD_FLOOR}: {failures}"
        finally:
            parallel_state.destroy_model_parallel()
