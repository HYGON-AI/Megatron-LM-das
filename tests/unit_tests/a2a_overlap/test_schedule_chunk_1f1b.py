# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import gc
import sys

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    set_global_variables,
)

from tests.unit_tests.a2a_overlap.utils import (
    compare_captures,
    deterministic_mode,
    get_test_config,
)
from tests.unit_tests.test_utilities import Utils

from hcu_megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from hcu_megatron.megatron_adaptor import repatch
from hcu_megatron.training.arguments import destroy_adaptor_args, get_adaptor_args


def create_test_adaptor_args(
        overlap_ep_comm_with_split_attn=False,
        integrate_recompute_to_ep_comm_overlap=False,
        ep_overlap_early_recompute=False,
        schedule_method="vanilla",
    ):
    sys.argv = ['test_schedule_chunk_1f1b.py']
    args = get_adaptor_args()
    args.overlap_ep_comm_with_split_attn = overlap_ep_comm_with_split_attn
    args.integrate_recompute_to_ep_comm_overlap = integrate_recompute_to_ep_comm_overlap
    args.ep_overlap_early_recompute = ep_overlap_early_recompute
    args.schedule_method = schedule_method
    return args


def create_test_args(
        overlap_ep_comm_with_split_attn=False,
        integrate_recompute_to_ep_comm_overlap=False,
        ep_overlap_early_recompute=False,
        schedule_method="vanilla",
    ):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_schedule_chunk_1f1b.py']
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 64
    args.max_position_embeddings = 512
    args.micro_batch_size = 1
    args.create_attention_mask_in_dataloader = True
    args.seq_length = 32

    args.overlap_ep_comm_with_split_attn = overlap_ep_comm_with_split_attn
    args.integrate_recompute_to_ep_comm_overlap = integrate_recompute_to_ep_comm_overlap
    args.ep_overlap_early_recompute = ep_overlap_early_recompute
    args.schedule_method = schedule_method

    validate_args(args)
    set_global_variables(args, False)
    return args


def build_model(config, use_padding_mask=False, dualpipev_first_chunk=False):
    seq_len = 32
    max_seq_len = 300
    # ids = random.sample([i for i in range(max_seq_len)], seq_len)
    ids = [i for i in range(seq_len)]

    # build input tensors
    data = {
        "input_ids": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "labels": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "position_ids": torch.tensor([i for i in range(seq_len)], dtype=torch.int64)
        .repeat((1, 1))
        .cuda(),
        "attention_mask": torch.ones((1, 1, seq_len, seq_len), dtype=bool).cuda(),
    }

    # Optionally add padding_mask with same shape as input_ids
    if use_padding_mask:
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool).cuda()
        padding_mask[0, -8:] = True
        data["padding_mask"] = padding_mask

    # build layer spec
    args = get_args()
    if args.schedule_method == "dualpipev":
        args.dualpipev_first_chunk = dualpipev_first_chunk
    transformer_layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec.layer_specs[-1], True)

    # build model
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=128,
        pre_process=True,
        post_process=True,
        max_sequence_length=max_seq_len,
    )
    f_schedule_plan = gpt_model.build_schedule_plan(**data)
    return gpt_model, f_schedule_plan, data


def test_get_transformer_layer_schedule_plan(mocker):
    from hcu_megatron.core.models.common.model_chunk_schedule_plan import (
        get_transformer_layer_schedule_plan,
        TransformerLayerSchedulePlanWithoutSplitAttn,
        TransformerLayerSchedulePlanWithSplitAttn,
    )
    from hcu_megatron.core.models.common.model_chunk_schedule_plan_with_recompute \
        import TransformerLayerSchedulePlanWithoutSplitAttn as TransformerLayerSchedulePlanWithoutSplitAttnRecompute
    from hcu_megatron.core.models.common.model_chunk_schedule_plan_with_recompute \
        import TransformerLayerSchedulePlanWithSplitAttn as TransformerLayerSchedulePlanWithSplitAttnRecompute

    create_test_adaptor_args(
        overlap_ep_comm_with_split_attn=False,
        integrate_recompute_to_ep_comm_overlap=False,
    )
    assert get_transformer_layer_schedule_plan() == TransformerLayerSchedulePlanWithoutSplitAttn

    create_test_adaptor_args(
        overlap_ep_comm_with_split_attn=True,
        integrate_recompute_to_ep_comm_overlap=False,
    )
    assert get_transformer_layer_schedule_plan() == TransformerLayerSchedulePlanWithSplitAttn

    create_test_adaptor_args(
        overlap_ep_comm_with_split_attn=False,
        integrate_recompute_to_ep_comm_overlap=True,
    )
    assert get_transformer_layer_schedule_plan() == TransformerLayerSchedulePlanWithoutSplitAttnRecompute

    create_test_adaptor_args(
        overlap_ep_comm_with_split_attn=True,
        integrate_recompute_to_ep_comm_overlap=True,
    )
    assert get_transformer_layer_schedule_plan() == TransformerLayerSchedulePlanWithSplitAttnRecompute

    destroy_adaptor_args()


class TestA2AOverlap:
    """
    Test class for all-to-all overlap optimization in transformer models.

    This class contains tests to verify that the all-to-all overlap optimization
    produces the same results as the reference implementation.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        destroy_adaptor_args()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("layers,mtp_layers", [([2, 1], 0), ([1, 2], 1)])
    @pytest.mark.parametrize("overlap_ep_comm_with_split_attn", [True, False])
    def test_1f1b_schedule_model_chunk(
        self,
        mtp_layers,
        layers,
        overlap_ep_comm_with_split_attn,
    ):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """
        adaptor_args = create_test_adaptor_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
        )
        megatron_args = create_test_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
        )
        repatch(vars(adaptor_args), vars(megatron_args))

        microbatches = 1

        gpt_models = []
        schedule_plans = []
        ref_captures = []
        datas = []

        # create TransformerConfig
        extra_kwargs = {"moe_token_dispatcher_type": "alltoall"}
        if mtp_layers > 0:
            extra_kwargs["mtp_num_layers"] = mtp_layers
            extra_kwargs["mtp_loss_scaling_factor"] = 1.1
        with deterministic_mode():
            for layer_num in layers:
                output_tensors = []
                # build config
                config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
                # build model
                gpt_model, schedule_plan, data = build_model(config)
                gpt_model.cuda()
                gpt_models.append(gpt_model)
                datas.append(data)
                schedule_plans.append(schedule_plan)

                # run reference
                for _ in range(microbatches):
                    loss = gpt_model.forward(**data)
                    loss = float16_to_fp32(loss)
                    loss.backward(torch.ones_like(loss))
                    output_tensors.append(loss)

                capture = {"outputs": output_tensors}
                for name, param in gpt_model.named_parameters():
                    capture[name] = param.grad
                ref_captures.append(capture)
                gpt_model.zero_grad()

            assert gpt_models[0].embedding is not None
            assert gpt_models[1].embedding is not None
            # run a2a overlap
            capture_0 = {"outputs": []}
            capture_1 = {"outputs": []}
            a2a_captures = [capture_0, capture_1]
            for i in range(microbatches):
                # 1st forward
                if i > 0:
                    assert (
                        schedule_plans[0].pre_process is None
                    ), "pre_process should be released after backward"
                    schedule_plans[0] = gpt_models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = gpt_models[1].build_schedule_plan(**datas[1])
                f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
                capture_0["outputs"].append(f_input_0)
                # overlap
                f_input_1 = TransformerModelChunkSchedulePlan.run(
                    schedule_plans[1], schedule_plans[0], b_grad=torch.ones_like(f_input_0)
                )
                capture_1["outputs"].append(f_input_1)
                # last backward
                TransformerModelChunkSchedulePlan.run(
                    None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
                )
            for i in range(len(gpt_models)):
                for name, param in gpt_models[i].named_parameters():
                    a2a_captures[i][name] = param.grad

            # compare results
            for i in range(len(ref_captures)):
                comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
                assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

            # release resources is necessary, otherwise later testcases will oom
            for i in range(len(schedule_plans)):
                schedule_plans[i] = None
                ref_captures[i] = None
                a2a_captures[i] = None
                for k in datas[i]:
                    datas[i][k] = None
                datas[i] = None
                gpt_models[i].zero_grad()
                gpt_models[i] = None
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("layers", [[2, 1]])
    @pytest.mark.parametrize("tp_size", [1, 2])
    @pytest.mark.parametrize("overlap_ep_comm_with_split_attn", [True, False])
    def test_1f1b_schedule_model_chunk_with_padding_mask(
        self,
        layers,
        tp_size,
        overlap_ep_comm_with_split_attn,
    ):
        """
        Verifies all-to-all overlap optimization with padding_mask produces
        the same results as the reference implementation with various TP/EP/CP combinations.
        """
        adaptor_args = create_test_adaptor_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
        )
        megatron_args = create_test_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
        )
        repatch(vars(adaptor_args), vars(megatron_args))

        # Re-initialize model parallel with the specified configuration
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
        )
        set_streams()

        microbatches = 1

        gpt_models = []
        schedule_plans = []
        ref_captures = []
        datas = []

        # create TransformerConfig
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "tensor_model_parallel_size": tp_size,
            "sequence_parallel": tp_size > 1,
        }
        with deterministic_mode():
            for layer_num in layers:
                output_tensors = []
                # build config
                config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
                # build model with padding_mask
                gpt_model, schedule_plan, data = build_model(config, use_padding_mask=True)
                gpt_model.cuda()
                gpt_models.append(gpt_model)
                datas.append(data)
                schedule_plans.append(schedule_plan)

                # run reference
                for _ in range(microbatches):
                    loss = gpt_model.forward(**data)
                    loss = float16_to_fp32(loss)
                    loss.backward(torch.ones_like(loss))
                    output_tensors.append(loss)

                capture = {"outputs": output_tensors}
                for name, param in gpt_model.named_parameters():
                    capture[name] = param.grad
                ref_captures.append(capture)
                gpt_model.zero_grad()
            assert gpt_models[0].embedding is not None
            assert gpt_models[1].embedding is not None
            # run a2a overlap
            capture_0 = {"outputs": []}
            capture_1 = {"outputs": []}
            a2a_captures = [capture_0, capture_1]
            for i in range(microbatches):
                # 1st forward
                if i > 0:
                    assert (
                        schedule_plans[0].pre_process is None
                    ), "pre_process should be released after backward"
                    schedule_plans[0] = gpt_models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = gpt_models[1].build_schedule_plan(**datas[1])
                f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
                capture_0["outputs"].append(f_input_0)
                # overlap
                f_input_1 = TransformerModelChunkSchedulePlan.run(
                    schedule_plans[1], schedule_plans[0], b_grad=torch.ones_like(f_input_0)
                )
                capture_1["outputs"].append(f_input_1)
                # last backward
                TransformerModelChunkSchedulePlan.run(
                    None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
                )
            for i in range(len(gpt_models)):
                for name, param in gpt_models[i].named_parameters():
                    a2a_captures[i][name] = param.grad

            # compare results
            for i in range(len(ref_captures)):
                comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
                assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

            # release resources is necessary, otherwise later testcases will oom
            for i in range(len(schedule_plans)):
                schedule_plans[i] = None
                ref_captures[i] = None
                a2a_captures[i] = None
                for k in datas[i]:
                    datas[i][k] = None
                datas[i] = None
                gpt_models[i].zero_grad()
                gpt_models[i] = None
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("layers", [[2, 2]])
    @pytest.mark.parametrize("tp_size", [1])
    @pytest.mark.parametrize("overlap_ep_comm_with_split_attn", [True, False])
    @pytest.mark.parametrize("block_level_wgrad_compute", [True])
    def test_1f1b_schedule_model_chunk_dualpipev(
        self,
        layers,
        tp_size,
        overlap_ep_comm_with_split_attn,
        block_level_wgrad_compute,
    ):
        """
        Verifies all-to-all overlap optimization with padding_mask produces
        the same results as the reference implementation with various TP/EP/CP combinations.
        """
        adaptor_args = create_test_adaptor_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
            schedule_method="dualpipev",
        )
        megatron_args = create_test_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
            schedule_method="dualpipev",
        )
        repatch(vars(adaptor_args), vars(megatron_args))

        # Re-initialize model parallel with the specified configuration
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
        )
        set_streams()

        microbatches = 1

        gpt_models = []
        schedule_plans = []
        ref_captures = []
        datas = []

        # create TransformerConfig
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "tensor_model_parallel_size": tp_size,
            "sequence_parallel": tp_size > 1,
            "delay_wgrad_compute": True,
        }
        with deterministic_mode():
            for model_id, layer_num in enumerate(layers):
                output_tensors = []
                # build config
                config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
                # build model with padding_mask
                gpt_model, schedule_plan, data = build_model(
                    config,
                    use_padding_mask=True,
                    dualpipev_first_chunk=(model_id == 0)
                )
                gpt_model.cuda()
                gpt_models.append(gpt_model)
                datas.append(data)
                schedule_plans.append(schedule_plan)

                # run reference
                for _ in range(microbatches):
                    loss = gpt_model.forward(**data)
                    loss = float16_to_fp32(loss)
                    loss.backward(torch.ones_like(loss))
                    gpt_model.backward_dw()
                    output_tensors.append(loss)

                capture = {"outputs": output_tensors}
                for name, param in gpt_model.named_parameters():
                    capture[name] = param.grad
                ref_captures.append(capture)
                gpt_model.zero_grad()
            assert gpt_models[0].embedding is not None
            assert gpt_models[1].embedding is not None
            # run a2a overlap
            capture_0 = {"outputs": []}
            capture_1 = {"outputs": []}
            a2a_captures = [capture_0, capture_1]
            for i in range(microbatches):
                # 1st forward
                if i > 0:
                    assert (
                        schedule_plans[0].pre_process is None
                    ), "pre_process should be released after backward"
                    schedule_plans[0] = gpt_models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = gpt_models[1].build_schedule_plan(**datas[1])
                f_input_0, _ = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
                capture_0["outputs"].append(f_input_0)
                # overlap
                f_input_1, chunk_backward_dw = TransformerModelChunkSchedulePlan.run(
                    schedule_plans[1],
                    schedule_plans[0],
                    b_grad=torch.ones_like(f_input_0),
                    block_level_wgrad_compute=block_level_wgrad_compute,
                )
                capture_1["outputs"].append(f_input_1)
                if block_level_wgrad_compute:
                    chunk_backward_dw()
                # last backward
                _, chunk_backward_dw = TransformerModelChunkSchedulePlan.run(
                    None,
                    schedule_plans[1],
                    b_grad=torch.ones_like(f_input_1),
                    block_level_wgrad_compute=block_level_wgrad_compute,

                )
                if block_level_wgrad_compute:
                    chunk_backward_dw()
            for i in range(len(gpt_models)):
                for name, param in gpt_models[i].named_parameters():
                    a2a_captures[i][name] = param.grad

            # compare results
            for i in range(len(ref_captures)):
                comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
                assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

            # release resources is necessary, otherwise later testcases will oom
            for i in range(len(schedule_plans)):
                schedule_plans[i] = None
                ref_captures[i] = None
                a2a_captures[i] = None
                for k in datas[i]:
                    datas[i][k] = None
                datas[i] = None
                gpt_models[i].zero_grad()
                gpt_models[i] = None
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("layers,mtp_layers", [([2, 1], 0), ([1, 2], 1)])
    @pytest.mark.parametrize("overlap_ep_comm_with_split_attn", [True, False])
    @pytest.mark.parametrize("ep_overlap_early_recompute", [True, False])
    def test_1f1b_schedule_model_chunk_recompute(
        self,
        mtp_layers,
        layers,
        overlap_ep_comm_with_split_attn,
        ep_overlap_early_recompute,
    ):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """
        adaptor_args = create_test_adaptor_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
            integrate_recompute_to_ep_comm_overlap=True,
            ep_overlap_early_recompute=ep_overlap_early_recompute,
        )
        megatron_args = create_test_args(
            overlap_ep_comm_with_split_attn=overlap_ep_comm_with_split_attn,
            integrate_recompute_to_ep_comm_overlap=True,
            ep_overlap_early_recompute=ep_overlap_early_recompute,
        )
        repatch(vars(adaptor_args), vars(megatron_args))

        microbatches = 1

        gpt_models = []
        schedule_plans = []
        ref_captures = []
        datas = []

        # create TransformerConfig
        extra_kwargs = {"moe_token_dispatcher_type": "alltoall"}
        if mtp_layers > 0:
            extra_kwargs["mtp_num_layers"] = mtp_layers
            extra_kwargs["mtp_loss_scaling_factor"] = 1.1
        with deterministic_mode():
            for layer_num in layers:
                output_tensors = []
                # build config
                config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
                # build model
                gpt_model, schedule_plan, data = build_model(config)
                gpt_model.cuda()
                gpt_models.append(gpt_model)
                datas.append(data)
                schedule_plans.append(schedule_plan)

                # run reference
                for _ in range(microbatches):
                    loss = gpt_model.forward(**data)
                    loss = float16_to_fp32(loss)
                    loss.backward(torch.ones_like(loss))
                    output_tensors.append(loss)

                capture = {"outputs": output_tensors}
                for name, param in gpt_model.named_parameters():
                    capture[name] = param.grad
                ref_captures.append(capture)
                gpt_model.zero_grad()

            assert gpt_models[0].embedding is not None
            assert gpt_models[1].embedding is not None
            # run a2a overlap
            capture_0 = {"outputs": []}
            capture_1 = {"outputs": []}
            a2a_captures = [capture_0, capture_1]
            for i in range(microbatches):
                # 1st forward
                if i > 0:
                    assert (
                        schedule_plans[0].pre_process is None
                    ), "pre_process should be released after backward"
                    schedule_plans[0] = gpt_models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = gpt_models[1].build_schedule_plan(**datas[1])
                f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
                capture_0["outputs"].append(f_input_0)
                # overlap
                f_input_1 = TransformerModelChunkSchedulePlan.run(
                    schedule_plans[1], schedule_plans[0], b_grad=torch.ones_like(f_input_0)
                )
                capture_1["outputs"].append(f_input_1)
                # last backward
                TransformerModelChunkSchedulePlan.run(
                    None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
                )
            for i in range(len(gpt_models)):
                for name, param in gpt_models[i].named_parameters():
                    a2a_captures[i][name] = param.grad

            # compare results
            for i in range(len(ref_captures)):
                comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
                assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

            # release resources is necessary, otherwise later testcases will oom
            for i in range(len(schedule_plans)):
                schedule_plans[i] = None
                ref_captures[i] = None
                a2a_captures[i] = None
                for k in datas[i]:
                    datas[i][k] = None
                datas[i] = None
                gpt_models[i].zero_grad()
                gpt_models[i] = None
            gc.collect()
            torch.cuda.empty_cache()
