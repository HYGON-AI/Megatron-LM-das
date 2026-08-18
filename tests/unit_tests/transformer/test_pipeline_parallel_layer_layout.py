import sys
from contextlib import nullcontext

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_global_variables,
)
from tests.unit_tests.test_utilities import Utils

from hcu_megatron.megatron_adaptor import repatch
from hcu_megatron.training.arguments import get_adaptor_args


def create_test_adaptor_args():
    sys.argv = ['test_pipeline_parallel_layer_layout.py']
    args = get_adaptor_args()
    args.schedule_method = "dualpipev"
    return args

def create_test_args():
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_pipeline_parallel_layer_layout.py']
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 8
    args.max_position_embeddings = 512
    args.micro_batch_size = 1
    args.create_attention_mask_in_dataloader = True
    args.seq_length = 256
    args.schedule_method = "dualpipev"
    args.delay_wgrad_compute = True

    validate_args(args)
    set_global_variables(args, False)
    return args


class TestPipelineParallelLayoutTransformerBlock:
    @classmethod
    def setup_class(cls):
        adaptor_args = create_test_adaptor_args()
        megatron_args = create_test_args()
        repatch(vars(adaptor_args), vars(megatron_args))

    @pytest.mark.parametrize(
        "num_layers, pp_size, pipeline_model_parallel_layout, should_assert_error",
        [
            # No embedding layer provided
            (7, 2, [["decoder"] * 6, ["decoder", "loss"]], True),
            # No loss layer provided
            (7, 2, [["embedding"] + ["decoder"] * 6, ["decoder"]], True),
            # Invalid layer type
            (7, 2, [["embedding"], ["invalid_type"] * 7 + ["loss"]], True),
            # Invalid pp size
            (7, 2, [["embedding"], ["decoder"] * 7, ["loss"]], True),
            # Invalid layout
            (
                7,
                2,
                [[["embedding", "decoder"], ["decoder"] * 4], ["decoder"], ["decoder", "loss"]],
                True,
            ),
            # Invalid layout
            (
                7,
                2,
                [[["embedding", "decoder"], ["decoder"] * 4], ["decoder"] * 2 + ["loss"]],
                True,
            ),
            # Invalid layout
            (7, 2, [[["embedding"] + ["decoder"] * 5], ["decoder"] * 2 + ["loss"]], True),
            # Usual pp case
            (
                7,
                2,
                [
                    [["embedding", "decoder"], ["decoder"] * 3],
                    [["decoder"] * 2, ["decoder", "loss"]],
                ],
                True,
            ),
            (
                62,
                8,
                [["embedding"] + ["decoder"] * 3] + [["decoder"] * 2] * 29 + [["decoder"], ["loss"]],
                True,
            ),
            # Usual pp case
            (
                7,
                2,
                [["embedding", "decoder"], ["decoder"] * 4, ["decoder"], ["decoder", "loss"]],
                False,
            ),
            # Empty stage
            (7, 2, [["embedding"], ["decoder"] * 7, [], ["loss"]], False),
            # Usual uneven vpp case with standalone embedding and loss layer
            (7, 2, [["embedding"], ["decoder"] * 6, ["decoder"], ["loss"]], False),
        ],
    )
    def test_layer_builder(
        self, num_layers, pp_size, pipeline_model_parallel_layout, should_assert_error
    ):
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
        )
        context = (
            pytest.raises((AssertionError, ValueError)) if should_assert_error else nullcontext()
        )
        with context:
            transformer_config = TransformerConfig(
                num_layers=num_layers,
                pipeline_model_parallel_layout=pipeline_model_parallel_layout,
                pipeline_model_parallel_size=pp_size,
                pipeline_dtype=torch.bfloat16,
                hidden_size=128,
                num_attention_heads=16,
            )
            total_build_layers = 0
            for i in range(pp_size):
                parallel_state.set_pipeline_model_parallel_rank(i)
                for j in range(2):
                    num_layers_test = get_num_layers_to_build(transformer_config, vp_stage=j)
                    total_build_layers += num_layers_test
        if not should_assert_error:
            assert (
                total_build_layers == num_layers
            ), f"total build layers {total_build_layers} should be equal to num_layers {num_layers}"
        parallel_state.set_pipeline_model_parallel_world_size(None)

    @pytest.mark.parametrize(
        ('pipeline_model_parallel_layout', 'layer_number_golden_answer'),
        [
            (
                [
                    ["embedding"],
                    ["decoder"],
                    ["decoder"] * 2,
                    ["decoder"],
                    [],
                    ["decoder"],
                    ["decoder"],
                    ["decoder"] * 2 + ["loss"],
                ],
                [[[], [7, 8]], [[1], [6]], [[2, 3], [5]], [[4], []]],
            )
        ],
    )
    def test_layout_layer_number(self, pipeline_model_parallel_layout, layer_number_golden_answer):
        tp_size = 1
        pp_size = 4
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)

        # Initialize GPT model
        default_config_kwargs = dict(
            num_layers=8,
            hidden_size=8,
            num_attention_heads=8,
            use_cpu_initialization=True,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            pipeline_model_parallel_layout=pipeline_model_parallel_layout,
        )
        transformer_config = TransformerConfig(**default_config_kwargs)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        gpt_model = []
        args = get_args()
        args.dualpipev_first_chunk = True
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=4,
            pre_process=(pp_rank == 0),
            post_process=False,
            vp_stage=0,
        )
        this_model.model_type = ModelType.encoder_or_decoder
        gpt_model.append(this_model)

        args.dualpipev_first_chunk = False
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=4,
            pre_process=False,
            post_process=(pp_rank == 0),
            vp_stage=1,
        )
        this_model.model_type = ModelType.encoder_or_decoder
        gpt_model.append(this_model)

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        for vpp_rank in range(2):
            layers = gpt_model[vpp_rank].decoder.layers
            layer_numbers = [l.layer_number for l in layers]
            golden_answer_curr_stage = layer_number_golden_answer[pp_rank][vpp_rank]
            assert len(layers) == len(
                golden_answer_curr_stage
            ), f"{pp_rank=}, {vpp_rank=}, {len(layers)=}, {len(golden_answer_curr_stage)=}"
            assert (
                layer_numbers == golden_answer_curr_stage
            ), f"{pp_rank=}, {vpp_rank=}, {layer_numbers=}, {golden_answer_curr_stage=}"
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "pp_size, input_layout_str, input_layout_list",
        [
            (
                2,
                "Et|t*4|t|tL",
                [["embedding", "decoder"], ["decoder"] * 4, ["decoder"], ["decoder", "loss"]],
            ),
            (2, "E|t*6|t|L", [["embedding"], ["decoder"] * 6, ["decoder"], ["loss"]]),
            (
                4,
                "E|t|t*2|t||(t|)*2,t*2,L",
                [
                    ["embedding"],
                    ["decoder"],
                    ["decoder"] * 2,
                    ["decoder"],
                    [],
                    ["decoder"],
                    ["decoder"],
                    ["decoder"] * 2 + ["loss"],
                ],
            ),
            (
                8,
                "Et*3|(tt|)*13,m|L",
                [["embedding"] + ["decoder"] * 3] + [["decoder"] * 2] * 13 + [["mtp"], ["loss"]],
            ),
            (
                16,
                "Et*2|(tt|)*29,t|mL",
                [["embedding"] + ["decoder"] * 2]
                + [["decoder"] * 2] * 29
                + [["decoder"]]
                + [["mtp", "loss"]],
            ),
        ],
    )
    def test_parsing_layout_from_str(self, pp_size, input_layout_str, input_layout_list):
        parsed_layout_from_str = PipelineParallelLayerLayout.from_str(input_layout_str, pp_size)
        parsed_layout_baseline = PipelineParallelLayerLayout(input_layout_list, pp_size)
        assert parsed_layout_from_str.layout == parsed_layout_baseline.layout
        assert (
            parsed_layout_from_str.virtual_pipeline_model_parallel_size
            == parsed_layout_baseline.virtual_pipeline_model_parallel_size
        )

    @pytest.mark.parametrize(
        "pp_size, input_layout",
        [
            (2, "Et|t*4|t|tL"),
            (2, [["embedding", "decoder"], ["decoder"] * 4, ["decoder"], ["decoder", "loss"]]),
            (8, [["embedding"] + ["decoder"] * 3] + [["decoder"] * 2] * 13 + [["mtp"], ["loss"]]),
        ],
    )
    def test_repr_returns_string(self, pp_size, input_layout):
        """Test that __repr__ always returns a string for both str and list inputs."""
        layout = PipelineParallelLayerLayout(input_layout, pp_size)
        repr_result = repr(layout)

        # Assert that repr returns a string
        assert isinstance(
            repr_result, str
        ), f"__repr__ must return a string, but got {type(repr_result).__name__}"

        # Assert that the returned string matches the expected value
        if isinstance(input_layout, str):
            # For string input, repr should return the exact same string
            assert repr_result == input_layout, (
                f"For string input, repr should return the original string.\n"
                f"Expected: {input_layout!r}\n"
                f"Got: {repr_result!r}"
            )
        else:
            # For list input, repr should return str(input_layout)
            expected_repr = str(input_layout)
            assert repr_result == expected_repr, (
                f"For list input, repr should return str(input_layout).\n"
                f"Expected: {expected_repr!r}\n"
                f"Got: {repr_result!r}"
            )
