# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import sys
from contextlib import nullcontext

import pytest
import torch

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    set_global_variables,
)
from tests.unit_tests.test_utilities import Utils

from hcu_megatron.core.pipeline_parallel.dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
from hcu_megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
    forward_backward_pipelining_zbh1,
)
from hcu_megatron.core.pipeline_parallel.seq1f1b.schedules import (
    seq1f1b_forward_backward_pipelining_without_interleaving,
    seq1f1b_forward_backward_pipelining_with_interleaving
)
from hcu_megatron.core.pipeline_parallel.ripipe_schedules import forward_backward_ripipe_pipelining
from hcu_megatron.megatron_adaptor import repatch
from hcu_megatron.training.arguments import destroy_adaptor_args, get_adaptor_args

rank = Utils.rank


def create_test_adaptor_args():
    sys.argv = ['test_schedules.py']
    destroy_adaptor_args()
    args = get_adaptor_args()
    return args


def create_test_args():
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_schedules.py']
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 8
    args.max_position_embeddings = 512
    args.micro_batch_size = 1
    args.create_attention_mask_in_dataloader = True
    args.seq_length = 256

    validate_args(args)
    set_global_variables(args, False)
    return args


@pytest.mark.parametrize(
    "schedule_method, forward_backward_func, should_assert_error",
    [("dualpipev", forward_backward_pipelining_with_cutinhalf, False),
     ("seq1f1b", seq1f1b_forward_backward_pipelining_without_interleaving, False),
     ("interleaved_seq1f1b", seq1f1b_forward_backward_pipelining_with_interleaving, False),
     ("ripipe", forward_backward_ripipe_pipelining, False),
     ("unsupport_schedule", forward_backward_pipelining_with_cutinhalf, True)]
)
def test_get_forward_backward_func(schedule_method, forward_backward_func, should_assert_error):
    adaptor_args = create_test_adaptor_args()
    adaptor_args.schedule_method = schedule_method
    megatron_args = create_test_args()
    megatron_args.schedule_method = schedule_method
    repatch(vars(adaptor_args), vars(megatron_args))

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
    )

    context = (
        pytest.raises((AssertionError, ValueError)) if should_assert_error else nullcontext()
    )
    with context:
        fb_func = schedule.get_forward_backward_func()

    if not should_assert_error:
        assert fb_func == forward_backward_func

    Utils.destroy_model_parallel()


def test_forward_backward_pipelining_with_cutinhalf(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    adaptor_args = create_test_adaptor_args()
    adaptor_args.schedule_method = "dualpipev"
    megatron_args = create_test_args()
    megatron_args.schedule_method = "dualpipev"
    repatch(vars(adaptor_args), vars(megatron_args))

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
    )

    def forward_step_func(data_iterator, model, microbatch_id=None):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        forward_backward_func
        == forward_backward_pipelining_with_cutinhalf
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
    )
    config.hidden_size = hidden_size
    model.config = config
    model.pre_process = False

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=8,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=4,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_pipelining_zbh1(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    adaptor_args = create_test_adaptor_args()
    adaptor_args.schedule_method = "zb_h1"
    megatron_args = create_test_args()
    megatron_args.schedule_method = "zb_h1"
    repatch(vars(adaptor_args), vars(megatron_args))

    from megatron.training import get_args
    args = get_args()
    print(f"{getattr(args, 'wandb_project', '')=}, {args.save=}", flush=True)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    def forward_step_func(data_iterator, model, microbatch_id=None):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(sequence_length, micro_batch_size, hidden_size).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == forward_backward_pipelining_zbh1

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
    )
    config.hidden_size = hidden_size
    model.config = config
    model.pre_process = False

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=model,
        num_microbatches=8,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_pipelining_without_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    adaptor_args = create_test_adaptor_args()
    adaptor_args.schedule_method = "vanilla"
    adaptor_args.delay_1f1b_cooldown_wgrad_compute = True
    megatron_args = create_test_args()
    megatron_args.schedule_method = "vanilla"
    adaptor_args.delay_1f1b_cooldown_wgrad_compute = True
    repatch(vars(adaptor_args), vars(megatron_args))

    from megatron.training import get_args
    args = get_args()
    print(f"{getattr(args, 'wandb_project', '')=}, {args.save=}", flush=True)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    def forward_step_func(data_iterator, model, microbatch_id=None):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(sequence_length, micro_batch_size, hidden_size).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == forward_backward_pipelining_without_interleaving

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
    )
    config.hidden_size = hidden_size
    model.config = config
    model.pre_process = False

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=[model],
        num_microbatches=4,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()
