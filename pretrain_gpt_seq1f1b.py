# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Pretrain GPT"""

import os
import torch
import torch.distributed
import copy
from typing import List, Optional, Tuple, Union
from functools import partial
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.training import get_args, print_rank_0, get_timers, get_tokenizer
from megatron.training import inprocess_restart
from hcu_megatron.core.pipeline_parallel.seq1f1b.sp_utils import get_splits
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.training.utils import (
    unwrap_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
import megatron.legacy.model  # isort: skip
from megatron.training.datasets.sft_dataset import SFTDataset
# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import
try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
def get_batch_sp():
    offset = -1
    global_data = None
    count = 0

    def get_data(*args, **kwargs):
        pipe_sp = get_args().pipe_sp_splits
        nonlocal global_data, offset
        nonlocal count
        if offset == -1 or offset + 1 == pipe_sp:
            global_data = get_batch(*args, **kwargs)
            # torch.save(global_data, f"./cache/data/global_data_{count}.pt")
            count += 1

        offset = (offset + 1) % pipe_sp
        tokens, labels, loss_mask, attention_mask, position_ids = global_data
        if tokens != None:
            seq_length = tokens.size(1)
            global_args = get_args()
            global_args.pipe_sp_strategy = "average" if global_args.pipe_sp_splits == 1 else global_args.pipe_sp_strategy
            if global_args.pipe_sp_strategy == "uniform_comp":
                l_s = 0
                for idx, split in enumerate(get_splits()):
                    _tokens = tokens[:, l_s:l_s + split]
                    _labels = labels[:, l_s:l_s + split]
                    # _loss_mask = loss_mask[:, l_s:l_s+split]
                    _loss_mask = loss_mask
                    _loss_mask._start = l_s
                    _loss_mask._end = l_s + split
                    _position_ids = position_ids[:, l_s:l_s + split]
                    local_data = (_tokens, _labels, _loss_mask, attention_mask, _position_ids, offset)
                    l_s += split
                    if idx == offset:
                        break

            elif global_args.pipe_sp_strategy == "average":
                tokens = tokens.chunk(pipe_sp, dim=1)[offset]
                labels = labels.chunk(pipe_sp, dim=1)[offset]
                loss_mask._start = seq_length // pipe_sp * offset
                loss_mask._end = seq_length // pipe_sp * (offset + 1)
                # loss_mask = loss_mask.chunk(pipe_sp, dim=1)[offset]
                position_ids = position_ids.chunk(pipe_sp, dim=1)[offset]
                local_data = (tokens, labels, loss_mask, attention_mask, position_ids, offset)

            return local_data
        else:
            return tokens, labels, loss_mask, attention_mask, position_ids, offset
    return get_data

get_batch_sp_func = get_batch_sp()




def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def model_provider(
        pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    if bool(int(os.getenv("USE_FLUX_OVERLAP", "0"))):
        assert args.transformer_impl == "transformer_engine"
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
            )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )
        print_rank_0(model)

    return model


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    start = loss_mask._start
    end = loss_mask._end
    loss_mask_p = loss_mask[:, start:end]
    loss_mask = loss_mask.contiguous()
    loss_mask = loss_mask.view(-1).float()
    args = get_args()
    if get_args().pipe_sp_splits > 1:
        loss_mask_p = loss_mask_p.contiguous().view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask_p) / loss_mask.sum() * args.pipe_sp_splits
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
                result=loss,
                rejection_func=torch.isnan,
                message="found NaN in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        rerun_state_machine.validate_result(
                result=loss,
                rejection_func=torch.isinf,
                message="found Inf in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
                result=loss,
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context="loss",
                ),
                message="Spiky loss",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    tokens, labels, loss_mask, attention_mask, position_ids, offset = get_batch_sp_func(
        data_iterator, vp_stage)
    timers('batch-generator').stop()
    global stimer
    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, micro_sp_idx=offset)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, micro_sp_idx=offset
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, micro_sp_idx=offset
                )

    return output_tensor, partial(loss_func, loss_mask)

def is_dataset_built_on_rank(vp_stage=None):
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    args = get_args()
    tokenizer = get_tokenizer()
    datatype = torch.int64
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # # get batches based on the TP rank you are on
    # batch = get_batch_on_this_tp_rank(data_iterator)
    #
    # # slice batch along sequence dimension for context parallelism
    # batch = get_batch_on_this_cp_rank(batch)

    # Get the masks and postition ids.

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    tokens = tensor_parallel.broadcast_data(['tokens'], data, datatype)
    labels = tensor_parallel.broadcast_data(['labels'], data, datatype)

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens['tokens'],
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens['tokens'],labels['labels'],loss_mask,attention_mask, position_ids


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    # long sequence will causal OOM here since mask is N^2 memory
    # attention_mask = torch.tril(torch.ones(
    #     (att_mask_batch, seq_length, seq_length), device=data.device)).view(
    #         att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    # attention_mask = (attention_mask < 0.5)
    attention_mask = None

    return attention_mask, loss_mask, position_ids

def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
