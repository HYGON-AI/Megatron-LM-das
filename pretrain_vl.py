# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import os
import torch

from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    unwrap_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    average_losses_across_data_parallel_group
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.datasets.sft_dataset import SFTDataset

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from dcu_megatron import megatron_adaptor

# VL depend
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from copy import deepcopy

from megatron.core.num_microbatches_calculator import get_num_microbatches

from dcu_megatron.core.datasets.VL_helpers import TaskEncoder,print_error_handler
from dcu_megatron.core.models.qwen_vl.vl_model import Qwen2_5VLModel
from dcu_megatron.core.models.qwen_vl.vision_config import Qwen2VLTransformerConfig, get_vision_model_config, get_vision_projection_config
from dcu_megatron.core.models.qwen_vl.vision_layer_specs import get_mlp_module_spec,get_qwen2vl_vision_layer_spec
# from dcu_megatron.core.models.qwen_vl.auguments import get_patch_args
from dcu_megatron.core.tensor_parallel.vl_data import broadcast_data
# from dcu_megatron.training.tokenizer import build_tokenizer, get_tokenizer
from megatron.training.tokenizer import build_tokenizer

stimer = StragglerDetector()


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
    pre_process=True, post_process=True, vp_stage=None, add_encoder=True, add_decoder=True
) -> Union[Qwen2_5VLModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args) # 构建全局tokenizer
    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    if bool(int(os.getenv("USE_FLUX_OVERLAP", "0"))):
        assert args.transformer_impl == "transformer_engine"
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("The Qwen2-VL model is only implemented with TransformerEngine!")

    print_rank_0('building VL model config ...')
    
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args, Qwen2VLTransformerConfig)

    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None   
    
    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.num_layers_in_first_pipeline_stage = None
    vision_projector_config = get_vision_projection_config(deepcopy(config), vision_config.hidden_size, vision_config.spatial_merge_size)
    
    print_rank_0('building VL model ...')

    if args.use_legacy_models:
        raise NotImplementedError("The VL model is only implemented with mcore models!")
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        
        vision_layer_spec = get_qwen2vl_vision_layer_spec()
        vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules
        # print(f"args.padded_vocab_size: {args.padded_vocab_size}")
        model = Qwen2_5VLModel(
            language_transformer_config=config,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=args.padded_vocab_size,
            language_max_sequence_length=args.max_position_embeddings,

            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False, # NOTE: no class token to drop?

            vision_projection_config=vision_projector_config,
            vision_projection_layer_spec=vision_projector_spec, 
            vision_projection_type='mlp',
            allow_missing_vision_projection_checkpoint= False, # TD: may parameterized

            language_position_embedding_type=args.position_embedding_type,
            language_rotary_percent=args.rotary_percent,
            language_rotary_base=args.rotary_base,
            
            pre_process=pre_process,
            post_process=post_process,
            add_decoder=add_decoder,
            add_encoder=add_encoder,

            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=None,
            vp_stage=vp_stage,
        )
        model.freeze(
            freeze_language_model=args.freeze_LM, 
            freeze_vision_model=args.freeze_ViT, 
            freeze_vision_projection=False
        )
        
    print_rank_0(model)
    return model

# Slightly modified from Qwen2_5VLForConditionalGeneration.get_rope_index
def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    args = get_args()
    tokenizer = get_tokenizer()
    spatial_merge_size = args.spatial_merge_size
    image_token_id = tokenizer.image_token_id
    video_token_id = tokenizer.video_token_id
    vision_start_token_id = tokenizer.vision_start_token_id
    tokens_per_second = 2
    if second_per_grid_ts is not None:
        second_per_grid_ts = second_per_grid_ts.cpu()

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

def get_ltor_masks_and_position_ids(
        input_ids, 
        image_thw_grids,
        video_thw_grids,
        target, 
        pad_token, 
        second_per_grid_ts,
        ignore_index=None
    ):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    position_ids, _ = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=input_ids != pad_token
    )

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    # Attention mask.
    attention_mask = None
   

    return attention_mask, loss_mask, position_ids

def get_batch(data_iterator):
    """Generate a batch"""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and parallel_state.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None
    data_text =  broadcast_data(["text"], data, torch.int64)["text"]
    target =  broadcast_data(["target"], data, torch.int64)["target"]
    # shape: num_tiles x c x h x w
    imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]

    # shape: num_tiles x c x h x w
    videos = broadcast_data(["videos"], data, torch.float32)["videos"]
    # shape: n_image_samples
    image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]
    # shape: n_video_samples
    video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
    # shape: n_video_samples
    second_per_grid_ts = broadcast_data(['second_per_grid_ts'], data, torch.float32)['second_per_grid_ts']


    image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
    video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    # NOTE: no sequence packing in LLM inputs
    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, image_thw_grids, video_thw_grids, labels, tokenizer.pad_token_id, second_per_grid_ts
    )
    torch.cuda.nvtx.range_pop()
    # if attention_mask is None:
    #     batch_size, seq_len = tokens.shape
    #     attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=tokens.device, dtype=torch.bool)
    return (
        tokens, 
        labels, 
        loss_mask, 
        attention_mask, 
        position_ids, 
        imgs, 
        videos,
        image_thw_grids, 
        video_thw_grids,
        image_input_mask, 
        video_input_mask
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    return loss[0] * args.context_parallel_size, {"lm loss": averaged_loss}

# def loss_func(
#     loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
# ):
#     """Loss function.

#     Args:
#         loss_mask (torch.Tensor): Used to mask out some portions of the loss
#         output_tensor (torch.Tensor): The tensor with the losses
#         model (GPTModel, optional): The model (can be wrapped)

#     Returns:
#         the loss scalar for this micro-batch
#         the number of non-padded tokens in this microbatch
#         a dict containing reporting metrics on the loss and number of tokens across
#             the data parallel ranks
#     """
#     args = get_args()

#     losses = output_tensor.view(-1).float()
#     loss_mask = loss_mask.view(-1).float()
#     loss = torch.sum(losses * loss_mask)

#     # Check individual rank losses are not NaN prior to DP all-reduce.
#     rerun_state_machine = get_rerun_state_machine()
#     if args.check_for_nan_in_loss_and_grad:
#         rerun_state_machine.validate_result(
#             result=loss,
#             rejection_func=torch.isnan,
#             message="found NaN in local forward loss calculation",
#             tolerance=0.0,  # forward pass calculations are determinisic
#             fatal=True,
#         )
#         rerun_state_machine.validate_result(
#             result=loss,
#             rejection_func=torch.isinf,
#             message="found Inf in local forward loss calculation",
#             tolerance=0.0,  # forward pass calculations are determinisic
#             fatal=True,
#         )
#     # Check for spiky loss
#     if args.check_for_spiky_loss:
#         rerun_state_machine.validate_result(
#             result=loss,
#             rejection_func=partial(
#                 rerun_state_machine.is_unexpectedly_large,
#                 threshold=SPIKY_LOSS_FACTOR,
#                 context="loss",
#             ),
#             message="Spiky loss",
#             tolerance=0.0,  # forward pass calculations are determinisic
#             fatal=False,
#         )

#     num_tokens = loss_mask.sum().clone().detach().to(torch.int)
#     reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

#     return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: Qwen2_5VLModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (tokens, labels, loss_mask, attention_mask, position_ids,
        imgs, videos, image_thw_grids, video_thw_grids, image_input_mask, video_input_mask)= get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        imgs = imgs.view(imgs.size(0), -1)   # [400, 1536]
        vision_data = torch.cat([imgs, videos], dim=0)
        vision_grid = torch.cat([image_thw_grids, video_thw_grids], dim=0)
        
        if args.use_legacy_models:
            raise NotImplementedError("The VL model is only implemented with mcore models!")
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask)
            else:
                output_tensor = model(
                    input_ids = tokens,
                    position_ids = position_ids,
                    vision_data = vision_data,
                    vision_grid_thw =  vision_grid,
                    video_start_index = image_input_mask.sum().cpu().item(),
                    image_input_mask = image_input_mask,
                    video_input_mask = video_input_mask,
                    attention_mask = attention_mask,
                    labels = labels
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


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

def datasets_provider(worker_config=None):
    """Create multimodal train, validation and test datasets."""
    args = get_args()
    train_dataset = get_train_dataset(
        args.train_data_path[0],
        batch_size=args.micro_batch_size,
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        virtual_epoch_length=1000,
        max_samples_per_sequence=100,
        shuffle_buffer_size=100,
        handler=print_error_handler,
        image_decode="pil",
    )

    val_datasets = get_val_datasets(
        args.valid_data_path[0],
        batch_size=args.micro_batch_size,
        # This is the total number over all workers
        # limit=args.eval_iters * get_num_microbatches(),
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        handler=print_error_handler,
        image_decode="pil",
    )

    val_datasets_without_source_datasets = [
        # Limit the dataset to eval_iters * num_microbatches
        LimitDataset(
            # Repeat the inner dataset in case it's too short
            RepeatDataset(val_ds, worker_config=worker_config),
            length=args.eval_iters * get_num_microbatches(),
            worker_config=worker_config,
            reset_after_epoch=True,
        )
        for val_ds, _src_ds in val_datasets
    ]

    return train_dataset, val_datasets_without_source_datasets, None

class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()
    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )

    print_rank_0("> building train, validation, and test datasets for VL model ...")

    train_ds, valid_ds, test_ds = datasets_provider(worker_config)

    print_rank_0("> finished creating VL datasets ...")

    train_dataloader = EnergonDataloader(get_savable_loader(train_ds, worker_config=worker_config))
    valid_dataloader = [
        EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
        for valid_ds in valid_ds
    ]

    print_rank_0("> finished creating VL dataloader ...")
    
    if args.load is not None:
        raise NotImplementedError("The Qwen VL does not support loading ckpt !")

    return train_dataloader, valid_dataloader, None

def write_online_eval_to_tensorboard(data, iteration, writer):
    """Write online evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)

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
        args_defaults={'tokenizer_type': 'Qwen2VLTokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
    )
