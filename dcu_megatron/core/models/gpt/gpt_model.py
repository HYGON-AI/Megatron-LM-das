import os

from collections import OrderedDict
from typing import Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.packed_seq_params import PackedSeqParams

from dcu_megatron.core.tensor_parallel import FluxColumnParallelLinear


def gpt_model_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # Output
        if self.post_process or self.mtp_process:
            if int(os.getenv("USE_FLUX_OVERLAP", "0")):
                parallel_linear_impl = FluxColumnParallelLinear
            else:
                parallel_linear_impl = tensor_parallel.ColumnParallelLinear
            self.output_layer = parallel_linear_impl(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    return wrapper


def gpt_model_forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    loss_mask: Optional[Tensor] = None,
) -> Tensor:
    """Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units

    Args:
        runtime_gather_output (bool): Gather output at runtime. Default None means
            `parallel_output` arg in the constructor will be used.
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

    # Decoder embedding.
    if decoder_input is not None:
        pass
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    rotary_pos_cos = None
    rotary_pos_sin = None
    if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
        if not self.training and self.config.flash_decode and inference_params:
            # Flash decoding uses precomputed cos and sin for RoPE
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                inference_params.max_sequence_length,
                self.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
            )
        else:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == 'thd',
            )
    if (
        (self.config.enable_cuda_graph or self.config.flash_decode)
        and rotary_pos_cos is not None
        and inference_params
    ):
        sequence_len_offset = torch.tensor(
            [inference_params.sequence_len_offset] * inference_params.current_batch_size,
            dtype=torch.int32,
            device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
        )
    else:
        sequence_len_offset = None

    # Run decoder.
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
    )

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if self.mtp_process:
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            output_layer=self.output_layer,
            output_weight=output_weight,
            runtime_gather_output=runtime_gather_output,
            compute_language_model_loss=self.compute_language_model_loss,
            **(extra_block_kwargs or {}),
        )

    if (
        self.mtp_process is not None
        and getattr(self.decoder, "main_final_layernorm", None) is not None
    ):
        # move block main model final norms here
        hidden_states = self.decoder.main_final_layernorm(hidden_states)

    if not self.post_process:
        return hidden_states

    logits, _ = self.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )

    if has_config_logger_enabled(self.config):
        payload = OrderedDict(
            {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            }
        )
        log_config_to_disk(self.config, payload, prefix='input_and_logits')

    if labels is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    loss = self.compute_language_model_loss(labels, logits)

    return loss
