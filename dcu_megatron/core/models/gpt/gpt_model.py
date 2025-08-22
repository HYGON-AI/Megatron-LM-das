import os

from collections import OrderedDict
from typing import Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import WrappedTensor, deprecate_inference_params


def gpt_model_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # Output
        if (
            (self.post_process or self.mtp_process)
            and int(os.getenv("USE_FLUX_OVERLAP", "0"))
        ):
            from dcu_megatron.core.tensor_parallel.layers import FluxColumnParallelLinear

            self.output_layer = FluxColumnParallelLinear(
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


def gpt_model_postprocess(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
):
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """
    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if mtp_in_postprocess:
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

    if (
        not self.training
        and inference_context is not None
        and inference_context.materialize_only_last_token_logits
    ):
        if inference_context.is_static_batching():
            hidden_states = hidden_states[-1:, :, :]
        else:
            # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
            # state ([B, H]) → unsqueeze back to [1, B, H]
            # (so that the output layer, which expects S×B×H, receives only the final token)
            hidden_states = inference_context.last_token_logits(
                hidden_states.squeeze(1).unsqueeze(0)
            ).unsqueeze(1)
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



class GPTModel:
    """
    patch megatron GPTModel
    """
    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,

    ):
        """Builds a computation schedule plan for the model.

        This function creates a schedule plan for a model chunk, including
        preprocessing, transformer layers, and postprocessing.
        The schedule plan is used to optimize computation and memory usage
        in distributed environments.

        Args:
            input_ids (Tensor): Input token IDs.
            position_ids (Tensor): Position IDs.
            attention_mask (Tensor): Attention mask.
            decoder_input (Tensor, optional): Decoder input tensor. Defaults to None.
            labels (Tensor, optional): Labels for loss computation. Defaults to None.
            inference_params (InferenceParams, optional):
                Parameters for inference. Defaults to None.
            packed_seq_params (PackedSeqParams, optional):
                Parameters for packed sequences. Defaults to None.
            extra_block_kwargs (dict, optional):
                Additional keyword arguments for blocks. Defaults to None.
            runtime_gather_output (Optional[bool], optional):
                Whether to gather output at runtime. Defaults to None.
            loss_mask (Optional[Tensor], optional): Loss mask. Defaults to None.

        Returns:
            ModelChunkSchedulePlan: The model chunk schedule plan.
        """
        from ..common.model_chunks_schedule_plan import TransformerModelChunkSchedulePlan

        return TransformerModelChunkSchedulePlan(
            self,
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params,
            extra_block_kwargs,
            runtime_gather_output,
            loss_mask,
        )

    def backward_dw(self):
        self.decoder.backward_dw()
