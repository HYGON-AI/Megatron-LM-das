import logging
from typing import Literal, Optional
from functools import wraps
from collections import OrderedDict

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock

from dcu_megatron.core.utils import tensor_slide
from dcu_megatron.core.transformer.mtp.multi_token_predictor import MultiTokenPredictor
from dcu_megatron.core.transformer.transformer_config import TransformerConfig


def gpt_model_init(
    self,
    config: TransformerConfig,
    transformer_layer_spec: ModuleSpec,
    vocab_size: int,
    max_sequence_length: int,
    pre_process: bool = True,
    post_process: bool = True,
    fp16_lm_cross_entropy: bool = False,
    parallel_output: bool = True,
    share_embeddings_and_output_weights: bool = False,
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
    rotary_percent: float = 1.0,
    rotary_base: int = 10000,
    rope_scaling: bool = False,
    scatter_embedding_sequence_parallel: bool = True,
    seq_len_interpolation_factor: Optional[float] = None,
    mtp_spec: ModuleSpec = None
) -> None:
    super(GPTModel, self).__init__(config=config)

    if has_config_logger_enabled(config):
        log_config_to_disk(config, locals(), prefix=type(self).__name__)

    self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
    self.vocab_size = vocab_size
    self.max_sequence_length = max_sequence_length
    self.pre_process = pre_process
    self.post_process = post_process
    self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
    self.parallel_output = parallel_output
    self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
    self.position_embedding_type = position_embedding_type

    # megatron core pipelining currently depends on model type
    # TODO: remove this dependency ?
    self.model_type = ModelType.encoder_or_decoder

    # These 4 attributes are needed for TensorRT-LLM export.
    self.max_position_embeddings = max_sequence_length
    self.rotary_percent = rotary_percent
    self.rotary_base = rotary_base
    self.rotary_scaling = rope_scaling

    if self.pre_process:
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=position_embedding_type,
            scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
        )

    if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            use_cpu_initialization=self.config.use_cpu_initialization,
        )

    # Transformer.
    self.decoder = TransformerBlock(
        config=self.config,
        spec=transformer_layer_spec,
        pre_process=self.pre_process,
        post_process=self.post_process
    )

    # Output
    if post_process:
        if self.config.defer_embedding_wgrad_compute:
            # The embedding activation buffer preserves a reference to the input activations
            # of the final embedding projection layer GEMM. It will hold the activations for
            # all the micro-batches of a global batch for the last pipeline stage. Once we are
            # done with all the back props for all the microbatches for the last pipeline stage,
            # it will be in the pipeline flush stage. During this pipeline flush we use the
            # input activations stored in embedding activation buffer and gradient outputs
            # stored in gradient buffer to calculate the weight gradients for the embedding
            # final linear layer.
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=self.pre_process
            and self.share_embeddings_and_output_weights,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )

    # add mtp
    self.mtp_spec: ModuleSpec = mtp_spec
    self.num_nextn_predict_layers = self.config.num_nextn_predict_layers
    self.share_mtp_embedding_and_output_weight = self.config.share_mtp_embedding_and_output_weight
    self.recompute_mtp_norm = self.config.recompute_mtp_norm
    self.recompute_mtp_layer = self.config.recompute_mtp_layer
    self.mtp_loss_scale = self.config.mtp_loss_scale
    if self.post_process and self.training and self.num_nextn_predict_layers:
        self.mtp_layers = torch.nn.ModuleList(
            [
                MultiTokenPredictor(
                    config,
                    self.mtp_spec.submodules,
                    vocab_size=self.vocab_size,
                    max_sequence_length=self.max_sequence_length,
                    layer_number=i,
                    pre_process=self.pre_process,
                    fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                    parallel_output=self.parallel_output,
                    position_embedding_type=self.position_embedding_type,
                    rotary_percent=self.rotary_percent,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    share_mtp_embedding_and_output_weight=self.share_mtp_embedding_and_output_weight,
                    recompute_mtp_norm=self.recompute_mtp_norm,
                    recompute_mtp_layer=self.recompute_mtp_layer,
                    add_output_layer_bias=False
                )
                for i in range(self.num_nextn_predict_layers)
            ]
        )

    if self.pre_process or self.post_process:
        self.setup_embeddings_and_output_layer()

    if has_config_logger_enabled(self.config):
        log_config_to_disk(
            self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
        )

    if self.num_nextn_predict_layers and (self.pre_process or self.post_process):
        setup_mtp_embeddings(self)


def shared_embedding_or_mtp_embedding_weight(self) -> Tensor:
    """Gets the embedding weight when share embedding and mtp embedding weights set to True.

    Returns:
        Tensor: During pre processing it returns the input embeddings weight while during post processing it returns
         mtp embedding layers weight
    """
    assert self.num_nextn_predict_layers > 0
    if self.pre_process:
        return self.embedding.word_embeddings.weight
    elif self.post_process:
        return self.mtp_layers[0].embedding.word_embeddings.weight
    return None


def setup_mtp_embeddings(self):
    """
    Share embedding layer in mtp layer.
    """
    if self.pre_process:
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
    # Set `is_embedding_or_output_parameter` attribute.
    for i in range(self.num_nextn_predict_layers):
        if self.post_process and self.mtp_layers[i].embedding.word_embeddings.weight is not None:
            self.mtp_layers[i].embedding.word_embeddings.weight.is_embedding_or_output_parameter = True

    if not self.share_mtp_embedding_and_output_weight:
        return

    if self.pre_process and self.post_process:
        # Zero out wgrad if sharing embeddings between two layers on same
        # pipeline stage to make sure grad accumulation into main_grad is
        # correct and does not include garbage values (e.g., from torch.empty).
        self.shared_embedding_or_mtp_embedding_weight().zero_out_wgrad = True
        return

    if self.pre_process and not self.post_process:
        assert parallel_state.is_pipeline_first_stage()
        self.shared_embedding_or_mtp_embedding_weight().shared_embedding = True

    if self.post_process and not self.pre_process:
        assert not parallel_state.is_pipeline_first_stage()
        for i in range(self.num_nextn_predict_layers):
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.mtp_layers[i].embedding.word_embeddings.weight.data.fill_(0)
            self.mtp_layers[i].embedding.word_embeddings.weight.shared = True
            self.mtp_layers[i].embedding.word_embeddings.weight.shared_embedding = True

    # Parameters are shared between the word embeddings layers, and the
    # heads at the end of the model. In a pipelined setup with more than
    # one stage, the initial embedding layer and the head are on different
    # workers, so we do the following:
    # 1. Create a second copy of word_embeddings on the last stage, with
    #    initial parameters of 0.0.
    # 2. Do an all-reduce between the first and last stage to ensure that
    #    the two copies of word_embeddings start off with the same
    #    parameter values.
    # 3. In the training loop, before an all-reduce between the grads of
    #    the two word_embeddings layers to ensure that every applied weight
    #    update is the same on both stages.

    # Ensure that first and last stages have the same initial parameter
    # values.
    if torch.distributed.is_initialized():
        if parallel_state.is_rank_in_embedding_group():
            weight = self.shared_embedding_or_mtp_embedding_weight()
            weight.data = weight.data.cuda()
            torch.distributed.all_reduce(
                weight.data, group=parallel_state.get_embedding_group()
            )

    elif not getattr(LanguageModule, "embedding_warning_printed", False):
        logging.getLogger(__name__).warning(
            "Distributed processes aren't initialized, so the output layer "
            "is not initialized with weights from the word embeddings. "
            "If you are just manipulating a model this is fine, but "
            "this needs to be handled manually. If you are training "
            "something is definitely wrong."
        )
        LanguageModule.embedding_warning_printed = True


def slice_inputs(self, input_ids, labels, position_ids, attention_mask):
    if self.num_nextn_predict_layers == 0:
        return (
            [input_ids],
            [labels],
            [position_ids],
            [attention_mask],
        )

    return (
        tensor_slide(input_ids, self.num_nextn_predict_layers),
        tensor_slide(labels, self.num_nextn_predict_layers),
        generate_nextn_position_ids(position_ids, self.num_nextn_predict_layers),
        # not compatible with ppo attn_mask
        tensor_slide(attention_mask, self.num_nextn_predict_layers, dims=[-2, -1]),
    )


def generate_nextn_position_ids(tensor, slice_num):
    slides = tensor_slide(tensor, slice_num)
    if slides[0] is None:
        return slides

    for idx in range(1, len(slides)):
        slides[idx] = regenerate_position_ids(slides[idx], idx)
    return slides


def regenerate_position_ids(tensor, offset):
    if tensor is None:
        return None

    tensor = tensor.clone()
    for i in range(tensor.size(0)):
        row = tensor[i]
        zero_mask = (row == 0)        # 两句拼接情形
        if zero_mask.any():
            first_zero_idx = torch.argmax(zero_mask.int()).item()
            tensor[i, :first_zero_idx] = torch.arange(first_zero_idx)
        else:
            tensor[i] = tensor[i] - offset
    return tensor


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
) -> Tensor:
    """Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoeder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units

    Args:
        runtime_gather_output (bool): Gather output at runtime. Default None means
            `parallel_output` arg in the constructor will be used.
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

    # generate inputs for main and mtps
    input_ids, labels, position_ids, attention_mask = slice_inputs(
        self,
        input_ids,
        labels,
        position_ids,
        attention_mask
    )

    # Decoder embedding.
    if decoder_input is not None:
        pass
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids[0], position_ids=position_ids[0])
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
        attention_mask=attention_mask[0],
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
    )

    if not self.post_process:
        return hidden_states

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    loss = 0
    # Multi token prediction module
    if self.num_nextn_predict_layers and self.training:
        if not self.share_embeddings_and_output_weights and self.share_mtp_embedding_and_output_weight:
            output_weight = self.output_layer.weight
            output_weight.zero_out_wgrad = True

        embedding_weight = self.shared_embedding_or_mtp_embedding_weight() if self.share_mtp_embedding_and_output_weight else None
        mtp_hidden_states = hidden_states
        for i in range(self.num_nextn_predict_layers):
            mtp_hidden_states, mtp_loss = self.mtp_layers[i](
                mtp_hidden_states,  # [s,b,h]
                input_ids[i + 1],
                position_ids[i + 1] if position_ids[0] is not None else None,
                attention_mask[i + 1] if attention_mask[0] is not None else None,
                labels[i + 1] if labels[0] is not None else None,
                inference_params,
                packed_seq_params,
                extra_block_kwargs,
                embeding_weight=embedding_weight,
                output_weight=output_weight,
            )

            loss += self.mtp_loss_scale / self.num_nextn_predict_layers * mtp_loss
   
    if (
        self.num_nextn_predict_layers
        and getattr(self.decoder, "final_layernorm", None) is not None
    ):
        # move block main model final norms here
        hidden_states = self.decoder.final_layernorm(hidden_states)

    logits, _ = self.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
    )

    if has_config_logger_enabled(self.config):
        payload = OrderedDict(
            {
                'input_ids': input_ids[0],
                'position_ids': position_ids[0],
                'attention_mask': attention_mask[0],
                'decoder_input': decoder_input,
                'logits': logits,
            }
        )
        log_config_to_disk(self.config, payload, prefix='input_and_logits')

    if labels[0] is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    loss += self.compute_language_model_loss(labels[0], logits)

    return loss
