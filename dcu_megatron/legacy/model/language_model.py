from typing import Optional
from megatron.core.utils import deprecate_inference_params
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.legacy.model.module import MegatronModule



class TransformerLanguageModelPatch(MegatronModule):
    """Transformer language model.

    Args:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        enc_attn_mask,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        retriever_input_ids=None,
        retriever_position_ids=None,
        retriever_attn_mask=None,
        enc_dec_attn_mask=None,
        tokentype_ids=None,
        inference_context=None,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden=False,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        micro_sp_idx=None
    ):

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(
                enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids
            )
        else:
            encoder_input = None

        # Retriever embedding.
        if self.add_retriever and self.pre_process:
            retriever_input = self.embedding(
                retriever_input_ids, retriever_position_ids, tokentype_ids=tokentype_ids
            )
        else:
            retriever_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_context is not None:
                rotary_pos_emb = self.rotary_pos_emb(inference_context.max_sequence_length)
            else:
                rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    retriever_input=retriever_input,
                    retriever_attn_mask=retriever_attn_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    micro_sp_idx=micro_sp_idx
                )
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids, dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
        )

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output
