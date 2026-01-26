from megatron.core.utils import deprecate_inference_params
from megatron.legacy.model.module import MegatronModule
from megatron.legacy.model.gpt_model import post_language_model_processing


class GPTModelPatch(MegatronModule):
    """GPT-2 Language model."""

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_context=None, *, inference_params=None, micro_sp_idx=None):

        inference_context = deprecate_inference_params(inference_context, inference_params)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_context=inference_context, micro_sp_idx=micro_sp_idx)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output
