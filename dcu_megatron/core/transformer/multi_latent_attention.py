from megatron.core.transformer.multi_latent_attention import MLASelfAttention as MegatronCoreMLASelfAttention


class MLASelfAttention(MegatronCoreMLASelfAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def backward_dw(self):
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()
        self.linear_proj.backward_dw()
