class SelfAttention():
    def backward_dw(self):
        self.linear_qkv.backward_dw()
        self.linear_proj.backward_dw()
