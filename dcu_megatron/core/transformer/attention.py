class SelfAttention():
    def backward_dw(self):
        self.linear_qkv.backward_dw()
        self.linear_proj.backward_dw()

    def backward_qkv_dw(self):
        self.linear_qkv.backward_dw()

    def backward_proj_dw(self):
        self.linear_proj.backward_dw()
