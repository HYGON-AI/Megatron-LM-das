class MoELayer():
    def backward_dw(self):
        self.experts.backward_dw()
        self.shared_experts.backward_dw()
