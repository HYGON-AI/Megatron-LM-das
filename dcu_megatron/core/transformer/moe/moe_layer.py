class MoELayer():
    def backward_dw(self):
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
