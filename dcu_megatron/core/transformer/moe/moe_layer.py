class MoELayer():
    def backward_dw(self):
        self.backward_routed_expert_dw()
        self.backward_shared_expert_dw()

    def backward_shared_expert_dw(self):
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def backward_routed_expert_dw(self):
        self.experts.backward_dw()
