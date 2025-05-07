from megatron.core.transformer.experts import TEGroupedMLP as MegatronCoreTEGroupedMLP

class TEGroupedMLP(MegatronCoreTEGroupedMLP):
    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()
