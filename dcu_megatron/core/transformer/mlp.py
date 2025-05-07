from megatron.core.transformer.mlp import MLP as MegatronCoreMLP

class MLP(MegatronCoreMLP):
    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()
	