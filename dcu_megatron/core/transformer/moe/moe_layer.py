from megatron.core.transformer.moe.moe_layer import MoELayer as MegatronCoreMoELayer


class MoELayer(MegatronCoreMoELayer):
    def backward_dw(self):
        self.experts.backward_dw()
        self.shared_experts.backward_dw()
