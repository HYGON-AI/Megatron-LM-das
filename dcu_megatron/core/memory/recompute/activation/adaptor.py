from dcu_megatron.core.memory.recompute.activation.activation_recompute_forward import core_activation_recompute_forward_impl


def dcu_activation_recompute_forward(self, hidden_states, per_token_scale=None):
    """MLP.
    Core impl, MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    return core_activation_recompute_forward_impl(self, hidden_states, per_token_scale)
