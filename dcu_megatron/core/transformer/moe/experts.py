class GroupedMLP():
    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass

    
class TEGroupedMLP():
    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class SequentialMLP():
    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        try:
            for expert in self.local_experts:
                expert.backward_dw()
        except Exception as e:
            raise Exception(
                f"Unknown error occurred during SequentialMLP backward_dw() execution: {str(e)}"
            )    
