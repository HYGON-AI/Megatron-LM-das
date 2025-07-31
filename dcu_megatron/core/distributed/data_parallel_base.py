class _BaseDataParallel():
    def backward_dw(self, *inputs, **kwargs):
        """
        Calls the wrapped module's backward_dw() method.
        """
        return self.module.backward_dw(*inputs, **kwargs)
  