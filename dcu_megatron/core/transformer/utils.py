DelayReleaseQKVLinearTensor = False

def get_delay_release_qkv_linear_tensor():
    global DelayReleaseQKVLinearTensor
    return DelayReleaseQKVLinearTensor


def set_delay_release_qkv_linear_tensor(delay_release_qkv_linear_tensor):
    global DelayReleaseQKVLinearTensor
    DelayReleaseQKVLinearTensor = delay_release_qkv_linear_tensor


class DelayReleaseQKVLinearTensorContextManager:
    """A reusable context manager for switch DelayReleaseQKVLinearTensor"""

    def __init__(self, delay_release_qkv_linear_tensor):
        self.delay_release_qkv_linear_tensor = delay_release_qkv_linear_tensor

    def __enter__(self):
        self.origin_delay_release_qkv_linear_tensor = get_delay_release_qkv_linear_tensor()
        set_delay_release_qkv_linear_tensor(self.delay_release_qkv_linear_tensor)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_delay_release_qkv_linear_tensor(self.origin_delay_release_qkv_linear_tensor)
