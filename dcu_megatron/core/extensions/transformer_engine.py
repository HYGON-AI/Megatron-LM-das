try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch import cpu_offload_v1 as cpu_offload
except ImportError:
    try:
        from transformer_engine.pytorch import cpu_offload
    except ImportError:
        cpu_offload = None
try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
except ImportError:
    Float8Tensor = None
