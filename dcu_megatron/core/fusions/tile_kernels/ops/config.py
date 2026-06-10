import functools
import torch

_num_sms = 0


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.multi_processor_count


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms()
    return _num_sms


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    if hasattr(prop, 'shared_memory_per_multiprocessor'):
        return prop.shared_memory_per_multiprocessor

    gcn_arch_name = getattr(prop, 'gcnArchName', None)
    gcn_arch_base = gcn_arch_name.split(':', 1)[0] if isinstance(gcn_arch_name, str) else None
    if gcn_arch_base in {'gfx936', 'gfx938'}:
        return 64 * 1024

    raise AttributeError(
        'Unable to determine max shared memory per SM: missing '
        f"'shared_memory_per_multiprocessor' and unsupported "
        f"gcnArchName={gcn_arch_name!r} (base={gcn_arch_base!r})"
    )
