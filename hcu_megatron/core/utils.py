import torch

from statistics import mean
from collections import defaultdict
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from .parallel_state import _COMM_LOGS


_flux_version = None


def get_flux_version():
    """Get flux version from __version__; if not available use pip's. Use caching."""

    def get_flux_version_str():
        import flux

        if hasattr(flux, '__version__'):
            return str(flux.__version__)
        else:
            return version("flux")

    global _flux_version
    if _flux_version is None:
        _flux_version = PkgVersion(get_flux_version_str())
    return _flux_version


def is_flux_min_version(version, check_equality=True):
    """Check if minimum version of `flux` is installed."""
    if check_equality:
        return get_flux_version() >= PkgVersion(version)
    return get_flux_version() > PkgVersion(version)


def _get_elapsed_time_all_ranks(barrier):
    # First make sure all the callers are in sync.
    if barrier:
        torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    grouped = defaultdict(list)
    for entry in _COMM_LOGS:
        key = (entry['type'], entry['group'])
        grouped[key].append(entry['time'])

    rank_name_to_time = torch.zeros(
            (world_size, len(grouped.keys())), dtype=torch.float, device=torch.cuda.current_device()
        )

    for i, name in enumerate(grouped):
        rank_name_to_time[rank, i] = mean(grouped[name])

    torch.distributed.all_gather_into_tensor(rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1))

    return rank_name_to_time, grouped


def _get_global_min_max_time(barrier):
    rank_name_to_time, grouped = _get_elapsed_time_all_ranks(barrier)
    name_to_min_max_time = {}
    for i, name in enumerate(grouped):
        rank_to_time = rank_name_to_time[:, i]
        # filter out the ones we did not have any timings for
        rank_to_time = rank_to_time[rank_to_time > 0.0]
        # If the timer exists:
        if rank_to_time.numel() > 0:
            name_to_min_max_time[name] = (
                rank_to_time.min().item() / 0.001,
                rank_to_time.max().item() / 0.001,
            )    

    return name_to_min_max_time


def _get_global_min_max_time_string(barrier):
    name_to_min_max_time = _get_global_min_max_time(barrier)
    output_string = '(min, max) time across ranks (ms):'
    for name in name_to_min_max_time:
        min_time, max_time = name_to_min_max_time[name]
        name = ":".join(name)
        output_string += f"\n    {(name + ' ').ljust(48, '.')}: ({min_time:.2f}, {max_time:.2f})"

    return output_string


def log(
        rank: int = None,
        barrier: bool = False,
    ):
    output_string = _get_global_min_max_time_string(barrier)
    # If no input rank is provided, log on last rank.
    if rank is None:
        rank = torch.distributed.get_world_size() - 1
    if rank == torch.distributed.get_rank() and output_string is not None:
        print(output_string, flush=True)
