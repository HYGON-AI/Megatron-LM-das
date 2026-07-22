from typing import Union

import torch


def print_rank_message(message, rank_id: Union[int, set, list]):
    """If distributed is initialized, print only on rank specified by rank_id."""
    if isinstance(rank_id, int):
        rank_id = {rank_id}
    rank_id = set(rank_id)
    if torch.distributed.is_initialized():
        current_rank = torch.distributed.get_rank()
        if not rank_id or current_rank in rank_id:
            print(f"[rank {current_rank}] {message}", flush=True)
    else:
        print(message, flush=True)
