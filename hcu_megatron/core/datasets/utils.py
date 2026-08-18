# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared across VLM datasets.

Only the utilities actually referenced by this package are kept:

- ``print_rank_0``       — logging helper used across modules.
- ``get_iterator``       — wrap a ``DataLoader`` into a Megatron ``RerunDataIterator``.
- ``random_pad_list``    — pad a token id list with randomly resampled ids.

Anything that was previously here but unused (mask/position helpers, chat
prompt boilerplate, sequence-length inference, arg printers) has been removed
to keep this file narrowly scoped.
"""

import random

import torch


IGNORE_INDEX = -100


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def _cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_iterator(dataloader, dataloader_type="cyclic"):
    """Return dataset iterator wrapped in Megatron's RerunDataIterator."""
    from megatron.core.rerun_state_machine import RerunDataIterator

    if dataloader is None:
        return dataloader
    if dataloader_type == "single":
        return RerunDataIterator(iter(dataloader))
    if dataloader_type == "cyclic":
        return RerunDataIterator(iter(_cyclic_iter(dataloader)))
    raise RuntimeError("unexpected dataloader type")


def random_pad_list(lst, pad_len, ban_token_ids=None):
    assert pad_len >= 0, f'maybe max_seq_len calc wrong {pad_len}'
    if pad_len == 0:
        return lst
    if ban_token_ids is not None:
        filtered_lst = [x for x in lst if x not in ban_token_ids]
        padding = random.choices(filtered_lst, k=pad_len)
    else:
        padding = random.choices(lst, k=pad_len)
    return lst + padding
