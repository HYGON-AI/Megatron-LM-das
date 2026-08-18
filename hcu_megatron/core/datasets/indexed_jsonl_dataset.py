# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Indexed JSONL dataset for VLM multimodal training.

Overview
========
Reads JSONL files split across multiple *domains* (subject-matter buckets, e.g.
"math" / "chat" / "code"), interleaves samples across domains according to a
per-domain probability, and yields the raw JSON dict to upstream code (which
handles tokenization + image loading).

Domain scheduling
=================
``generate_global_batch_domain_id(gbs, probs)`` builds a length-``gbs`` list of
per-batch-slot domain ids. Example (gbs=16, probs=[0.5, 0.3, 0.2]):

    [0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2]

Each ``__iter__`` step consumes one slot, wraps around at ``gbs``, so the
long-run domain mix matches ``probs``.

For DP training, the same list is sliced per rank so domain assignment is
deterministic and non-overlapping across ranks.

History
=======
This module previously carried an unused text-only pretrain variant and a
never-wired ``ConsumedByThisRank`` / ``update_consumed`` consumption-tracking
system. Both were removed; only the multimodal variant remains. If accurate
resume-from-checkpoint is needed, add it back and wire it into
``hcu_megatron/training/training.py``'s checkpoint path.
"""

from __future__ import annotations

import glob
import json
import os

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from hcu_megatron.core.datasets.utils import print_rank_0


# ---------------------------------------------------------------------------
# Domain scheduling helpers
# ---------------------------------------------------------------------------
def adjust_domain_id_list_for_dp(gbs, dp_rank, dp_world_size, domain_id_list):
    """Slice a length-gbs domain id list into the per-rank subrange."""
    fake_ring_domain_id_list = domain_id_list + domain_id_list
    offset = gbs // dp_world_size
    start_index = offset * dp_rank
    end_index = start_index + gbs
    return fake_ring_domain_id_list[start_index:end_index]


def generate_global_batch_domain_id(
    gbs, domains_probs, dp_rank, dp_world_size, top_domains_to_cut=1,
):
    """Build a length-gbs list of domain ids honouring ``domains_probs``.

    If the raw ``int(p * gbs)`` allocation doesn't sum to ``gbs``, the delta is
    distributed across the ``top_domains_to_cut`` largest domains.

    If ``domains_probs`` is None, distributes ``gbs`` uniformly across domains
    — callers may rely on this for eval-only runs where mixing weights are moot.
    """
    assert gbs >= dp_world_size and gbs % dp_world_size == 0
    if domains_probs is None:
        raise ValueError("domains_probs must not be None; got None")

    domain_samples = [max(1, int(p * gbs)) for p in domains_probs]
    total_samples = sum(domain_samples)

    if total_samples != gbs:
        top_domain_indices = sorted(
            range(len(domain_samples)), key=lambda i: domain_samples[i], reverse=True,
        )[:top_domains_to_cut]
        differences = abs(total_samples - gbs)
        for index in top_domain_indices:
            cut_nums = differences // top_domains_to_cut + int(
                index < differences % top_domains_to_cut,
            )
            # If this fires, domains outnumber gbs (e.g. 20 domains, gbs=10) —
            # rethink the config or don't use this dataset.
            assert cut_nums < domain_samples[index], "cannot cut more samples than allocated"
            if total_samples < gbs:
                domain_samples[index] += cut_nums
            else:
                domain_samples[index] -= cut_nums

    domain_id_list = []
    for domain_id, num_samples in enumerate(domain_samples):
        domain_id_list.extend([domain_id] * num_samples)
    assert len(domain_id_list) == gbs, "samples num mismatch"
    return adjust_domain_id_list_for_dp(gbs, dp_rank, dp_world_size, domain_id_list)


# ---------------------------------------------------------------------------
# Base indexed JSONL dataset
# ---------------------------------------------------------------------------
class BaseIndexedJsonlDataset(TorchIterableDataset):
    """Iterable base class for JSONL datasets with domain-weighted scheduling.

    Subclasses implement ``__iter__`` to define what to do with each raw JSON
    record. This class owns:

      - resolving each domain's file list at init time
      - the length-``global_batch_size`` domain schedule
      - lazy per-domain iterator creation
      - epoch rollover when a domain runs out of samples
    """

    def __init__(
        self,
        path_likes,                    # list[str] — one path (dir or .jsonl) per domain
        domain_probabilities,          # list[float] — sampling probabilities per domain
        domain_names,                  # list[str] — display names per domain
        global_batch_size,             # int
        rank=0,                        # global rank
        dp_rank=0,                     # data-parallel rank
        dp_size=1,                     # data-parallel world size
        num_workers=1,                 # DataLoader workers per rank
        seed=0,
        train=False,
        top_domains_to_cut=1,          # how many largest domains to rebalance
    ):
        assert isinstance(path_likes, list) and len(path_likes) > 0
        assert domain_probabilities is not None, \
            "domain_probabilities is required (list of floats, one per domain)"
        assert len(domain_probabilities) == len(path_likes)
        assert domain_names is not None and len(domain_names) == len(path_likes)
        assert top_domains_to_cut <= len(domain_names)

        self.path_likes = path_likes
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.rank = rank
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.num_workers = num_workers
        self.global_batch_size = global_batch_size
        self.top_domains_to_cut = top_domains_to_cut

        self.train = train
        self.seed = seed

        self._print_domain_id_map()

        # Resolve each domain's file list once; avoids re-globbing at every epoch rollover.
        self.files_by_domain = [self._resolve_files(p) for p in path_likes]
        # Per-domain in-session epoch counter (in-memory only — not persisted).
        self.epoch_by_domain = [0] * len(path_likes)

        # Lazy per-domain iterator list, populated on first __iter__ entry.
        self.ds_list = None

        # Length-gbs domain schedule, sliced to this dp_rank.
        self.global_batch_domain_id = generate_global_batch_domain_id(
            gbs=global_batch_size,
            domains_probs=domain_probabilities,
            dp_rank=dp_rank,
            dp_world_size=dp_size,
            top_domains_to_cut=top_domains_to_cut,
        )
        if self.dp_rank in (0, 1):
            print(
                f"Dataset init dp rank {dp_rank} train {train} "
                f"global_batch_domain_id {self.global_batch_domain_id}"
            )

        # Cursor into ``global_batch_domain_id``.
        self.domain_cand_off = 0
        self.in_iter = False

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _print_domain_id_map(self):
        domain_id_map = []
        for domain_id, path_like in enumerate(self.path_likes):
            domain_id_map.append({
                "domain_id": domain_id,
                "domain_name": self.domain_names[domain_id],
                "domain_path_like": path_like,
                "domain_probabilities": self.domain_probabilities[domain_id],
            })
        print_rank_0(
            type(self).__name__ + " id / domain mapping "
            + json.dumps(domain_id_map, indent=4)
        )

    @staticmethod
    def _resolve_files(path_like):
        """Expand a path-like (directory or single .jsonl) into a sorted file list."""
        if os.path.isdir(path_like):
            files = sorted(glob.glob(os.path.join(path_like, "*.jsonl")))
        else:
            files = [path_like]
        if not files:
            raise FileNotFoundError(f"no .jsonl files found under {path_like!r}")
        return files

    # ------------------------------------------------------------------
    # Iterator plumbing
    # ------------------------------------------------------------------
    def _iter_domain_files(self, domain_id):
        """Yield each parsed JSON record from ``self.files_by_domain[domain_id]``.

        The domain id is stamped onto every record so downstream can dispatch
        without threading it through the pipeline separately.
        """
        for fpath in self.files_by_domain[domain_id]:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj["domain_id"] = torch.tensor(domain_id, dtype=torch.int64)
                    yield obj

    def _advance_epoch(self, domain_id):
        """In-session epoch bump. Emitted in yielded dicts as ``domain_epoch``."""
        self.epoch_by_domain[domain_id] += 1

    def _make_domain_iterators(self):
        """Build one iterator per domain and peek the first element as a fail-fast check."""
        import itertools

        ds_list = []
        for domain_id in range(len(self.path_likes)):
            ds_iter = self._iter_domain_files(domain_id)
            try:
                first = next(ds_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"domain {domain_id} ({self.domain_names[domain_id]!r}) has zero "
                    f"records under {self.path_likes[domain_id]!r}"
                ) from exc
            ds_list.append(itertools.chain([first], ds_iter))
        return ds_list

    def __iter__(self):
        raise NotImplementedError("Subclasses must implement __iter__")


# ---------------------------------------------------------------------------
# Multimodal indexed JSONL dataset
# ---------------------------------------------------------------------------
class MultimodalIndexedJsonlDataset(BaseIndexedJsonlDataset):
    """Multimodal iterable JSONL dataset.

    Yields raw JSON dicts (conversations + images) to upstream, which handles
    tokenization + image processing. See ``mm_dataset.MultiModalDataset``.
    """

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True

        if self.ds_list is None:
            self.ds_list = self._make_domain_iterators()

        while True:
            domain_id = self.global_batch_domain_id[self.domain_cand_off]
            ds = self.ds_list[domain_id]

            try:
                example = next(ds)
            except StopIteration:
                # domain exhausted → new epoch, retry once
                self._advance_epoch(domain_id)
                self.ds_list[domain_id] = self._iter_domain_files(domain_id)
                ds = self.ds_list[domain_id]
                example = next(ds)

            # ``_iter_domain_files`` already stamped domain_id onto the record.
            # We only need to pop it out into the wrapper dict.
            assert example["domain_id"].item() == domain_id
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0

            ret_d = {
                "json_data": example,
                "train": self.train,
                "domain_id": example["domain_id"],
                "worker_id": torch.tensor(worker_id, dtype=torch.int64),
                "domain_epoch": torch.tensor(self.epoch_by_domain[domain_id], dtype=torch.int64),
                "domain_line": 1,
                "domain_cand_off": self.domain_cand_off,
            }
            self.domain_cand_off = (self.domain_cand_off + 1) % len(self.global_batch_domain_id)
            yield ret_d
