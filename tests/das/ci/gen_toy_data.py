#!/usr/bin/env python3
# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Generate mmap data (.bin/.idx) for the functional smoke test:
#   - semantics match megatron NullTokenizer (vocab_size, eod = vocab_size - 1)
#   - written with the upstream IndexedDatasetBuilder (no hand-rolled format)
#   - fixed seed (42), reproducible
#
# See gen_toy_data.sh for the CLI (env DAS_TOY_VOCAB_SIZE / DAS_TOY_NUM_DOCS).
import argparse
import random

import numpy as np
import torch
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-prefix", required=True, help="produces <prefix>.bin/.idx")
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--num-docs", type=int, default=2000)
    ap.add_argument("--doc-tokens-min", type=int, default=24)
    ap.add_argument("--doc-tokens-max", type=int, default=64)
    args = ap.parse_args()

    rng = random.Random(42)
    eod_id = args.vocab_size - 1

    builder = IndexedDatasetBuilder(f"{args.output_prefix}.bin", dtype=np.int32)
    for _ in range(args.num_docs):
        doc = [
            rng.randrange(0, args.vocab_size - 1)
            for _ in range(rng.randint(args.doc_tokens_min, args.doc_tokens_max))
        ]
        doc.append(eod_id)  # append-eod semantics
        builder.add_item(torch.tensor(doc, dtype=torch.int32))
    builder.finalize(f"{args.output_prefix}.idx")

    print(
        f"toy data ready: {args.output_prefix}.bin/.idx "
        f"(vocab={args.vocab_size}, docs={args.num_docs}, eod={eod_id})"
    )


if __name__ == "__main__":
    main()
