#!/usr/bin/env bash
# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Generate mmap data (.bin/.idx) for the functional smoke test, with
# NullTokenizer-compatible semantics.
#
#   --output-prefix  required, produces <prefix>.bin/.idx
#   --num-docs       document count (default 2000)
#   DAS_TOY_VOCAB_SIZE  vocab size (default 256, eod = vocab_size - 1)
set -euo pipefail

usage() {
    echo "Usage: $0 --output-prefix PREFIX [--num-docs N]" >&2
    exit 1
}

OUTPUT_PREFIX=""
NUM_DOCS=""

while [[ $# -gt 0 ]]; do
    case $1 in
    --output-prefix)
        OUTPUT_PREFIX="$2"
        shift 2
        ;;
    --num-docs)
        NUM_DOCS="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

if [[ -z "${OUTPUT_PREFIX}" ]]; then
    echo "Error: --output-prefix is required"
    usage
fi
if [[ -z "${NUM_DOCS}" ]]; then
    NUM_DOCS=2000
fi
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/gen_toy_data.py" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --vocab-size "${DAS_TOY_VOCAB_SIZE:-256}" \
    --num-docs "${NUM_DOCS}"

echo "toy data ready: ${OUTPUT_PREFIX}.bin/.idx"
