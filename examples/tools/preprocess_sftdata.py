#!/usr/bin/env python
"""Convert ms_agent_bench conversations into the SFT jsonl format used by
megatron/training/datasets/sft_dataset.py, then split into train.jsonl /
valid.jsonl for train_qwen3_8B.sh --data_path=<out_dir>.

Source line schema:
    {"id": ..., "conversations": [{"from": "system"|"user"|"assistant",
                                    "value": "..."}]}
Target line schema:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
"""
import argparse
import json
import random
import sys
from pathlib import Path

FROM_TO_ROLE = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "tool": "tool",
    "observation": "tool",
}


def convert(record):
    convs = record.get("conversations") or record.get("messages")
    if not convs:
        return None
    messages = []
    for turn in convs:
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") if "content" in turn else turn.get("value")
        if role is None or content is None:
            return None
        messages.append(
            {"role": FROM_TO_ROLE.get(role.lower(), role.lower()), "content": content}
        )
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return {"messages": messages}


def load_samples(src):
    samples = []
    with open(src, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip line {lineno}: {e}", file=sys.stderr)
                continue
            converted = convert(record)
            if converted is not None:
                samples.append(converted)
    return samples


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default=str(here / "ms_bench" / "ms_agent_bench_v1_sft_sample_head.jsonl"),
        help="Path to source jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(here / "sft_data"),
        help="Directory to write train.jsonl and valid.jsonl into.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to place in valid.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.jsonl"
    valid_out = out_dir / "valid.jsonl"

    samples = load_samples(args.src)
    if not samples:
        raise SystemExit("no valid samples produced")

    random.seed(args.seed)
    random.shuffle(samples)

    if len(samples) > 1:
        n_valid = max(1, int(round(len(samples) * args.valid_ratio)))
    else:
        n_valid = 0
    valid = samples[:n_valid]
    train = samples[n_valid:] or samples

    dump(train_out, train)
    dump(valid_out, valid)

    print(f"total={len(samples)} train={len(train)} valid={len(valid)}")
    print(f"wrote {train_out}")
    print(f"wrote {valid_out}")
    print()
    print("Use with train_qwen3_8B.sh:")
    print(f"  --data_path={out_dir}")


if __name__ == "__main__":
    main()
