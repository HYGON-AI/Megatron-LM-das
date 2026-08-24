# Copyright (c) 2026, HYGON-AI. All rights reserved.
#
# Submodule pin verification (patterned after HYGON-AI/verl-das
# tests/hcu/ci/verify_submodules.py):
#   - hcu_megatron is tightly coupled to the Megatron-LM submodule version
#     (interface drift breaks runtime), so CI pins the gitlink SHA and checks
#     both the gitlink and the actual checkout before every test run;
#   - update EXPECTED_SUBMODULES when upgrading submodules.
import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# Must match `git ls-tree HEAD <path>` of the repo; keep in sync when
# submodules are upgraded.
EXPECTED_SUBMODULES = {
    "3rdparty/Megatron-LM": "571370c829ca768fe37244f4e2e7f28d8accc4ab",
    "3rdparty/Megatron-Energon": "ea11c980eb7f0cb22fd25549e1ceebfe710618f5",
    "3rdparty/Megatron-Bridge": "0fbfe7d3e970fbd75c1281d71cee586ce1f3df5e",
}


def parse_gitlink_sha(output: str) -> str | None:
    fields = output.strip().split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        return None
    return fields[2]


def verify_submodules(
    repository_root: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    root = repository_root.resolve()
    errors = []

    for path, expected_sha in EXPECTED_SUBMODULES.items():
        gitlink_result = run(
            ["git", "-C", str(root), "ls-tree", "HEAD", "--", path],
            capture_output=True,
            text=True,
            check=False,
        )
        gitlink_sha = (
            parse_gitlink_sha(gitlink_result.stdout)
            if gitlink_result.returncode == 0
            else None
        )
        if gitlink_sha != expected_sha:
            actual = gitlink_sha or "missing"
            errors.append(
                f"{path} gitlink SHA mismatch: expected {expected_sha}, got {actual}"
            )

        checkout_result = run(
            ["git", "-C", str(root / path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        checkout_sha = (
            checkout_result.stdout.strip() if checkout_result.returncode == 0 else None
        )
        if checkout_sha != expected_sha:
            actual = checkout_sha or "missing"
            errors.append(
                f"{path} checkout SHA mismatch: expected {expected_sha}, got {actual}"
            )

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify pinned Megatron-LM-das CI submodules."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repository root containing the pinned submodules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = verify_submodules(args.repo_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Pinned HCU CI submodules are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
