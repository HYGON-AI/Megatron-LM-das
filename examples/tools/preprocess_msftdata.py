#!/usr/bin/env python
"""End-to-end preprocessing for the multimodal SFT dataset used by qwen3.5.

Pipeline:
    1. Download the CSV zip archive(s) from ModelScope (train / valid splits).
    2. Unzip the archive(s) into --data-dir.
    3. Concurrently download every image referenced by the CSV(s) into
       <data-dir>/<split>_images/, with retry + resume support.
    4. Convert each CSV row into the multimodal SFT jsonl format expected by
       Megatron:
           {"conversations": [{"role": "user",      "content": "介绍一下图片内容<image>"},
                              {"role": "assistant", "content": "<caption>"}],
            "images": [{"image_path": "<absolute path>"}]}

Layout produced under --data-dir:
    train2014.csv                    (from the zip)
    val2014.csv                      (from the zip)
    train_images/<uniq>_<id>_<file>  (downloaded images)
    valid_images/<uniq>_<id>_<file>
    train.jsonl                      (SFT training set)
    valid.jsonl                      (SFT validation set)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


DEFAULT_URLS = {
    "train": "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/train2014.csv.zip",
    "valid": "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/val2014.csv.zip",
}


def download_file(url: str, dest: Path, timeout: int = 60, chunk: int = 1 << 15) -> Path:
    """Stream ``url`` to ``dest``. Skip when the file is already present."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"↓ {dest.name}", leave=False
        ) as bar:
            for buf in r.iter_content(chunk_size=chunk):
                if not buf:
                    continue
                f.write(buf)
                bar.update(len(buf))
    tmp.rename(dest)
    return dest


def unzip(archive: Path, out_dir: Path) -> list[Path]:
    """Extract ``archive`` into ``out_dir`` and return the list of member paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        # Detect and skip when already extracted.
        need = [m for m in members if not (out_dir / m).exists()]
        if need:
            print(f"[unzip] {archive.name} -> {out_dir} ({len(need)} new files)")
            zf.extractall(out_dir, members=need)
        else:
            print(f"[unzip] {archive.name} already extracted")
    return [out_dir / m for m in members]


def parse_csv(csv_path: Path) -> list[tuple[str, str, str, str]]:
    """Return ``[(uniq_id, image_id, caption, image_url), ...]`` from the CSV."""
    rows: list[tuple[str, str, str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # If the first row isn't a header, keep it as data.
        if header and header[0] != "uniq_id" and len(header) >= 4:
            rows.append((header[0], header[1], header[2], header[3]))
        for row in reader:
            if len(row) >= 4:
                rows.append((row[0], row[1], row[2], row[3]))
    return rows


def image_dest(image_dir: Path, uniq_id: str, image_id: str, url: str) -> Path:
    filename = os.path.basename(urlparse(url).path)
    return image_dir / f"{uniq_id}_{image_id}_{filename}"


def download_image(
    task: tuple[str, str, str, Path], timeout: int, retries: int
) -> tuple[bool, str, str]:
    uniq_id, image_id, url, save_path = task
    if save_path.exists() and save_path.stat().st_size > 0:
        return True, uniq_id, "skipped"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(save_path.suffix + ".part")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for buf in r.iter_content(chunk_size=8192):
                        if buf:
                            f.write(buf)
            tmp.rename(save_path)
            return True, uniq_id, "ok"
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(0.5 * attempt)
    if tmp.exists():
        tmp.unlink()
    return False, uniq_id, f"failed: {last_err}"


def download_images(
    rows: list[tuple[str, str, str, str]],
    image_dir: Path,
    max_workers: int,
    timeout: int,
    retries: int,
) -> list[tuple[str, str]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (uniq, iid, url, image_dest(image_dir, uniq, iid, url))
        for uniq, iid, _, url in rows
    ]
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_image, t, timeout, retries): t for t in tasks}
        with tqdm(total=len(tasks), desc=f"↓ {image_dir.name}", unit="img") as bar:
            for fut in as_completed(futures):
                ok, uid, msg = fut.result()
                if not ok:
                    failures.append((uid, msg))
                bar.update(1)
    return failures


def build_conversation(caption: str, image_path: Path) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": "介绍一下图片内容<image>"},
            {"role": "assistant", "content": caption.strip()},
        ],
        "images": [{"image_path": str(image_path)}],
    }


def rows_to_jsonl(
    rows: list[tuple[str, str, str, str]],
    image_dir: Path,
    out_path: Path,
    require_image: bool = True,
) -> tuple[int, int]:
    written = missing = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for uniq_id, image_id, caption, url in rows:
            img = image_dest(image_dir, uniq_id, image_id, url)
            if require_image and not img.exists():
                missing += 1
                continue
            out.write(json.dumps(build_conversation(caption, img), ensure_ascii=False) + "\n")
            written += 1
    return written, missing


def process_split(
    split: str,
    url: str,
    data_dir: Path,
    max_workers: int,
    timeout: int,
    retries: int,
    skip_images: bool,
    require_image: bool,
    limit: int | None,
) -> None:
    print(f"\n=== split: {split} ===")
    archive = data_dir / Path(urlparse(url).path).name  # e.g. train2014.csv.zip
    download_file(url, archive)
    extracted = unzip(archive, data_dir)

    csv_candidates = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csv_candidates:
        raise RuntimeError(f"no CSV file found inside {archive}")
    csv_path = csv_candidates[0]

    rows = parse_csv(csv_path)
    print(f"[csv] {csv_path.name}: {len(rows)} rows")
    if limit is not None and limit >= 0 and limit < len(rows):
        rows = rows[:limit]
        print(f"[csv] limited to first {len(rows)} rows (--limit)")

    image_dir = data_dir / f"{split}_images"
    if skip_images:
        print(f"[images] skipped (use existing files under {image_dir})")
    else:
        failures = download_images(rows, image_dir, max_workers, timeout, retries)
        print(f"[images] ok={len(rows) - len(failures)} failed={len(failures)}")
        if failures:
            log = data_dir / f"{split}_failed.txt"
            with open(log, "w", encoding="utf-8") as f:
                for uid, msg in failures:
                    f.write(f"{uid}\t{msg}\n")
            print(f"[images] failure log -> {log}")

    out_name = "train.jsonl" if split == "train" else "valid.jsonl"
    out_path = data_dir / out_name
    written, missing = rows_to_jsonl(rows, image_dir, out_path, require_image=require_image)
    print(f"[jsonl] wrote {written} lines -> {out_path} (skipped {missing} rows missing an image)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "coco_2014_caption"),
        help="Working directory for archives, CSVs, images and jsonl outputs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--train-url", default=DEFAULT_URLS["train"], help="CSV zip URL for the train split."
    )
    parser.add_argument(
        "--valid-url", default=DEFAULT_URLS["valid"], help="CSV zip URL for the valid split."
    )
    parser.add_argument("--max-workers", type=int, default=20, help="Concurrent image downloads.")
    parser.add_argument("--timeout", type=int, default=15, help="Per-request timeout (seconds).")
    parser.add_argument("--retries", type=int, default=3, help="Retries per image URL.")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not download images (assume they already exist under <data-dir>/<split>_images).",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Emit jsonl lines even when the referenced image was not downloaded.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N rows of each split (applies to image download and jsonl output). Use to keep runs cheap; omit for the full dataset.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        metavar="N",
        help="Override --limit for the train split only.",
    )
    parser.add_argument(
        "--valid-limit",
        type=int,
        default=None,
        metavar="N",
        help="Override --limit for the valid split only.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data-dir] {data_dir}")

    urls = {"train": args.train_url, "valid": args.valid_url}
    per_split_limits = {"train": args.train_limit, "valid": args.valid_limit}
    for split in args.splits:
        limit = per_split_limits[split] if per_split_limits[split] is not None else args.limit
        process_split(
            split=split,
            url=urls[split],
            data_dir=data_dir,
            max_workers=args.max_workers,
            timeout=args.timeout,
            retries=args.retries,
            skip_images=args.skip_images,
            require_image=not args.keep_missing,
            limit=limit,
        )

    print("\nDone. Use with your training script, e.g.:")
    print(f"  --train-data-path {data_dir / 'train.jsonl'}")
    print(f"  --valid-data-path {data_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
