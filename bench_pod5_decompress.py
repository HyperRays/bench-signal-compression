#!/usr/bin/env python3
"""
Benchmark pod5 full-decode throughput with file I/O effectively removed.

Strategy:
  1. Pre-read the entire .pod5 file into the OS page cache (untimed).
     Pod5 uses mmap by default, so every subsequent access is a memcpy
     from cached RAM — indistinguishable from an in-memory reader.
  2. Open `pod5.Reader(path)` and run N warmup passes (untimed).
  3. Time M passes where we iterate all reads and materialize `read.signal`
     (forces the full decompression path through the C++ backend).

Usage:
    python bench_pod5_decompress.py <pod5_path> [options]

    <pod5_path> may be a .pod5 file or a directory containing .pod5 files.
"""

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

import pod5


def find_pod5_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.pod5"))


def warm_page_cache(files: List[Path]) -> int:
    """Read each file end-to-end to force pages into the OS cache. Returns total bytes."""
    total = 0
    buf = bytearray(1 << 20)
    mv = memoryview(buf)
    for f in files:
        with open(f, "rb") as fh:
            while True:
                n = fh.readinto(mv)
                if not n:
                    break
                total += n
    return total


def decode_one_file(path: Path) -> Tuple[int, int]:
    """Iterate every read and materialize its signal. Returns (read_count, sample_count)."""
    reads = 0
    samples = 0
    with pod5.Reader(path) as reader:
        for read in reader.reads():
            sig = read.signal
            samples += sig.shape[0]
            reads += 1
    return reads, samples


def _worker_decode(path: str) -> Tuple[int, int]:
    return decode_one_file(Path(path))


def time_pass(files: List[Path], workers: int) -> Tuple[float, int, int]:
    """One timed pass over every file. Returns (wall_seconds, reads, samples)."""
    if workers == 1:
        t0 = time.perf_counter()
        total_reads = 0
        total_samples = 0
        for f in files:
            r, s = decode_one_file(f)
            total_reads += r
            total_samples += s
        return time.perf_counter() - t0, total_reads, total_samples

    ctx = mp.get_context("spawn")
    paths = [str(f) for f in files]
    t0 = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_worker_decode, paths)
    elapsed = time.perf_counter() - t0
    total_reads = sum(r for r, _ in results)
    total_samples = sum(s for _, s in results)
    return elapsed, total_reads, total_samples


def main():
    p = argparse.ArgumentParser(
        description="Benchmark pod5 decompression with file I/O removed via page-cache prewarm"
    )
    p.add_argument("pod5_path", type=Path, help="Path to a .pod5 file or directory")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2, help="Warmup passes (default: 2)")
    p.add_argument("--repeats", type=int, default=5, help="Timed passes (default: 5)")
    p.add_argument(
        "--skip-prewarm",
        action="store_true",
        help="Skip the page-cache prewarm (use if you want to include cold I/O)",
    )
    args = p.parse_args()

    pod5_path = args.pod5_path.resolve()
    if not pod5_path.exists():
        print(f"Error: {pod5_path} does not exist", file=sys.stderr)
        sys.exit(1)

    files = find_pod5_files(pod5_path)
    if not files:
        print(f"Error: no .pod5 files under {pod5_path}", file=sys.stderr)
        sys.exit(1)

    on_disk_bytes = sum(f.stat().st_size for f in files)
    print(f"Found {len(files)} .pod5 file(s), {on_disk_bytes / 1e6:.2f} MB on disk")

    if not args.skip_prewarm:
        print("Prewarming OS page cache (untimed)...")
        t0 = time.perf_counter()
        n = warm_page_cache(files)
        dt = time.perf_counter() - t0
        print(f"  read {n / 1e6:.2f} MB in {dt:.2f}s ({n / dt / 1e6:.1f} MB/s)")

    # Determine dataset size by running one (untimed) discovery pass.
    print("Discovering dataset size (untimed)...")
    t0 = time.perf_counter()
    total_reads = 0
    total_samples = 0
    for f in files:
        r, s = decode_one_file(f)
        total_reads += r
        total_samples += s
    discovery_time = time.perf_counter() - t0
    raw_mb = (total_samples * 2) / 1e6  # int16
    print(
        f"  {total_reads:,} reads, {total_samples:,} samples "
        f"({raw_mb:.2f} MB raw int16) — first full pass took {discovery_time:.2f}s"
    )

    if args.warmup > 0:
        print(f"Warmup: {args.warmup} pass(es)...")
        for _ in range(args.warmup):
            time_pass(files, args.workers)

    print(f"Timing: {args.repeats} pass(es), workers={args.workers}...")
    times: List[float] = []
    for i in range(args.repeats):
        t, _, _ = time_pass(files, args.workers)
        print(f"  pass {i + 1}: {t:.4f} s  ({raw_mb / t:.1f} MB/s raw, "
              f"{total_samples / t / 1e6:.2f} Msamples/s)")
        times.append(t)

    best = min(times)
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0

    print("=" * 60)
    print(f"  Files:                    {len(files)}")
    print(f"  Workers:                  {args.workers}")
    print(f"  Reads / pass:             {total_reads:,}")
    print(f"  Samples / pass:           {total_samples:,}")
    print(f"  Raw / pass:               {raw_mb:.2f} MB (int16)")
    print(f"  On-disk compressed:       {on_disk_bytes / 1e6:.2f} MB")
    print(f"  Compression ratio:        {(total_samples * 2) / on_disk_bytes:.2f}x (raw/on-disk)")
    print(f"  Best time:                {best:.4f} s  ({raw_mb / best:.1f} MB/s)")
    print(f"  Median time:              {median:.4f} s  ({raw_mb / median:.1f} MB/s)")
    print(f"  Mean ± stdev:             {mean:.4f} ± {stdev:.4f} s")
    print(f"  Samples/sec (median):     {total_samples / median:,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
