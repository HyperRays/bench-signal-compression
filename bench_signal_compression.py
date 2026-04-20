#!/usr/bin/env python3
"""
Benchmark VBZ compression/decompression throughput and compression ratio
for raw signal data extracted from fast5 files.

Usage:
    python benchmarks/bench_signal_compression.py <fast5_path> [options]

    <fast5_path> can be a single .fast5 file or a directory of .fast5 files.

Examples:
    # Benchmark with 4 worker processes
    python benchmarks/bench_signal_compression.py /data/fast5s/ --workers 4

    # Sweep across 1,2,4,8 workers
    python benchmarks/bench_signal_compression.py /data/fast5s/ --sweep 1,2,4,8

    # Sweep with chunked compression
    python benchmarks/bench_signal_compression.py /data/fast5s/ --sweep 1,2,4,8,16 --chunked

    # Single run, verbose per-read stats
    python benchmarks/bench_signal_compression.py /data/fast5s/ --workers 1 --max-reads 500 --verbose
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import numpy.typing as npt

try:
    import vbz_h5py_plugin  # noqa: F401
except ImportError:
    pass

from pod5.signal_tools import (
    DEFAULT_SIGNAL_CHUNK_SIZE,
    vbz_compress_signal,
    vbz_compress_signal_chunked,
    vbz_decompress_signal,
    vbz_decompress_signal_chunked,
)


def find_fast5_files(path: Path) -> List[Path]:
    """Find all fast5 files under the given path."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.fast5"))


def extract_signals_from_fast5(
    path: Path, max_reads: int
) -> List[npt.NDArray[np.int16]]:
    """Extract raw signal arrays from a multi-read fast5 file."""
    signals = []
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            if not key.startswith("read_"):
                continue
            try:
                signal = f[key]["Raw"]["Signal"][()]
                signals.append(signal.astype(np.int16))
            except KeyError:
                continue
            if max_reads > 0 and len(signals) >= max_reads:
                break
    return signals


# -- Worker functions for multiprocessing (must be top-level for pickling) --


def _worker_contiguous(
    signal: npt.NDArray[np.int16],
) -> Tuple[int, int, float, float]:
    """Compress and decompress a single signal, return (raw_bytes, comp_bytes, t_comp, t_decomp)."""
    raw_bytes = signal.nbytes

    t0 = time.perf_counter()
    compressed = vbz_compress_signal(signal)
    t_comp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = vbz_decompress_signal(compressed, len(signal))
    t_decomp = time.perf_counter() - t0

    return raw_bytes, len(compressed), t_comp, t_decomp


def _worker_chunked(args: Tuple[npt.NDArray[np.int16], int]) -> Tuple[int, int, float, float]:
    """Chunked variant — args is (signal, chunk_size)."""
    signal, chunk_size = args
    raw_bytes = signal.nbytes

    t0 = time.perf_counter()
    chunks, chunk_lengths = vbz_compress_signal_chunked(signal, chunk_size)
    t_comp = time.perf_counter() - t0

    compressed_bytes = sum(len(c) for c in chunks)

    t0 = time.perf_counter()
    _ = vbz_decompress_signal_chunked(chunks, chunk_lengths)
    t_decomp = time.perf_counter() - t0

    return raw_bytes, compressed_bytes, t_comp, t_decomp


# -- Benchmark runners --


def run_benchmark(
    signals: List[npt.NDArray[np.int16]],
    workers: int,
    chunked: bool,
    chunk_size: int,
    verbose: bool,
) -> Dict[str, float]:
    """Run the benchmark with the given number of worker processes."""
    total_samples = sum(len(s) for s in signals)

    if workers == 1:
        # Single-process: avoid pool overhead, support verbose
        results = []
        for i, signal in enumerate(signals):
            if chunked:
                r = _worker_chunked((signal, chunk_size))
            else:
                r = _worker_contiguous(signal)
            results.append(r)

            if verbose:
                raw_b, comp_b, tc, td = r
                ratio = raw_b / comp_b if comp_b > 0 else 0
                c_mbps = raw_b / tc / 1e6 if tc > 0 else float("inf")
                d_mbps = raw_b / td / 1e6 if td > 0 else float("inf")
                print(
                    f"  read {i:>6d}: {len(signal):>10,} samples, "
                    f"ratio={ratio:.2f}x, "
                    f"compress={c_mbps:.1f} MB/s, "
                    f"decompress={d_mbps:.1f} MB/s"
                )
    else:
        if verbose:
            print(f"  (--verbose ignored with --workers > 1)")

        # Use spawn context to avoid fork-safety issues with numpy/hdf5
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            if chunked:
                work = [(s, chunk_size) for s in signals]
                results = pool.map(_worker_chunked, work)
            else:
                results = pool.map(_worker_contiguous, signals)

    # Aggregate
    total_raw = sum(r[0] for r in results)
    total_comp = sum(r[1] for r in results)
    # Wall-clock time: for parallel workers, use max of per-worker sums
    # But we actually want total wall time, so we time the whole run above.
    # The per-read times tell us aggregate CPU time.
    total_cpu_comp = sum(r[2] for r in results)
    total_cpu_decomp = sum(r[3] for r in results)

    ratio = total_raw / total_comp if total_comp > 0 else 0
    raw_mb = total_raw / 1e6

    return {
        "workers": workers,
        "reads": len(signals),
        "total_samples": total_samples,
        "raw_mb": raw_mb,
        "compressed_mb": total_comp / 1e6,
        "ratio": ratio,
        "cpu_compress_s": total_cpu_comp,
        "cpu_decompress_s": total_cpu_decomp,
        # Effective throughput = data size / (cpu time / num workers)
        "compress_mbps": raw_mb / (total_cpu_comp / workers) if total_cpu_comp > 0 else float("inf"),
        "decompress_mbps": raw_mb / (total_cpu_decomp / workers) if total_cpu_decomp > 0 else float("inf"),
    }


def run_benchmark_wallclock(
    signals: List[npt.NDArray[np.int16]],
    workers: int,
    chunked: bool,
    chunk_size: int,
    verbose: bool,
) -> Dict[str, float]:
    """Wrapper that also measures true wall-clock time."""
    t0 = time.perf_counter()
    result = run_benchmark(signals, workers, chunked, chunk_size, verbose)
    wall_time = time.perf_counter() - t0

    raw_mb = result["raw_mb"]
    result["wall_time_s"] = wall_time
    result["wall_compress_mbps"] = raw_mb / (wall_time / 2) if wall_time > 0 else float("inf")
    # A rough split: wall time covers both compress + decompress.
    # For a cleaner wall-clock measurement, we run them separately:
    return result


def run_benchmark_split_wallclock(
    signals: List[npt.NDArray[np.int16]],
    workers: int,
    chunked: bool,
    chunk_size: int,
) -> Dict[str, float]:
    """Measure wall-clock time for compression and decompression separately."""
    total_samples = sum(len(s) for s in signals)
    raw_mb = sum(s.nbytes for s in signals) / 1e6

    ctx = mp.get_context("spawn")

    # --- Compression pass ---
    if workers == 1:
        t0 = time.perf_counter()
        if chunked:
            compressed = [vbz_compress_signal_chunked(s, chunk_size) for s in signals]
        else:
            compressed = [vbz_compress_signal(s) for s in signals]
        wall_compress = time.perf_counter() - t0
    else:
        if chunked:
            work = [(s, chunk_size) for s in signals]
            t0 = time.perf_counter()
            with ctx.Pool(processes=workers) as pool:
                compressed = pool.starmap(vbz_compress_signal_chunked, work)
            wall_compress = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            with ctx.Pool(processes=workers) as pool:
                compressed = pool.map(vbz_compress_signal, signals)
            wall_compress = time.perf_counter() - t0

    # Calculate compressed size
    if chunked:
        total_comp_bytes = sum(sum(len(c) for c in chunks) for chunks, _ in compressed)
    else:
        total_comp_bytes = sum(len(c) for c in compressed)

    # --- Decompression pass ---
    if chunked:
        if workers == 1:
            t0 = time.perf_counter()
            for chunks, chunk_lengths in compressed:
                vbz_decompress_signal_chunked(chunks, chunk_lengths)
            wall_decompress = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            with ctx.Pool(processes=workers) as pool:
                pool.starmap(vbz_decompress_signal_chunked, compressed)
            wall_decompress = time.perf_counter() - t0
    else:
        decomp_args = list(zip(compressed, [len(s) for s in signals]))
        if workers == 1:
            t0 = time.perf_counter()
            for c, n in decomp_args:
                vbz_decompress_signal(c, n)
            wall_decompress = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            with ctx.Pool(processes=workers) as pool:
                pool.starmap(vbz_decompress_signal, decomp_args)
            wall_decompress = time.perf_counter() - t0

    total_raw = sum(s.nbytes for s in signals)
    ratio = total_raw / total_comp_bytes if total_comp_bytes > 0 else 0

    return {
        "workers": workers,
        "reads": len(signals),
        "total_samples": total_samples,
        "raw_mb": raw_mb,
        "compressed_mb": total_comp_bytes / 1e6,
        "ratio": ratio,
        "wall_compress_s": wall_compress,
        "wall_decompress_s": wall_decompress,
        "compress_mbps": raw_mb / wall_compress if wall_compress > 0 else float("inf"),
        "decompress_mbps": raw_mb / wall_decompress if wall_decompress > 0 else float("inf"),
    }


def print_result(result: Dict[str, float], mode: str, chunk_size: int) -> None:
    """Print a single benchmark result."""
    print(f"  Workers:                  {result['workers']}")
    print(f"  Mode:                     {mode}")
    if mode == "chunked":
        print(f"  Chunk size:               {chunk_size:,} samples")
    print(f"  Reads:                    {result['reads']:,}")
    print(f"  Total samples:            {result['total_samples']:,}")
    print(f"  Raw size:                 {result['raw_mb']:.2f} MB")
    print(f"  Compressed size:          {result['compressed_mb']:.2f} MB")
    print(f"  Compression ratio:        {result['ratio']:.2f}x")
    print(f"  Compress wall time:       {result['wall_compress_s']:.3f} s")
    print(f"  Decompress wall time:     {result['wall_decompress_s']:.3f} s")
    print(f"  Compression throughput:   {result['compress_mbps']:.1f} MB/s")
    print(f"  Decompression throughput: {result['decompress_mbps']:.1f} MB/s")


def print_sweep_table(results: List[Dict[str, float]]) -> None:
    """Print a summary table for a sweep across worker counts."""
    print(f"\n{'Workers':>8} {'Ratio':>8} {'Comp MB/s':>11} {'Decomp MB/s':>13} "
          f"{'Comp Wall(s)':>13} {'Decomp Wall(s)':>15}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['workers']:>8d} "
            f"{r['ratio']:>8.2f} "
            f"{r['compress_mbps']:>11.1f} "
            f"{r['decompress_mbps']:>13.1f} "
            f"{r['wall_compress_s']:>13.3f} "
            f"{r['wall_decompress_s']:>15.3f}"
        )

    # Speedup relative to single-worker
    if len(results) > 1:
        base_comp = results[0]["wall_compress_s"]
        base_decomp = results[0]["wall_decompress_s"]
        print(f"\n{'Workers':>8} {'Comp Speedup':>13} {'Decomp Speedup':>15}")
        print("-" * 40)
        for r in results:
            comp_speedup = base_comp / r["wall_compress_s"] if r["wall_compress_s"] > 0 else 0
            decomp_speedup = base_decomp / r["wall_decompress_s"] if r["wall_decompress_s"] > 0 else 0
            print(f"{r['workers']:>8d} {comp_speedup:>13.2f}x {decomp_speedup:>15.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VBZ signal compression throughput and ratio"
    )
    parser.add_argument(
        "fast5_path",
        type=Path,
        help="Path to a .fast5 file or directory of .fast5 files",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=0,
        help="Max reads to benchmark (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated worker counts to sweep, e.g. '1,2,4,8'",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked compression (matches pod5 default behavior)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_SIGNAL_CHUNK_SIZE,
        help=f"Samples per chunk when using --chunked (default: {DEFAULT_SIGNAL_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup reads before timing (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-read statistics (single-worker only)",
    )
    args = parser.parse_args()

    fast5_path = args.fast5_path.resolve()
    if not fast5_path.exists():
        print(f"Error: {fast5_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Parse sweep worker counts
    if args.sweep:
        worker_counts = [int(x.strip()) for x in args.sweep.split(",")]
    else:
        worker_counts = [args.workers]

    # Discover fast5 files
    fast5_files = find_fast5_files(fast5_path)
    if not fast5_files:
        print(f"Error: no .fast5 files found under {fast5_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(fast5_files)} fast5 file(s)")

    # Extract signals
    print("Extracting raw signals from fast5 files...")
    signals: List[npt.NDArray[np.int16]] = []
    remaining = args.max_reads
    for f5 in fast5_files:
        batch = extract_signals_from_fast5(f5, remaining)
        signals.extend(batch)
        if args.max_reads > 0:
            remaining = args.max_reads - len(signals)
            if remaining <= 0:
                break

    if not signals:
        print("Error: no reads found in fast5 files", file=sys.stderr)
        sys.exit(1)

    total_samples = sum(len(s) for s in signals)
    total_raw_mb = sum(s.nbytes for s in signals) / 1e6
    print(
        f"Loaded {len(signals)} reads, {total_samples:,} total samples "
        f"({total_raw_mb:.1f} MB raw)"
    )

    # Warmup
    warmup_count = min(args.warmup, len(signals))
    if warmup_count > 0:
        print(f"Warming up with {warmup_count} reads...")
        for s in signals[:warmup_count]:
            c = vbz_compress_signal(s)
            _ = vbz_decompress_signal(c, len(s))

    mode = "chunked" if args.chunked else "contiguous"

    # Run benchmark(s)
    all_results = []
    for wc in worker_counts:
        print(f"\nBenchmarking {mode} compression with {wc} worker(s)...")
        result = run_benchmark_split_wallclock(
            signals, wc, args.chunked, args.chunk_size
        )
        all_results.append(result)

        print("=" * 60)
        print_result(result, mode, args.chunk_size)
        print("=" * 60)

    # Print sweep summary table
    if len(all_results) > 1:
        print_sweep_table(all_results)


if __name__ == "__main__":
    main()
