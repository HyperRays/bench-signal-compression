#!/usr/bin/env python3
"""
Minimal fast5 -> pod5 converter that handles both single-read and multi-read
fast5 layouts. Modelled on the signal-extraction logic in
bench_signal_compression.py: we read raw signal arrays directly via h5py and
synthesize the non-signal metadata pod5 requires. Metadata is NOT faithful to
the source — this tool exists purely to produce pod5 files whose signal data
is correct for decompression benchmarking.

Usage:
    python3 fast5_to_pod5.py <fast5_dir_or_file> <output.pod5>
"""

import argparse
import datetime
import sys
import uuid
from pathlib import Path
from typing import Iterator, Tuple

import h5py
import numpy as np

try:
    import vbz_h5py_plugin  # noqa: F401
except ImportError:
    pass

import pod5
from pod5 import (
    Calibration,
    EndReason,
    EndReasonEnum,
    Pore,
    Read,
    RunInfo,
)


def find_fast5_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.fast5"))


def iter_signals(path: Path) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield (signal_int16, channel) from a fast5 file, single- or multi-read."""
    with h5py.File(str(path), "r") as f:
        if "Raw" in f and "Reads" in f["Raw"]:
            for read_name in f["Raw"]["Reads"]:
                try:
                    sig = f["Raw"]["Reads"][read_name]["Signal"][()]
                except KeyError:
                    continue
                try:
                    channel = int(f["UniqueGlobalKey"]["channel_id"].attrs["channel_number"])
                except (KeyError, ValueError):
                    channel = 1
                yield sig.astype(np.int16), channel
        else:
            for key in f.keys():
                if not key.startswith("read_"):
                    continue
                try:
                    sig = f[key]["Raw"]["Signal"][()]
                except KeyError:
                    continue
                try:
                    channel = int(f[key]["channel_id"].attrs["channel_number"])
                except (KeyError, ValueError):
                    channel = 1
                yield sig.astype(np.int16), channel


def make_dummy_run_info() -> RunInfo:
    now = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    return RunInfo(
        acquisition_id="bench-acq",
        acquisition_start_time=now,
        adc_max=8191,
        adc_min=-4096,
        context_tags={},
        experiment_name="bench",
        flow_cell_id="bench-fc",
        flow_cell_product_code="bench",
        protocol_name="bench",
        protocol_run_id="bench-run",
        protocol_start_time=now,
        sample_id="bench",
        sample_rate=4000,
        sequencing_kit="bench",
        sequencer_position="bench",
        sequencer_position_type="bench",
        software="fast5_to_pod5.py",
        system_name="bench",
        system_type="bench",
        tracking_id={},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help=".fast5 file or directory")
    ap.add_argument("output", type=Path, help="output .pod5 file")
    ap.add_argument("--batch", type=int, default=1000, help="reads per writer batch")
    args = ap.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.exists():
        print(f"Error: {inp} does not exist", file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = find_fast5_files(inp)
    if not files:
        print(f"Error: no .fast5 files under {inp}", file=sys.stderr)
        sys.exit(1)
    print(f"Converting {len(files)} fast5 file(s) -> {out}")

    run_info = make_dummy_run_info()
    calibration = Calibration(offset=0.0, scale=1.0)
    end_reason = EndReason(reason=EndReasonEnum.UNKNOWN, forced=False)

    total_reads = 0
    total_samples = 0
    batch: list = []

    with pod5.Writer(out) as writer:
        read_number = 0
        for f in files:
            for signal, channel in iter_signals(f):
                read = Read(
                    read_id=uuid.uuid4(),
                    pore=Pore(channel=channel, well=1, pore_type="bench"),
                    calibration=calibration,
                    read_number=read_number,
                    start_sample=0,
                    median_before=0.0,
                    end_reason=end_reason,
                    run_info=run_info,
                    signal=signal,
                )
                batch.append(read)
                read_number += 1
                total_reads += 1
                total_samples += signal.shape[0]

                if len(batch) >= args.batch:
                    writer.add_reads(batch)
                    batch.clear()

        if batch:
            writer.add_reads(batch)

    print(
        f"Wrote {total_reads:,} reads, {total_samples:,} samples "
        f"({(total_samples * 2) / 1e6:.2f} MB raw int16) -> {out} "
        f"({out.stat().st_size / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()