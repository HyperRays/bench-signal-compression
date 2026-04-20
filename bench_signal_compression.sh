#!/bin/bash
#SBATCH --job-name=vbz-bench
#SBATCH --output=vbz-bench-%j.out
#SBATCH --error=vbz-bench-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --partition=bio_part

# ── Configuration ────────────────────────────────────────────────────
# Override these via environment or edit directly:
#   sbatch --export=FAST5_PATH=/data/fast5s,SWEEP="1,2,4,8,16" bench_signal_compression.sh
FAST5_PATH="${FAST5_PATH:?Set FAST5_PATH to a .fast5 file or directory}"
SWEEP="${SWEEP:-8,16,32}"
MAX_READS="${MAX_READS:-0}"
CHUNKED="${CHUNKED:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-102400}"
WARMUP="${WARMUP:-10}"

# ── Environment setup ────────────────────────────────────────────────
# Uncomment / edit whichever applies to your cluster:
# module load python/3.10
# source /path/to/venv/bin/activate
# conda activate pod5-env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Build command ────────────────────────────────────────────────────
CMD="python3 ${SCRIPT_DIR}/bench_signal_compression.py ${FAST5_PATH}"
CMD+=" --sweep ${SWEEP}"
CMD+=" --warmup ${WARMUP}"

if [ "${MAX_READS}" -ne 0 ]; then
    CMD+=" --max-reads ${MAX_READS}"
fi

if [ "${CHUNKED}" -eq 1 ]; then
    CMD+=" --chunked --chunk-size ${CHUNK_SIZE}"
fi

# ── Run ──────────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Fast5 path:   ${FAST5_PATH}"
echo "Sweep:        ${SWEEP}"
echo "Max reads:    ${MAX_READS}"
echo "Chunked:      ${CHUNKED}"
echo "Chunk size:   ${CHUNK_SIZE}"
echo "============================================"
echo ""
echo "Running: ${CMD}"
echo ""

${CMD}
