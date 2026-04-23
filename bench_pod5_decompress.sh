#!/bin/bash
#SBATCH --job-name=pod5-decompress-bench
#SBATCH --output=pod5-decompress-bench-%j.out
#SBATCH --error=pod5-decompress-bench-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --partition=bio_part

# ── Configuration ────────────────────────────────────────────────────
# dataset_label:fast5_dir pairs
DATASETS=(
    "d4_green_algae_r94:/mnt/nvme1/soysalm/d4_green_algae_r94/fast5_files/"
    "d3_yeast_r94:/mnt/nvme1/soysalm/d3_yeast_r94/fast5_files/"
)

# Cache directory for converted pod5 files. Conversion is skipped if the
# output already exists (delete the file to force reconversion).
POD5_CACHE="${POD5_CACHE:-${SLURM_SUBMIT_DIR:-$PWD}/pod5_cache}"

SWEEP="${SWEEP:-8,16,32}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-5}"
SKIP_PREWARM="${SKIP_PREWARM:-0}"

# ── Environment setup ────────────────────────────────────────────────
# Uncomment / edit whichever applies to your cluster:
# module load python/3.10
# source /path/to/venv/bin/activate
# conda activate pod5-env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_THREADS="${CONVERT_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p "${POD5_CACHE}"

# ── Summary ──────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           $(hostname)"
echo "CPUs:           ${SLURM_CPUS_PER_TASK}"
echo "Sweep:          ${SWEEP}"
echo "Warmup:         ${WARMUP}"
echo "Repeats:        ${REPEATS}"
echo "Skip prewarm:   ${SKIP_PREWARM}"
echo "Pod5 cache:     ${POD5_CACHE}"
echo "Convert threads:${CONVERT_THREADS}"
echo "Datasets:"
for entry in "${DATASETS[@]}"; do
    echo "  - ${entry}"
done
echo "============================================"

POD5_CLI=(python3 -m pod5.tools.main)
"${POD5_CLI[@]}" --help >/dev/null 2>&1 || { echo "ERROR: 'python3 -m pod5.tools.main' failed — activate env in this script"; exit 1; }

IFS=',' read -ra WORKERS <<< "${SWEEP}"

for entry in "${DATASETS[@]}"; do
    LABEL="${entry%%:*}"
    FAST5_DIR="${entry#*:}"
    POD5_FILE="${POD5_CACHE}/${LABEL}.pod5"

    echo ""
    echo "############################################"
    echo "# Dataset: ${LABEL}"
    echo "# fast5:   ${FAST5_DIR}"
    echo "# pod5:    ${POD5_FILE}"
    echo "############################################"

    if [ ! -d "${FAST5_DIR}" ] && [ ! -f "${FAST5_DIR}" ]; then
        echo "WARNING: fast5 path ${FAST5_DIR} not found — skipping"
        continue
    fi

    # ── Convert fast5 -> pod5 (skip if cached) ────────────────────────
    if [ -f "${POD5_FILE}" ]; then
        echo "Using cached pod5: ${POD5_FILE} ($(du -h "${POD5_FILE}" | cut -f1))"
    else
        echo "Converting fast5 -> pod5 (${CONVERT_THREADS} threads)..."
        t0=$(date +%s)
        if ! "${POD5_CLI[@]}" convert fast5 \
                --output "${POD5_FILE}" \
                --threads "${CONVERT_THREADS}" \
                --recursive \
                "${FAST5_DIR}"; then
            echo "ERROR: pod5 convert failed for ${LABEL}"
            rm -f "${POD5_FILE}"
            continue
        fi
        t1=$(date +%s)
        echo "Conversion took $((t1 - t0))s, output $(du -h "${POD5_FILE}" | cut -f1)"
    fi

    # ── Run decompression benchmark sweep ─────────────────────────────
    for W in "${WORKERS[@]}"; do
        CMD="python3 ${SCRIPT_DIR}/bench_pod5_decompress.py ${POD5_FILE}"
        CMD+=" --workers ${W}"
        CMD+=" --warmup ${WARMUP}"
        CMD+=" --repeats ${REPEATS}"
        if [ "${SKIP_PREWARM}" -eq 1 ]; then
            CMD+=" --skip-prewarm"
        fi

        echo ""
        echo "============================================"
        echo ">>> ${LABEL}  workers=${W}"
        echo "============================================"
        echo "Running: ${CMD}"
        echo ""
        ${CMD}
    done
done

echo ""
echo "Finished: $(date -Iseconds)"