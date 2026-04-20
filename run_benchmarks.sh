#!/bin/bash
#SBATCH --job-name=vbz-bench
#SBATCH --output=vbz-bench-%j.out
#SBATCH --error=vbz-bench-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --partition=bio_part

set -euo pipefail

# dataset_label:fast5_dir pairs
DATASETS=(
    "d4_green_algae_r94:/mnt/nvme1/soysalm/d4_green_algae_r94/fast5_files/"
    "d3_yeast_r94:/mnt/nvme1/soysalm/d3_yeast_r94/fast5_files/"
)

# Extra args forwarded to the benchmark (override by passing on the command line).
BENCH_ARGS=("$@")
if [ ${#BENCH_ARGS[@]} -eq 0 ]; then
    BENCH_ARGS=(--sweep 1,2,4,8 --chunked)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for entry in "${DATASETS[@]}"; do
    label="${entry%%:*}"
    path="${entry#*:}"
    echo
    echo "############################################################"
    echo "# Dataset: $label"
    echo "# Path:    $path"
    echo "############################################################"
    python3 "./bench_signal_compression.py" "$path" "${BENCH_ARGS[@]}"
done
