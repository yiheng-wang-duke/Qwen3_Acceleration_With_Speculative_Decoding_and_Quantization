#!/usr/bin/env bash
#SBATCH --job-name=acceleratellm-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm-%x-%j.out
# Full benchmark: AR baseline, acceptance + theory, paired and full MT-Bench timing.
#
#   bash scripts/run_benchmark.sh                     # machine with one idle GPU
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_benchmark.sh
#   sbatch -p <partition> scripts/run_benchmark.sh    # Slurm (run `mkdir -p outputs` first)
#
# Environment overrides: PYTHON (default: python), ACCELERATELLM_ALLOW_SHARED_GPU=1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p outputs
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "HOST=$(hostname) CPU=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv || true
"$PYTHON" -m acceleratellm doctor
"$PYTHON" -m acceleratellm check-models --config configs/experiments/speculative.yaml
"$PYTHON" -m pytest -q
"$PYTHON" scripts/benchmark.py --config configs/experiments/speculative.yaml
"$PYTHON" scripts/summarize.py --latest
