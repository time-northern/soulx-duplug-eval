#!/usr/bin/env bash
#SBATCH -J soulx_vad_oracle
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -o silero_vad_oracle_%j.out
#SBATCH -e silero_vad_oracle_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/yuhao56/SoulX-Duplug}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/share/home/yuhao56/miniconda3/envs/soulx-vad-oracle}"
WORKERS="${VAD_WORKERS:-${SLURM_CPUS_PER_TASK:-12}}"

cd "$PROJECT_ROOT"
source "$(dirname "$(dirname "$CONDA_ENV_DIR")")/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python -u scripts/generate_silero_vad_endpoints.py \
  --project-root "$PROJECT_ROOT" \
  --workers "$WORKERS" \
  --languages en zh \
  --labels complete \
  --threshold 0.5 \
  --min-speech-duration-ms 50 \
  --min-silence-duration-ms 100 \
  --speech-pad-ms 0 \
  --overwrite
