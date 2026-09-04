#!/usr/bin/env bash
#SBATCH -J soulx_state_eval
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -G 1
#SBATCH -o soulx_state_eval_%J.out
#SBATCH -e soulx_state_eval_%J.out

set -euo pipefail

SOURCE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" \
        && -f "${SLURM_SUBMIT_DIR}/SoulX-Duplug-Eval/eval_config.yaml" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT="$(cd "$SOURCE_SCRIPT_DIR/.." && pwd)"
  fi
fi
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/SoulX-Duplug-Eval}"

if [[ ! -f "$EVAL_ROOT/eval_config.yaml" ]]; then
  echo "Evaluation framework not found: $EVAL_ROOT" >&2
  echo "Submit from the SoulX-Duplug repository root or export PROJECT_ROOT explicitly." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
CONFIG="$EVAL_ROOT/eval_config.yaml"
RUN_ID=""
LIMIT_PER_DATASET=0
CHECKPOINT=""

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "$exit_code" -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] job_status=completed run_id=${RUN_ID:-unknown}"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] job_status=failed exit_code=$exit_code run_id=${RUN_ID:-unknown}" >&2
  fi
  exit "$exit_code"
}
trap on_exit EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --run-id=*)
      RUN_ID="${1#*=}"
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      shift
      ;;
    --limit-per-dataset)
      LIMIT_PER_DATASET="$2"
      shift 2
      ;;
    --limit-per-dataset=*)
      LIMIT_PER_DATASET="${1#*=}"
      shift
      ;;
    -h|--help)
      "$PYTHON_BIN" "$EVAL_ROOT/infer/run_all.py" --help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_ROOT/$CONFIG"
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] job_status=started"
echo "run_id=$RUN_ID"
echo "slurm_job_id=${SLURM_JOB_ID:-not_in_slurm}"
echo "slurm_job_name=${SLURM_JOB_NAME:-soulx_state_eval}"
echo "slurm_cpus=${SLURM_CPUS_PER_TASK:-12}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "python_bin=$PYTHON_BIN"
echo "project_root=$PROJECT_ROOT"
echo "eval_root=$EVAL_ROOT"
echo "config=$CONFIG"
echo "checkpoint_override=${CHECKPOINT:-config_default}"
echo "result_dir=$EVAL_ROOT/output/$RUN_ID"

INFER_ARGS=(
  --config "$CONFIG"
  --run-id "$RUN_ID"
  --limit-per-dataset "$LIMIT_PER_DATASET"
)
if [[ -n "$CHECKPOINT" ]]; then
  INFER_ARGS+=(--checkpoint "$CHECKPOINT")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=inference status=started"
"$PYTHON_BIN" "$EVAL_ROOT/infer/run_all.py" "${INFER_ARGS[@]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=inference status=completed"

if [[ "$LIMIT_PER_DATASET" -ne 0 ]]; then
  echo "Subset inference finished. Formal evaluation is skipped because the manifest is incomplete."
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=interruption_evaluation status=started"
"$PYTHON_BIN" "$EVAL_ROOT/eval/eval_interruption.py" \
  --config "$CONFIG" \
  --run-id "$RUN_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=interruption_evaluation status=completed"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=rejection_evaluation status=started"
"$PYTHON_BIN" "$EVAL_ROOT/eval/eval_rejection.py" \
  --config "$CONFIG" \
  --run-id "$RUN_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=rejection_evaluation status=completed"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=vad_evaluation status=started"
"$PYTHON_BIN" "$EVAL_ROOT/eval/eval_vad.py" \
  --config "$CONFIG" \
  --run-id "$RUN_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=vad_evaluation status=completed"

echo "SoulX state-only evaluation complete: $RUN_ID"
