#!/usr/bin/env bash
#SBATCH -J soulx_vad_only
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -G 1
#SBATCH -o soulx_vad_only_%J.out
#SBATCH -e soulx_vad_only_%J.err

# Create a new formal VAD-only run for an arbitrary checkpoint.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" \
        && -f "${SLURM_SUBMIT_DIR}/SoulX-Duplug-Eval/eval_config.yaml" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT="$(cd "$SOURCE_DIR/.." && pwd)"
  fi
fi
EVAL_ROOT="$PROJECT_ROOT/SoulX-Duplug-Eval"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CONFIG="${VAD_CONFIG:-$EVAL_ROOT/eval_config.yaml}"
RUN_ID="${VAD_RUN_ID:-}"
CHECKPOINT="${VAD_CHECKPOINT:-${CHECKPOINT:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: sbatch SoulX-Duplug-Eval/run_vad.sh --run-id <new-run-id> [--checkpoint <path>] [--config <eval-config>]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ID" ]] || RUN_ID="vad-${SLURM_JOB_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}"
[[ "$CONFIG" = /* ]] || CONFIG="$PROJECT_ROOT/$CONFIG"

ARGS=(--config "$CONFIG" --run-id "$RUN_ID")
if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

echo "job_status=started run_id=$RUN_ID"
echo "protocol=formal_easy_turn_last_terminal_v1"
echo "checkpoint=${CHECKPOINT:-config_default}"
echo "result_dir=$EVAL_ROOT/output/$RUN_ID"
"$PYTHON_BIN" "$EVAL_ROOT/infer/run_vad.py" "${ARGS[@]}"
echo "job_status=completed run_id=$RUN_ID"
