#!/usr/bin/env bash
#SBATCH -J soulx_vad_refresh
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -G 1
#SBATCH -o soulx_vad_refresh_%J.out
#SBATCH -e soulx_vad_refresh_%J.err

# Submit this script with sbatch from the repository root. It reruns only the
# Easy-Turn/VAD inference through the formal shared scenario runner. The three
# reports are regenerated because they share the same run manifest/evidence.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/SoulX-Duplug-Eval/eval_config.yaml" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT="$(cd "$SOURCE_DIR/.." && pwd)"
  fi
fi
EVAL_ROOT="$PROJECT_ROOT/SoulX-Duplug-Eval"
if [[ ! -f "$EVAL_ROOT/infer/rerun_vad.py" ]]; then
  echo "Evaluation framework not found: $EVAL_ROOT" >&2
  echo "Submit from the repository root or export PROJECT_ROOT explicitly." >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$EVAL_ROOT/eval_config.yaml"
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: sbatch SoulX-Duplug-Eval/rerun_vad.sh --run-id <completed-run-id> [--config <eval-config>]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }
[[ "$CONFIG" = /* ]] || CONFIG="$PROJECT_ROOT/$CONFIG"

echo "job_status=started run_id=$RUN_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "project_root=$PROJECT_ROOT"
echo "protocol=formal_easy_turn_last_terminal_v1"
"$PYTHON_BIN" "$EVAL_ROOT/infer/rerun_vad.py" --config "$CONFIG" --run-id "$RUN_ID"
echo "job_status=completed run_id=$RUN_ID"
