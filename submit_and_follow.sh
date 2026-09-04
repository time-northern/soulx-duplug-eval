#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found; submit_and_follow.sh must run on a Slurm login node." >&2
  exit 1
fi
if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue was not found; it is required to follow job completion." >&2
  exit 1
fi

submission="$(
  sbatch \
    --parsable \
    --chdir="$PROJECT_ROOT" \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT" \
    "$SCRIPT_DIR/run_all.sh" \
    "$@"
)"
job_id="${submission%%;*}"
log_file="$PROJECT_ROOT/soulx_state_eval_${job_id}.out"

echo "submitted_job_id=$job_id"
echo "live_log=$log_file"
echo "Press Ctrl-C to stop following; the Slurm job will continue running."

while [[ ! -f "$log_file" ]]; do
  if ! squeue -h -j "$job_id" | grep -q .; then
    break
  fi
  sleep 1
done

if [[ ! -f "$log_file" ]]; then
  echo "The job left the queue before its log file appeared: $log_file" >&2
  exit 1
fi

tail -n +1 -F "$log_file" &
tail_pid=$!
cleanup_tail() {
  kill "$tail_pid" >/dev/null 2>&1 || true
  wait "$tail_pid" 2>/dev/null || true
}
on_interrupt() {
  cleanup_tail
  trap - EXIT INT TERM
  echo "Stopped following job $job_id; the Slurm job is still running." >&2
  exit 130
}
trap cleanup_tail EXIT
trap on_interrupt INT TERM

while squeue -h -j "$job_id" | grep -q .; do
  sleep 2
done

# Allow the filesystem and Slurm output writer to flush the final lines.
sleep 2
cleanup_tail
trap - EXIT INT TERM

if command -v sacct >/dev/null 2>&1; then
  state="$(sacct -n -X -j "$job_id" --format=State | awk 'NF {print $1; exit}')"
  echo "final_slurm_state=${state:-unknown}"
  if [[ -n "$state" && "$state" != COMPLETED* ]]; then
    exit 1
  fi
fi
