"""Create a formal VAD-only run using the shared Easy-Turn protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import default_run_id, run_inference


def main() -> int:
    eval_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run formal bilingual Easy-Turn VAD inference and evaluation."
    )
    parser.add_argument("--config", default=str(eval_root / "eval_config.yaml"))
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument(
        "--checkpoint",
        help="Override model.checkpoint from the evaluation config.",
    )
    args = parser.parse_args()

    run_dir = run_inference(
        config_path=args.config,
        run_id=args.run_id,
        scenarios=("easy_turn",),
        checkpoint=args.checkpoint,
        formal_scenario_evaluation=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(eval_root / "eval" / "eval_vad.py"),
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--run-dir",
            str(run_dir.resolve()),
        ],
        check=True,
    )
    print(f"Formal VAD-only evaluation complete: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
