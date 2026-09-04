"""Rerun formal VAD inference and metrics through the shared Easy-Turn path.

Only Easy-Turn inference is executed. All report files are regenerated because
the run-level reports share the same inference manifest and evidence set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_eval_config, replace_scenario_inference


EVALUATION_SCRIPTS = (
    "eval_interruption.py",
    "eval_rejection.py",
    "eval_vad.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace Easy-Turn evidence and regenerate evaluation reports."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "eval_config.yaml"),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--run-dir")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_eval_config(config_path)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = (Path(config["_output_root"]) / args.run_id).resolve()
    replace_scenario_inference(
        config_path,
        "easy_turn",
        run_id=args.run_id,
        run_dir_value=args.run_dir,
    )

    eval_root = Path(__file__).resolve().parents[1]
    for name in EVALUATION_SCRIPTS:
        subprocess.run(
            [
                sys.executable,
                str(eval_root / "eval" / name),
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
            ],
            check=True,
        )
    print(f"VAD replacement complete: run={run_dir.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
