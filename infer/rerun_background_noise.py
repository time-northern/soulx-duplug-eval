"""Replace only a completed run's background-noise inference evidence.

This is intended for protocol updates such as a duration cap or trailing silence.  It preserves
the inference evidence for every other scenario, then invalidates the old
formal reports so they can be regenerated from the mixed run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    atomic_write_json,
    atomic_write_jsonl,
    discover_sid_noise_samples,
    load_eval_config,
    resolve_path,
    warm_up_model,
    infer_sample,
)


def load_completed_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not True or manifest.get("error_count") != 0:
        raise RuntimeError("Background-noise replacement requires a complete zero-error run")
    return manifest


def resolve_target_run_dir(
    config: Dict[str, Any], run_id: Optional[str], run_dir_value: Optional[str | Path]
) -> Path:
    if run_dir_value is not None:
        run_dir = Path(run_dir_value).expanduser().resolve()
    elif run_id:
        run_dir = (Path(config["_output_root"]) / run_id).resolve()
    else:
        raise ValueError("Provide exactly one of --run-id or --run-dir")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def rerun_background_noise(
    config_path: str | Path,
    run_id: Optional[str] = None,
    run_dir_value: Optional[str | Path] = None,
) -> Path:
    config = load_eval_config(config_path)
    run_dir = resolve_target_run_dir(config, run_id, run_dir_value)
    manifest = load_completed_manifest(run_dir)
    project_root = Path(config["_project_root"])
    dataset_root_value = (config.get("datasets") or {}).get("background_noise", {}).get("neutral")
    if not dataset_root_value:
        raise ValueError("datasets.background_noise.neutral is required")
    samples, errors = discover_sid_noise_samples(
        "neutral", resolve_path(project_root, str(dataset_root_value))
    )
    if errors:
        details_path = run_dir / "background_noise_rerun_errors.jsonl"
        atomic_write_jsonl(details_path, errors)
        raise RuntimeError(
            f"Background-noise preflight failed with {len(errors)} error(s); see {details_path}"
        )
    if not samples:
        raise RuntimeError("Background-noise discovery returned no samples")

    runtime_config = run_dir / "configs" / "neutral_config_used.yaml"
    if not runtime_config.is_file():
        raise FileNotFoundError(f"Missing saved neutral runtime config: {runtime_config}")
    settings = (manifest.get("stream_settings") or {}).get("neutral") or {}
    try:
        sample_rate = int(settings["sample_rate"])
        chunk_size = int(settings["chunk_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest has no valid neutral stream settings") from exc
    max_duration_sec = float(
        (config.get("evaluation") or {}).get("background_noise_max_duration_sec", 20.0)
    )
    if max_duration_sec <= 0:
        raise ValueError("evaluation.background_noise_max_duration_sec must be positive")
    trailing_silence_sec = float(
        (config.get("evaluation") or {}).get(
            "background_noise_trailing_silence_sec", 2.0
        )
    )
    if trailing_silence_sec < 0:
        raise ValueError(
            "evaluation.background_noise_trailing_silence_sec must be non-negative"
        )

    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from service.model import TurnModel

    model = TurnModel(config_path=str(runtime_config))
    try:
        warm_up_model(
            model,
            samples[0],
            chunk_size,
            sample_rate,
            max_duration_sec=max_duration_sec,
            trailing_silence_sec=trailing_silence_sec,
        )
        rows: List[Dict[str, Any]] = []
        for index, sample in enumerate(samples, 1):
            rows.append(
                infer_sample(
                    model,
                    sample,
                    chunk_size,
                    sample_rate,
                    project_root,
                    max_duration_sec=max_duration_sec,
                    trailing_silence_sec=trailing_silence_sec,
                )
            )
            print(f"[neutral] {index}/{len(samples)} {sample.sample_id}", flush=True)
    except Exception:
        # Keep the old evidence untouched if the replacement cannot complete.
        raise
    finally:
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    inference_path = run_dir / "inference" / "background_noise.jsonl"
    atomic_write_jsonl(inference_path, sorted(rows, key=lambda row: row["sample_id"]))

    sample_counts = dict(manifest.get("sample_counts") or {})
    sample_counts["background_noise/neutral"] = len(samples)
    manifest["sample_counts"] = dict(sorted(sample_counts.items()))
    manifest["num_samples"] = sum(int(value) for value in sample_counts.values())
    manifest["input_truncation"] = {
        "background_noise": {
            "max_duration_sec": max_duration_sec,
            "policy": "use the first max_duration_sec of each clip when longer",
            "trailing_silence_sec": trailing_silence_sec,
            "trailing_silence_policy": "append silence after truncation and evaluate all resulting state outputs",
        }
    }
    manifest.pop("sampling", None)
    manifest.pop("evaluations", None)
    manifest["status"] = "inference_complete"
    manifest["complete"] = True
    manifest["error_count"] = 0
    atomic_write_json(run_dir / "manifest.json", manifest)
    print(
        f"Background-noise replacement complete: run={manifest['run_id']} "
        f"samples={len(samples)} max_duration_sec={max_duration_sec} "
        f"trailing_silence_sec={trailing_silence_sec}",
        flush=True,
    )
    return inference_path


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "eval_config.yaml"
    parser = argparse.ArgumentParser(
        description="Replace one completed run's background-noise inference evidence."
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if bool(args.run_id) == bool(args.run_dir):
        raise ValueError("Provide exactly one of --run-id and --run-dir")
    rerun_background_noise(args.config, args.run_id, args.run_dir)
