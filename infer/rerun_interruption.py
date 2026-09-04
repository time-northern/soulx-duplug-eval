"""Replace only a completed run's interruption inference evidence.

The existing inference evidence for all other scenarios is preserved.  The
replacement uses the run's saved runtime configs (and therefore its original
checkpoint and deployment settings), then invalidates the old reports so the
shell entry point can regenerate them from the updated evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    atomic_write_json,
    atomic_write_jsonl,
    discover_all_samples,
    infer_sample,
    load_eval_config,
    utc_now,
    warm_up_model,
)


SCENARIO = "interruption"


def resolve_target_run_dir(
    config: Dict[str, Any],
    run_id: Optional[str],
    run_dir_value: Optional[str | Path],
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


def load_completed_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("error_count") != 0:
        raise RuntimeError("Interruption replacement requires a complete zero-error run")
    selected = manifest.get("selected_scenarios", [])
    inference_files = manifest.get("inference_files", {})
    sample_counts = manifest.get("sample_counts", {})
    if not (
        SCENARIO in selected
        or SCENARIO in inference_files
        or any(key.startswith(f"{SCENARIO}/") for key in sample_counts)
    ):
        raise RuntimeError("Original run did not select interruption")
    return manifest


def rerun_interruption(
    config_path: str | Path,
    run_id: Optional[str] = None,
    run_dir_value: Optional[str | Path] = None,
) -> Path:
    config = load_eval_config(config_path)
    run_dir = resolve_target_run_dir(config, run_id, run_dir_value)
    manifest = load_completed_manifest(run_dir)

    stored_checkpoint = (manifest.get("model") or {}).get("checkpoint")
    if stored_checkpoint:
        config["model"]["checkpoint"] = stored_checkpoint

    samples, errors, _ = discover_all_samples(config, (SCENARIO,))
    if errors:
        details_path = run_dir / "interruption_rerun_errors.jsonl"
        atomic_write_jsonl(details_path, errors)
        raise RuntimeError(
            f"Interruption preflight failed with {len(errors)} error(s); "
            f"see {details_path}"
        )
    if not samples:
        raise RuntimeError("Interruption discovery returned no samples")

    actual_counts = Counter(sample.count_key for sample in samples)
    expected_counts = {
        key: int(value)
        for key, value in (manifest.get("sample_counts") or {}).items()
        if key.startswith(f"{SCENARIO}/")
    }
    if dict(sorted(actual_counts.items())) != dict(sorted(expected_counts.items())):
        raise RuntimeError(
            "Interruption inventory differs from the original run: "
            f"{dict(sorted(actual_counts.items()))} != "
            f"{dict(sorted(expected_counts.items()))}"
        )

    project_root = Path(config["_project_root"])
    samples_by_language = defaultdict(list)
    for sample in samples:
        samples_by_language[sample.language].append(sample)

    rows: List[Dict[str, Any]] = []
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from service.model import TurnModel

    for language in sorted(samples_by_language):
        language_samples = samples_by_language[language]
        runtime_config = run_dir / "configs" / f"{language}_config_used.yaml"
        if not runtime_config.is_file():
            raise FileNotFoundError(f"Missing saved runtime config: {runtime_config}")
        settings = (manifest.get("stream_settings") or {}).get(language) or {}
        try:
            sample_rate = int(settings["sample_rate"])
            chunk_size = int(settings["chunk_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest has no valid {language} stream settings"
            ) from exc

        print(
            f"Loading {language} model for {len(language_samples)} interruption samples",
            flush=True,
        )
        model = TurnModel(config_path=str(runtime_config))
        try:
            warm_up_model(model, language_samples[0], chunk_size, sample_rate)
            for index, sample in enumerate(language_samples, 1):
                rows.append(
                    infer_sample(
                        model,
                        sample,
                        chunk_size,
                        sample_rate,
                        project_root,
                    )
                )
                print(
                    f"[{language}] {index}/{len(language_samples)} {sample.sample_id}",
                    flush=True,
                )
        finally:
            del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    target = run_dir / "inference" / "interruption.jsonl"
    atomic_write_jsonl(target, sorted(rows, key=lambda row: row["sample_id"]))

    error_path = run_dir / "interruption_rerun_errors.jsonl"
    if error_path.exists():
        error_path.unlink()
    manifest.pop("evaluations", None)
    manifest.update(
        {
            "status": "inference_complete",
            "updated_at": utc_now(),
            "complete": True,
            "error_count": 0,
        }
    )
    atomic_write_json(run_dir / "manifest.json", manifest)
    print(
        f"Interruption replacement complete: run={manifest['run_id']} "
        f"samples={len(rows)}",
        flush=True,
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "eval_config.yaml"
    parser = argparse.ArgumentParser(
        description="Replace one completed run's interruption inference evidence."
    )
    parser.add_argument("--config", default=str(default_config))
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--run-dir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    rerun_interruption(args.config, args.run_id, args.run_dir)
