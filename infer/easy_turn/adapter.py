"""Adapt the Easy-Turn runner to the evaluation framework row schema."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from .dataset import EXPECTED_COUNTS


LANGUAGES = ("en", "zh")
LABELS = ("complete", "incomplete")


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def runtime_config(project_root: Path, checkpoint: Path, language: str) -> dict:
    return {
        "model_config": {
            "task": "state_prediction",
            "glm_tokenizer_path": str(
                project_root / "pretrained_models/glm-4-voice-tokenizer"
            ),
            "model_name": str(
                project_root / "pretrained_models/Qwen3-0.6B-expand_vocab_v2"
            ),
            "init_ckpt_path_lora": str(checkpoint),
            "enable_lora": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "enable_cascade_asr": True,
            "llm_dim": 1024,
        },
        "infer_config": {
            "seed": 42,
            "device": "cuda",
            "precision": "bf16",
            "max_wait_num": 5,
            "max_mistake_num": 5,
            "single_round": False,
            "input": {
                "chunk_size": 2560,
                "audio_back_size": 15360,
                "audio_ahead_size": 640,
                "sample_rate": 16000,
                "chunk_token_len_small": 2,
            },
            "asr": {
                "model_name": "sensevoice" if language == "en" else "paraformer",
                "language": language,
            },
        },
    }


def runner_command(
    runner: Path,
    run_id: str,
    language: str,
    label: str,
    selected_count: int,
    dataset_root: Path,
    official_root: Path,
    config_path: Path,
    asr_model_dir: Path,
    work_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--language",
        language,
        "--label",
        label,
        "--dataset-root",
        str(dataset_root),
        "--official-root",
        str(official_root),
        "--config",
        str(config_path),
        "--asr-model-dir",
        str(asr_model_dir),
        "--output",
        str(work_root / "results" / f"{language}_{label}.json"),
        "--run-id",
        f"{run_id}-{language}_{label}",
    ]
    expected_count = EXPECTED_COUNTS[(language, label)]
    if selected_count < expected_count:
        command.extend(["--diagnostic-limit", str(selected_count)])
    return command


def framework_row(record: Mapping[str, Any], language: str, label: str) -> dict:
    state_events = []
    for index, event in enumerate(record["trace"]):
        state = str(event["state"])
        raw_state = {"speak": "complete", "wait": "incomplete"}.get(
            state, state
        )
        state_events.append(
            {
                "chunk_index": index,
                "chunk_start_sec": event["timestamp"][0],
                "chunk_end_sec": event["timestamp"][1],
                "chunk_center_sec": sum(event["timestamp"]) / 2,
                "process_time_sec": 0.0,
                "queue_delay_sec": 0.0,
                "current_time_sec": event["timestamp"][1],
                "raw_state": raw_state,
                "public_state": state,
                "upstream_state": state,
            }
        )
    return {
        "schema_version": 2,
        "sample_id": record["sample_id"],
        "scenario": "easy_turn",
        "language": language,
        "label": label,
        "audio": {
            "path": record["wav_path"],
            "normalized_duration_sec": record["audio_duration_seconds"],
            "vad_post_silence_sec": 2.0,
        },
        "sample_order": record.get("sample_order"),
        "event": None,
        "state_events": state_events,
    }


def validate_easy_turn_runner(config: Mapping[str, Any]) -> list[dict]:
    """Return framework-style preflight errors for the Easy-Turn runner."""
    errors = []
    runner_config = config.get("scenario_runners", {}).get("easy_turn")
    if not isinstance(runner_config, Mapping):
        return [
            {
                "code": "runner_config_missing",
                "message": "Missing scenario_runners.easy_turn configuration",
                "scenario": "easy_turn",
            }
        ]
    if runner_config.get("implementation") != "easy_turn":
        return [
            {
                "code": "runner_config_invalid",
                "message": "Easy-Turn runner implementation must be easy_turn",
                "scenario": "easy_turn",
            }
        ]
    project_root = Path(config["_project_root"])
    required_paths = []
    try:
        runner_root = resolve_path(project_root, str(runner_config["runner_root"]))
        official_root = resolve_path(project_root, str(runner_config["official_root"]))
        required_paths.extend(
            [
                ("runner", runner_root / "runner.py"),
                ("dataset adapter", runner_root / "dataset.py"),
                ("official inference", official_root / "scripts/duplex_inference.py"),
            ]
        )
        required_paths.extend(
            (
                f"English {label} order manifest",
                runner_root / "orders" / f"en_{label}.txt",
            )
            for label in LABELS
        )
        for language in LANGUAGES:
            required_paths.append(
                (
                    f"{language} ASR model",
                    resolve_path(
                        project_root, str(runner_config["asr_model_dirs"][language])
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        return [
            {
                "code": "runner_config_invalid",
                "message": str(exc),
                "scenario": "easy_turn",
            }
        ]
    for description, path in required_paths:
        if not path.exists():
            errors.append(
                {
                    "code": "runner_asset_missing",
                    "message": f"Easy-Turn {description} is missing: {path}",
                    "scenario": "easy_turn",
                    "path": str(path),
                }
            )
    return errors


def run_easy_turn_runner(
    config: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
    sample_counts: Mapping[tuple[str, str], int],
) -> list[dict]:
    """Run Easy-Turn through temporary files and return framework rows."""
    with tempfile.TemporaryDirectory(prefix=f"soulx-easy-turn-{run_id}-") as root:
        return _run_easy_turn_runner(config, Path(root), run_id, sample_counts)


def _run_easy_turn_runner(
    config: Mapping[str, Any],
    work_root: Path,
    run_id: str,
    sample_counts: Mapping[tuple[str, str], int],
) -> list[dict]:
    runner_config = config.get("scenario_runners", {}).get("easy_turn")
    if not isinstance(runner_config, Mapping):
        raise ValueError("Missing scenario_runners.easy_turn configuration")

    project_root = Path(config["_project_root"])
    runner_root = resolve_path(project_root, str(runner_config["runner_root"]))
    official_root = resolve_path(project_root, str(runner_config["official_root"]))
    runner = runner_root / "runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Easy-Turn runner is missing: {runner}")

    checkpoint = resolve_path(project_root, str(config["model"]["checkpoint"]))
    configs_dir = work_root / "configs"
    configs_dir.mkdir(parents=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runner_root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    for language in LANGUAGES:
        selected_labels = [
            label for label in LABELS if sample_counts.get((language, label), 0) > 0
        ]
        if not selected_labels:
            continue
        dataset_root = resolve_path(
            project_root, str(config["datasets"]["easy_turn"][language])
        )
        asr_model_dir = resolve_path(
            project_root, str(runner_config["asr_model_dirs"][language])
        )
        if not asr_model_dir.is_dir():
            raise FileNotFoundError(
                f"Easy-Turn {language} ASR model directory is missing: "
                f"{asr_model_dir}"
            )
        candidate_config = configs_dir / f"{language}.yaml"
        candidate_config.write_text(
            yaml.safe_dump(
                runtime_config(project_root, checkpoint, language), sort_keys=False
            ),
            encoding="utf-8",
        )
        for label in selected_labels:
            subprocess.run(
                runner_command(
                    runner,
                    run_id,
                    language,
                    label,
                    sample_counts[(language, label)],
                    dataset_root,
                    official_root,
                    candidate_config,
                    asr_model_dir,
                    work_root,
                ),
                cwd=runner_root,
                env=environment,
                check=True,
            )

    rows = []
    for language in LANGUAGES:
        for label in LABELS:
            selected_count = sample_counts.get((language, label), 0)
            if selected_count == 0:
                continue
            result_path = work_root / "results" / f"{language}_{label}.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                raise RuntimeError(f"Incomplete Easy-Turn result: {result_path}")
            records = payload.get("records")
            if not isinstance(records, list) or len(records) != selected_count:
                raise RuntimeError(
                    f"Unexpected Easy-Turn result count for {language}/{label}: "
                    f"{len(records) if isinstance(records, list) else 'invalid'} "
                    f"!= {selected_count}"
                )
            rows.extend(framework_row(record, language, label) for record in records)
    return sorted(rows, key=lambda row: row["sample_id"])
