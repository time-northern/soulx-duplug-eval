"""Recover Easy-Turn inference rows from completed runner artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from common import atomic_write_jsonl, load_eval_config
from easy_turn.adapter import framework_row


SPLITS = (
    ("en", "complete"),
    ("en", "incomplete"),
    ("zh", "complete"),
    ("zh", "incomplete"),
)
EVALUATION_SCRIPTS = (
    "eval_interruption.py",
    "eval_rejection.py",
    "eval_vad.py",
)


def _expected_counts(manifest: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    sample_counts = manifest.get("sample_counts")
    if not isinstance(sample_counts, Mapping):
        raise ValueError("run manifest has no sample_counts mapping")
    expected = {}
    for language, label in SPLITS:
        key = f"easy_turn/{language}/{label}"
        value = sample_counts.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"run manifest has invalid count for {key}: {value}")
        expected[(language, label)] = value
    return expected


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid runner result JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"runner result root must be an object: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"runner result has no records list: {path}")
    return payload


def recover_easy_turn_inference(
    run_dir: Path, artifact_root: Path | None = None
) -> Path:
    """Validate four split results and atomically replace easy_turn.jsonl."""
    run_dir = run_dir.expanduser().resolve(strict=True)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"run manifest root must be an object: {manifest_path}")
    expected_counts = _expected_counts(manifest)

    root = (
        artifact_root.expanduser().resolve(strict=True)
        if artifact_root is not None
        else (run_dir / "artifacts" / "easy_turn").resolve(strict=True)
    )
    results_root = root / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(f"missing Easy-Turn results directory: {results_root}")

    rows = []
    source_statuses = {}
    for language, label in SPLITS:
        result_path = results_root / f"{language}_{label}.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"missing split result: {result_path}")
        payload = _load_payload(result_path)
        records = payload["records"]
        expected = expected_counts[(language, label)]
        if len(records) != expected:
            raise RuntimeError(
                f"unexpected record count for {language}/{label}: "
                f"{len(records)} != {expected}"
            )
        source_statuses[f"{language}/{label}"] = payload.get("status")
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"{result_path}: record {index} must be an object"
                )
            missing = [
                field
                for field in (
                    "sample_id",
                    "wav_path",
                    "audio_duration_seconds",
                    "trace",
                )
                if field not in record
            ]
            if missing:
                raise ValueError(
                    f"{result_path}: record {index} is missing {missing}"
                )
            rows.append(framework_row(record, language, label))

    duplicate_ids = sorted(
        sample_id
        for sample_id, count in Counter(row["sample_id"] for row in rows).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate recovered sample IDs: {duplicate_ids[:10]}")
    expected_total = sum(expected_counts.values())
    if len(rows) != expected_total:
        raise RuntimeError(f"recovered row count {len(rows)} != {expected_total}")

    target = run_dir / "inference" / "easy_turn.jsonl"
    atomic_write_jsonl(target, sorted(rows, key=lambda row: row["sample_id"]))
    print(
        json.dumps(
            {
                "recovered": len(rows),
                "source": str(results_root),
                "source_statuses": source_statuses,
                "target": str(target),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return target


def run_evaluations(config_path: Path, run_dir: Path) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover inference/easy_turn.jsonl from artifacts/easy_turn/results "
            "and regenerate evaluation reports."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "eval_config.yaml"),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--run-dir")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Override the default <run-dir>/artifacts/easy_turn source.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only recover inference/easy_turn.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve(strict=True)
    config = load_eval_config(config_path)
    run_dir = (
        Path(args.run_dir).expanduser().resolve(strict=True)
        if args.run_dir
        else (Path(config["_output_root"]) / args.run_id).resolve(strict=True)
    )
    recover_easy_turn_inference(run_dir, args.artifact_root)
    if not args.skip_evaluation:
        run_evaluations(config_path, run_dir)
    print(f"Easy-Turn artifact recovery complete: run={run_dir.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
