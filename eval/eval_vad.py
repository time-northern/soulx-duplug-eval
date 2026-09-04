"""Evaluate Easy-Turn VAD predictions from saved upstream state evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    format_rate,
    load_complete_manifest,
    load_eval_config,
    load_inference_records,
    macro_rate_summary,
    resolve_run_dir,
    strict_target,
    update_manifest_evaluation,
)


PRIMARY_RULE = "last-terminal-v1"
TERMINAL_TO_LABEL = {"speak": "complete", "wait": "incomplete"}
LEGACY_RAW_TO_TERMINAL = {"complete": "speak", "incomplete": "wait"}


def terminal_state(event: Mapping[str, Any]) -> str | None:
    """Read a terminal state from current or historical inference rows."""
    upstream_state = str(event.get("upstream_state", ""))
    if upstream_state in TERMINAL_TO_LABEL:
        return upstream_state
    public_state = str(event.get("public_state", ""))
    if public_state in TERMINAL_TO_LABEL:
        return public_state
    raw_state = str(event.get("raw_state", ""))
    if raw_state in TERMINAL_TO_LABEL:
        return raw_state
    return LEGACY_RAW_TO_TERMINAL.get(raw_state)


def classify_state_events(
    state_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Map the last upstream speak/wait state to an Easy-Turn label."""
    terminals = [
        (event, state)
        for event in state_events
        if (state := terminal_state(event)) is not None
    ]
    selected = terminals[-1] if terminals else None
    return {
        "rule": PRIMARY_RULE,
        "prediction": (
            TERMINAL_TO_LABEL[selected[1]]
            if selected is not None
            else "none"
        ),
        "selected_terminal": (
            {**dict(selected[0]), "resolved_terminal_state": selected[1]}
            if selected is not None
            else None
        ),
        "terminal_count": len(terminals),
    }


def _rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "rate": passed / total if total else None,
    }


def render_vad_report(metrics: Mapping[str, Any]) -> str:
    """Render the full Easy-Turn result table and aggregation evidence."""
    lines = [
        "# Easy-Turn VAD evaluation",
        "",
        f"Run ID: `{metrics['run_id']}`",
        "",
        "## Terminal-state protocol",
        "",
        "Each Easy-Turn clip is inferred with the configured post-audio silence. "
        "Prediction is the label represented by the final upstream `speak` or "
        "`wait` state; no terminal state is reported as `none`. Public-state "
        "acceptance and endpoint-oracle latency are not used for this metric.",
        "",
        "## Per-category results",
        "",
        "| Category | Correct | Incorrect | Total | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, summary in metrics["per_category"].items():
        lines.append(
            f"| {category} | {summary['passed']} | {summary['failed']} | "
            f"{summary['total']} | {format_rate(summary['rate'])} |"
        )

    lines.extend(["", "## Primary language-macro metrics", ""])
    for name in ("complete_accuracy", "incomplete_accuracy"):
        summary = metrics["aggregates"][name]
        lines.append(
            f"- `{name}`: {format_rate(summary['rate'])} "
            f"({summary['aggregation']}; evidence: "
            f"{summary['passed']}/{summary['total']})"
        )

    lines.extend(["", "## Sample-weighted reference metrics", ""])
    for name, summary in metrics["sample_weighted_aggregates"].items():
        lines.append(
            f"- `{name}`: {summary['passed']}/{summary['total']} "
            f"({format_rate(summary['rate'])})"
        )

    lines.extend(
        [
            "",
            "## Technical-target checks",
            "",
            "| Target | Value | Condition | Met |",
            "| --- | ---: | ---: | :---: |",
        ]
    )
    for name, target in metrics["technical_targets"].items():
        condition = f"{target['operator']} {format_rate(target['threshold'])}"
        lines.append(
            f"| {name} | {format_rate(target['value'])} | {condition} | "
            f"{'YES' if target['met'] else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def _read_rows(
    run_dir: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for record in load_inference_records(
        run_dir, manifest, ("easy_turn",)
    ):
        readout = classify_state_events(record["state_events"])
        rows.append(
            {
                "sample_id": record["sample_id"],
                "scenario": "easy_turn",
                "language": record["language"],
                "label": record["label"],
                "prediction": readout["prediction"],
                "passed": readout["prediction"] == record["label"],
                "decision_rule": PRIMARY_RULE,
                "terminal_count": readout["terminal_count"],
                "selected_terminal": readout["selected_terminal"],
            }
        )
    return rows


def evaluate_vad_run(
    config_path: str | Path,
    run_id: str | None = None,
    run_dir_value: str | Path | None = None,
) -> dict[str, Any]:
    config = load_eval_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, run_dir=run_dir_value)
    manifest = load_complete_manifest(run_dir)
    rows = _read_rows(run_dir, manifest)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"easy_turn/{row['language']}/{row['label']}"].append(row)
    per_category = {
        key: _rate(value) for key, value in sorted(groups.items())
    }

    language_summary = {}
    for language in ("en", "zh"):
        complete = per_category[f"easy_turn/{language}/complete"]
        incomplete = per_category[f"easy_turn/{language}/incomplete"]
        language_summary[language] = {
            "complete_accuracy": complete,
            "incomplete_accuracy": incomplete,
            "avg_accuracy": (complete["rate"] + incomplete["rate"]) / 2,
        }

    complete_all = _rate([row for row in rows if row["label"] == "complete"])
    incomplete_all = _rate(
        [row for row in rows if row["label"] == "incomplete"]
    )
    complete_macro = macro_rate_summary(
        {
            language: value["complete_accuracy"]
            for language, value in language_summary.items()
        },
        "macro_average_by_language",
    )
    incomplete_macro = macro_rate_summary(
        {
            language: value["incomplete_accuracy"]
            for language, value in language_summary.items()
        },
        "macro_average_by_language",
    )
    thresholds = config["evaluation"]["thresholds"]
    metrics = {
        "schema_version": 6,
        "run_id": manifest["run_id"],
        "scene": "vad_easy_turn",
        "complete": True,
        "protocol": {
            "primary_rule": PRIMARY_RULE,
            "decision_source": (
                "official duplex_predict_160_cascade_asr state_events; "
                "speak=>complete, wait=>incomplete"
            ),
            "post_silence_sec": 2.0,
            "far_field_filter": False,
        },
        "per_category": per_category,
        "language_summary": language_summary,
        "aggregates": {
            "complete_accuracy": complete_macro,
            "incomplete_accuracy": incomplete_macro,
            "avg_accuracy": (complete_macro["rate"] + incomplete_macro["rate"])
            / 2,
        },
        "sample_weighted_aggregates": {
            "complete_accuracy": complete_all,
            "incomplete_accuracy": incomplete_all,
        },
        "technical_targets": {
            "incomplete_accuracy": strict_target(
                incomplete_macro["rate"],
                ">",
                float(thresholds["vad_pause_accuracy_gt"]),
            ),
            "complete_accuracy": strict_target(
                complete_macro["rate"],
                ">",
                float(thresholds["vad_normal_endpoint_accuracy_gt"]),
            ),
        },
    }

    output_dir = run_dir / "evaluation" / "vad"
    metrics_path = output_dir / "metrics.json"
    atomic_write_jsonl(output_dir / "samples.jsonl", rows)
    atomic_write_json(metrics_path, metrics)
    atomic_write_text(output_dir / "report.md", render_vad_report(metrics))
    update_manifest_evaluation(run_dir, "vad", metrics_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Easy-Turn VAD metrics from saved state evidence."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "eval_config.yaml"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if bool(args.run_id) == bool(args.run_dir):
        raise ValueError("Provide exactly one of --run-id and --run-dir")
    result = evaluate_vad_run(args.config, args.run_id, args.run_dir)
    print(json.dumps(result["aggregates"], ensure_ascii=False))
