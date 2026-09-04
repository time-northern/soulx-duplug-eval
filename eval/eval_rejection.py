from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from common import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    evaluate_records,
    interruption_latency_summary,
    load_complete_manifest,
    load_eval_config,
    load_inference_records,
    per_category_summary,
    rate_summary,
    render_report,
    resolve_run_dir,
    scenario_language_macro_rate_summary,
    strict_target,
    update_manifest_evaluation,
)


def evaluate_rejection_run(
    config_path: str | Path,
    run_id: Optional[str] = None,
    run_dir_value: Optional[str | Path] = None,
) -> Dict[str, Any]:
    config = load_eval_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, run_dir=run_dir_value)
    manifest = load_complete_manifest(run_dir)
    records = load_inference_records(
        run_dir,
        manifest,
        (
            "interruption",
            "backchannel",
            "background_noise",
            "easy_turn",
        ),
    )
    evaluation_config = config["evaluation"]
    latency_threshold = float(
        evaluation_config["interruption_latency_threshold_sec"]
    )
    rows = evaluate_records(records, latency_threshold)
    effective_rows = [
        row
        for row in rows
        if row["scenario"] == "interruption"
        or (row["scenario"] == "easy_turn" and row["label"] == "complete")
    ]
    invalid_rows = [
        row
        for row in rows
        if row["scenario"]
        in {
            "backchannel",
            "background_noise",
        }
    ]
    effective = scenario_language_macro_rate_summary(
        {
            "interruption": [row for row in effective_rows if row["scenario"] == "interruption"],
            "easy_turn_complete": [row for row in effective_rows if row["scenario"] == "easy_turn"],
        }
    )
    invalid = scenario_language_macro_rate_summary(
        {
            scenario: [row for row in invalid_rows if row["scenario"] == scenario]
            for scenario in ("backchannel", "background_noise")
        }
    )
    sample_weighted_effective = rate_summary(effective_rows)
    sample_weighted_invalid = rate_summary(invalid_rows)
    effective_false_rejection = (
        1.0 - effective["rate"] if effective["rate"] is not None else None
    )
    thresholds = evaluation_config["thresholds"]
    metrics: Dict[str, Any] = {
        "schema_version": 3,
        "run_id": manifest["run_id"],
        "scene": "rejection",
        "complete": True,
        "decision_source": "SoulX raw/public state only",
        "interruption_latency_threshold_sec": latency_threshold,
        "protocol": {
            "interruption_detection_pass_rule": (
                "any public nonidle whose chunk center is in the annotated "
                "speech window [start,end)"
            ),
            "interruption_latency_threshold_role": "diagnostic_only",
        },
        "per_category": per_category_summary(rows),
        "aggregates": {
            "effective_intent_acceptance_rate": effective,
            "invalid_intent_rejection_rate": invalid,
        },
        "sample_weighted_aggregates": {
            "effective_intent_acceptance_rate": sample_weighted_effective,
            "invalid_intent_rejection_rate": sample_weighted_invalid,
        },
        "latency": interruption_latency_summary(rows),
        "technical_targets": {
            "effective_intent_false_rejection_rate": strict_target(
                effective_false_rejection,
                "<",
                float(thresholds["rejection_effective_false_rejection_lt"]),
            ),
            "invalid_intent_rejection_rate": strict_target(
                invalid["rate"],
                ">",
                float(thresholds["rejection_invalid_rejection_gt"]),
            ),
        },
        "not_evaluated": list(config.get("not_evaluated") or []),
    }
    output_dir = run_dir / "evaluation" / "rejection"
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    atomic_write_jsonl(samples_path, rows)
    atomic_write_json(metrics_path, metrics)
    atomic_write_text(report_path, render_report("Rejection scene evaluation", metrics))
    update_manifest_evaluation(run_dir, "rejection", metrics_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "eval_config.yaml"
    parser = argparse.ArgumentParser(
        description="Evaluate rejection-scene metrics from saved state evidence."
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if bool(args.run_id) == bool(args.run_dir):
        raise ValueError("Provide exactly one of --run-id and --run-dir")
    result = evaluate_rejection_run(args.config, args.run_id, args.run_dir)
    print(
        f"Rejection evaluation complete: run={result['run_id']} "
        f"effective={result['aggregates']['effective_intent_acceptance_rate']['rate']} "
        f"invalid={result['aggregates']['invalid_intent_rejection_rate']['rate']}",
        flush=True,
    )
