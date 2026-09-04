from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_eval_config(config_path: str | Path) -> Dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    project_value = config.get("paths", {}).get("project_root", "..")
    project_root = Path(project_value).expanduser()
    if not project_root.is_absolute():
        project_root = config_file.parent / project_root
    project_root = project_root.resolve()
    output_value = config.get("paths", {}).get(
        "output_root", "SoulX-Duplug-Eval/output"
    )
    output_root = Path(output_value).expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    config["_config_path"] = config_file
    config["_project_root"] = project_root
    config["_output_root"] = output_root.resolve()
    return config


def resolve_run_dir(
    config: Mapping[str, Any],
    run_id: Optional[str] = None,
    run_dir: Optional[str | Path] = None,
) -> Path:
    if run_dir is not None:
        resolved = Path(run_dir).expanduser().resolve()
    elif run_id:
        resolved = (Path(config["_output_root"]) / run_id).resolve()
    else:
        raise ValueError("Either --run-id or --run-dir is required")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {resolved}")
    return resolved


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_complete_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not True:
        raise RuntimeError(
            "Formal metrics require manifest.complete == true; "
            f"current status is {manifest.get('status')!r}"
        )
    if manifest.get("error_count") != 0:
        raise RuntimeError("Formal metrics require a zero-error inference manifest")
    return manifest


def count_key(row: Mapping[str, Any]) -> str:
    scenario = str(row.get("scenario", ""))
    language = str(row.get("language", ""))
    if scenario == "easy_turn":
        return f"{scenario}/{language}/{row.get('label', '')}"
    return f"{scenario}/{language}"


def _validate_state_event(event: Mapping[str, Any], sample_id: str) -> None:
    required = (
        "chunk_index",
        "chunk_start_sec",
        "chunk_end_sec",
        "chunk_center_sec",
        "process_time_sec",
        "queue_delay_sec",
        "current_time_sec",
        "raw_state",
        "public_state",
    )
    missing = [field for field in required if field not in event]
    if missing:
        raise ValueError(f"{sample_id}: state event is missing {missing}")
    for field in (
        "chunk_start_sec",
        "chunk_end_sec",
        "chunk_center_sec",
        "process_time_sec",
        "queue_delay_sec",
        "current_time_sec",
    ):
        value = float(event[field])
        if not math.isfinite(value):
            raise ValueError(f"{sample_id}: {field} must be finite")
    if float(event["chunk_end_sec"]) <= float(event["chunk_start_sec"]):
        raise ValueError(f"{sample_id}: invalid chunk interval")
    if float(event["current_time_sec"]) < float(event["chunk_end_sec"]):
        raise ValueError(f"{sample_id}: current_time precedes chunk availability")


def validate_inference_record(row: Mapping[str, Any], expected_scenario: str) -> None:
    sample_id = str(row.get("sample_id", ""))
    if not sample_id:
        raise ValueError("Inference record has no sample_id")
    if row.get("scenario") != expected_scenario:
        raise ValueError(
            f"{sample_id}: expected scenario {expected_scenario}, got {row.get('scenario')}"
        )
    if row.get("language") not in {"en", "zh", "neutral"}:
        raise ValueError(f"{sample_id}: unsupported language {row.get('language')}")
    state_events = row.get("state_events")
    if not isinstance(state_events, list) or not state_events:
        raise ValueError(f"{sample_id}: state_events must be a non-empty list")
    for event in state_events:
        if not isinstance(event, dict):
            raise ValueError(f"{sample_id}: state event must be an object")
        _validate_state_event(event, sample_id)
    if expected_scenario != "easy_turn":
        event = row.get("event")
        if not isinstance(event, dict):
            raise ValueError(f"{sample_id}: event annotation is required")
        start = float(event.get("start_sec"))
        end = float(event.get("end_sec"))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"{sample_id}: invalid event annotation")
    else:
        if row.get("label") not in {"complete", "incomplete"}:
            raise ValueError(f"{sample_id}: invalid Easy-Turn label {row.get('label')}")
        valid_upstream_states = {
            "speak",
            "wait",
            "backchannel",
            "idle",
            "nonidle",
            "unknown",
        }
        for event in state_events:
            upstream_state = event.get("upstream_state")
            if (
                upstream_state is not None
                and upstream_state not in valid_upstream_states
            ):
                raise ValueError(
                    f"{sample_id}: invalid Easy-Turn upstream_state "
                    f"{upstream_state}"
                )


def load_inference_records(
    run_dir: Path,
    manifest: Mapping[str, Any],
    scenarios: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ids: set[str] = set()
    for scenario in scenarios:
        path = run_dir / "inference" / f"{scenario}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing inference evidence: {path}")
        scenario_rows = read_jsonl(path)
        for row in scenario_rows:
            validate_inference_record(row, scenario)
            sample_id = str(row["sample_id"])
            if sample_id in ids:
                raise ValueError(f"Duplicate sample_id across inference files: {sample_id}")
            ids.add(sample_id)
        rows.extend(scenario_rows)

    actual_counts = Counter(count_key(row) for row in rows)
    expected_all = manifest.get("sample_counts") or {}
    expected_counts = {
        key: int(value)
        for key, value in expected_all.items()
        if key.split("/", 1)[0] in scenarios
    }
    if dict(sorted(actual_counts.items())) != dict(sorted(expected_counts.items())):
        raise RuntimeError(
            "Inference evidence count does not match manifest: "
            f"actual={dict(sorted(actual_counts.items()))}, "
            f"expected={dict(sorted(expected_counts.items()))}"
        )
    return sorted(rows, key=lambda row: row["sample_id"])


def events_in_annotated_window(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    event = row["event"]
    start = float(event["start_sec"])
    end = float(event["end_sec"])
    events = [
        state_event
        for state_event in row["state_events"]
        if start <= float(state_event["chunk_center_sec"]) < end
    ]
    if not events:
        raise ValueError(f"{row['sample_id']}: event window contains no state output")
    return sorted(events, key=lambda state_event: int(state_event["chunk_index"]))


def evaluate_record(
    row: Mapping[str, Any], interruption_threshold_sec: float
) -> Dict[str, Any]:
    scenario = str(row["scenario"])
    result: Dict[str, Any] = {
        "sample_id": row["sample_id"],
        "scenario": scenario,
        "language": row["language"],
        "label": row["label"],
    }
    if scenario == "interruption":
        window_events = events_in_annotated_window(row)
        detection = next(
            (
                event
                for event in window_events
                if str(event["public_state"]) == "nonidle"
            ),
            None,
        )
        if detection is None:
            result.update(
                {
                    "passed": False,
                    "outcome": "miss",
                    "within_latency_threshold": False,
                    "detection_latency_sec": None,
                    "first_nonidle_current_time_sec": None,
                    "first_nonidle_chunk_index": None,
                }
            )
        else:
            onset = float(row["event"]["start_sec"])
            current_time = float(detection["current_time_sec"])
            latency = current_time - onset
            if latency < 0:
                raise ValueError(f"{row['sample_id']}: negative interruption latency")
            within_latency_threshold = latency < interruption_threshold_sec
            result.update(
                {
                    # Detection and responsiveness are intentionally separate:
                    # any public nonidle attributed to the annotated speech
                    # window is a successful interruption detection.
                    "passed": True,
                    "outcome": (
                        "on_time" if within_latency_threshold else "late"
                    ),
                    "within_latency_threshold": within_latency_threshold,
                    "detection_latency_sec": latency,
                    "first_nonidle_current_time_sec": current_time,
                    "first_nonidle_chunk_index": int(detection["chunk_index"]),
                }
            )
        result["rule"] = (
            "pass when any public nonidle is in the annotated window; "
            "latency is first nonidle current_time-onset; "
            f"current_time-onset < {interruption_threshold_sec}s is diagnostic only"
        )
        return result

    if scenario == "backchannel":
        window_events = events_in_annotated_window(row)
        raw_detected = any(
            str(event["raw_state"]) == "backchannel" for event in window_events
        )
        public_nonidle = any(
            str(event["public_state"]) == "nonidle" for event in window_events
        )
        passed = raw_detected and not public_nonidle
        result.update(
            {
                "passed": passed,
                "outcome": "rejected" if passed else "failed_rejection",
                "raw_backchannel_detected": raw_detected,
                "public_nonidle_detected": public_nonidle,
                "rule": "at least one raw backchannel and zero public nonidle in window",
            }
        )
        return result

    if scenario == "background_speech":
        window_events = events_in_annotated_window(row)
        speak_event = next(
            (
                event
                for event in window_events
                if str(event["public_state"]) == "speak"
            ),
            None,
        )
        passed = speak_event is None
        result.update(
            {
                "passed": passed,
                "outcome": "rejected" if passed else "false_speak",
                "first_speak_chunk_index": (
                    int(speak_event["chunk_index"]) if speak_event else None
                ),
                "first_speak_current_time_sec": (
                    float(speak_event["current_time_sec"]) if speak_event else None
                ),
                "rule": (
                    "zero public speak states from background-speech onset "
                    "to the end of the clip"
                ),
            }
        )
        return result

    if scenario == "talking_to_other":
        window_events = events_in_annotated_window(row)
        speak_event = next(
            (
                event
                for event in window_events
                if str(event["public_state"]) == "speak"
            ),
            None,
        )
        passed = speak_event is None
        result.update(
            {
                "passed": passed,
                "outcome": "rejected" if passed else "false_speak",
                "first_speak_chunk_index": (
                    int(speak_event["chunk_index"]) if speak_event else None
                ),
                "first_speak_current_time_sec": (
                    float(speak_event["current_time_sec"]) if speak_event else None
                ),
                "rule": (
                    "zero public speak states from the end of the third speech "
                    "segment to the end of the clip"
                ),
            }
        )
        return result

    if scenario == "background_noise":
        speak_event = next(
            (
                event
                for event in row["state_events"]
                if str(event["public_state"]) == "speak"
            ),
            None,
        )
        passed = speak_event is None
        result.update(
            {
                "passed": passed,
                "outcome": "rejected" if passed else "false_speak",
                "first_speak_chunk_index": (
                    int(speak_event["chunk_index"]) if speak_event else None
                ),
                "first_speak_current_time_sec": (
                    float(speak_event["current_time_sec"]) if speak_event else None
                ),
                "rule": (
                    "zero public speak states across the capped noise clip "
                    "and appended trailing silence"
                ),
            }
        )
        return result

    if scenario == "easy_turn":
        nonidle_events = [
            event
            for event in row["state_events"]
            if str(event["public_state"]) == "nonidle"
        ]
        passed = bool(nonidle_events)
        result.update(
            {
                "passed": passed,
                "outcome": "accepted" if passed else "false_rejection",
                "first_nonidle_chunk_index": (
                    int(nonidle_events[0]["chunk_index"]) if nonidle_events else None
                ),
                "rule": "at least one public nonidle across the whole utterance",
            }
        )
        return result

    raise ValueError(f"Unsupported scenario: {scenario}")


def evaluate_records(
    rows: Sequence[Mapping[str, Any]], interruption_threshold_sec: float
) -> List[Dict[str, Any]]:
    return [evaluate_record(row, interruption_threshold_sec) for row in rows]


def rate_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "rate": passed / total if total else None,
    }


def macro_rate_summary(
    components: Mapping[str, Mapping[str, Any]], aggregation: str
) -> Dict[str, Any]:
    """Return an equal-weight mean of already computed rate summaries.

    ``passed`` and ``total`` are retained as evidence counts only.  The reported
    ``rate`` is deliberately *not* their ratio: it is the unweighted mean of the
    component rates.
    """
    if not components:
        raise ValueError("Macro average requires at least one component")
    invalid = [key for key, value in components.items() if value.get("rate") is None]
    if invalid:
        raise ValueError(f"Macro average has empty components: {invalid}")
    values = list(components.values())
    return {
        "aggregation": aggregation,
        "components": dict(components),
        "component_count": len(components),
        "passed": sum(int(value["passed"]) for value in values),
        "failed": sum(int(value["failed"]) for value in values),
        "total": sum(int(value["total"]) for value in values),
        "rate": sum(float(value["rate"]) for value in values) / len(values),
    }


def language_macro_rate_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Equal-weight the available languages for one evaluation scenario."""
    by_language: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["language"])].append(row)
    components = {
        language: rate_summary(by_language[language])
        for language in sorted(by_language)
    }
    return macro_rate_summary(components, "macro_average_by_language")


def scenario_language_macro_rate_summary(
    scenario_rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    """Equal-weight scenarios after equal-weighting languages within each one."""
    components = {
        scenario: language_macro_rate_summary(rows)
        for scenario, rows in sorted(scenario_rows.items())
        if rows
    }
    return macro_rate_summary(
        components, "hierarchical_macro_average_by_scenario_then_language"
    )


def group_key(row: Mapping[str, Any]) -> str:
    if row["scenario"] == "easy_turn":
        return f"easy_turn/{row['language']}/{row['label']}"
    return f"{row['scenario']}/{row['language']}"


def per_category_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    return {key: rate_summary(grouped[key]) for key in sorted(grouped)}


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def interruption_latency_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    interruption_rows = [row for row in rows if row["scenario"] == "interruption"]
    latencies = [
        float(row["detection_latency_sec"])
        for row in interruption_rows
        if row.get("detection_latency_sec") is not None
    ]
    outcomes = Counter(str(row["outcome"]) for row in interruption_rows)
    total = len(interruption_rows)
    detected = len(latencies)
    on_time = outcomes.get("on_time", 0)
    return {
        "population": "all detected samples, including late detections; misses excluded",
        "total": total,
        "detected": detected,
        "missed": outcomes.get("miss", 0),
        "detection_rate": detected / total if total else None,
        "on_time": on_time,
        "late": outcomes.get("late", 0),
        "on_time_rate": on_time / total if total else None,
        "mean_sec": sum(latencies) / len(latencies) if latencies else None,
        "p50_sec": percentile(latencies, 0.50),
        "p95_sec": percentile(latencies, 0.95),
    }


def strict_target(value: Optional[float], operator: str, threshold: float) -> Dict[str, Any]:
    if operator == "<":
        met = value is not None and value < threshold
    elif operator == ">":
        met = value is not None and value > threshold
    else:
        raise ValueError(f"Unsupported target operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "met": met,
    }


def format_rate(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def format_seconds(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f} s"


def render_report(title: str, metrics: Mapping[str, Any]) -> str:
    lines = [f"# {title}", "", f"Run ID: `{metrics['run_id']}`", ""]
    lines.extend(
        [
            "## Per-category results",
            "",
            "| Category | Passed | Failed | Total | Rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, summary in metrics["per_category"].items():
        lines.append(
            f"| {key} | {summary['passed']} | {summary['failed']} | "
            f"{summary['total']} | {format_rate(summary['rate'])} |"
        )
    lines.extend(["", "## Primary macro scene metrics", ""])
    for key, summary in metrics["aggregates"].items():
        lines.append(
            f"- `{key}`: {format_rate(summary['rate'])} "
            f"({summary['aggregation']}; evidence: {summary['passed']}/{summary['total']})"
        )
    sample_weighted = metrics.get("sample_weighted_aggregates", {})
    if sample_weighted:
        lines.extend(["", "## Sample-weighted reference metrics", ""])
        for key, summary in sample_weighted.items():
            lines.append(
                f"- `{key}`: {summary['passed']}/{summary['total']} "
                f"({format_rate(summary['rate'])})"
            )
    if "latency" in metrics:
        latency = metrics["latency"]
        lines.extend(
            [
                "",
                "## Interruption latency",
                "",
                f"- Diagnostic on-time threshold: `< {metrics['interruption_latency_threshold_sec']:.3f} s`",
                f"- Population: {latency['population']}",
                f"- Detection rate: {format_rate(latency['detection_rate'])}",
                f"- On-time rate: {format_rate(latency['on_time_rate'])}",
                f"- On-time / late / miss: {latency['on_time']} / {latency['late']} / {latency['missed']}",
                f"- Mean: {format_seconds(latency['mean_sec'])}",
                f"- P50: {format_seconds(latency['p50_sec'])}",
                f"- P95: {format_seconds(latency['p95_sec'])}",
            ]
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
    for key, target in metrics["technical_targets"].items():
        condition = f"{target['operator']} {format_rate(target['threshold'])}"
        lines.append(
            f"| {key} | {format_rate(target['value'])} | {condition} | "
            f"{'YES' if target['met'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Not evaluated",
            "",
            *[f"- {item}: `not_evaluated`" for item in metrics["not_evaluated"]],
            "",
            "The report uses only SoulX raw/public state outputs. Dialogue-model and TTS cancellation latency are outside this evaluation.",
        ]
    )
    return "\n".join(lines)


def update_manifest_evaluation(run_dir: Path, name: str, metrics_path: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    evaluations = manifest.setdefault("evaluations", {})
    evaluations[name] = {
        "metrics": metrics_path.relative_to(run_dir).as_posix(),
        "complete": True,
    }
    if name == "vad":
        manifest["not_evaluated"] = [
            item
            for item in list(manifest.get("not_evaluated") or [])
            if item != "vad_endpoint_prediction"
        ]
    required_evaluations = (
        {"vad"}
        if manifest.get("evaluation_scope") == "vad_only"
        else {"interruption", "rejection", "vad"}
    )
    if required_evaluations.issubset(evaluations):
        manifest["status"] = "evaluation_complete"
    atomic_write_json(manifest_path, manifest)
