from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import time
import traceback
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


SCENARIOS: Tuple[str, ...] = (
    "interruption",
    "backchannel",
    "background_noise",
    "easy_turn",
)
OPTIONAL_SCENARIOS: Tuple[str, ...] = (
    "background_speech",
    "talking_to_other",
)
SUPPORTED_SCENARIOS: Tuple[str, ...] = SCENARIOS + OPTIONAL_SCENARIOS
SPECIALIZED_RUNNER_SCENARIOS: Tuple[str, ...] = ("easy_turn",)
LANGUAGES: Tuple[str, ...] = ("en", "zh", "neutral")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    scenario: str
    language: str
    label: str
    audio_path: Path
    timestamp_path: Optional[Path]
    timestamp_source: Optional[str]
    event_start_sec: Optional[float]
    event_end_sec: Optional[float]
    duration_sec: float
    original_sample_rate: int
    original_frames: int

    @property
    def count_key(self) -> str:
        if self.scenario == "easy_turn":
            return f"{self.scenario}/{self.language}/{self.label}"
        return f"{self.scenario}/{self.language}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_eval_config(config_path: str | Path) -> Dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    raw = load_yaml(config_file)
    paths = raw.get("paths") or {}
    project_value = paths.get("project_root", "..")
    project_root = Path(project_value).expanduser()
    if not project_root.is_absolute():
        project_root = config_file.parent / project_root
    project_root = project_root.resolve()
    raw["_config_path"] = config_file
    raw["_project_root"] = project_root
    raw["_output_root"] = resolve_path(
        project_root, str(paths.get("output_root", "SoulX-Duplug-Eval/output"))
    )
    return raw


def error_record(code: str, message: str, **context: Any) -> Dict[str, Any]:
    return {"code": code, "message": message, **context}


def relative_display(path: Optional[Path], project_root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


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


def _riff_wave_info(path: Path) -> Tuple[int, int, float]:
    """Read frame metadata from PCM, IEEE-float, or extensible RIFF/WAV."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Unsupported WAV container: {path}")
        sample_rate: Optional[int] = None
        block_align: Optional[int] = None
        data_size: Optional[int] = None
        while True:
            chunk_header = handle.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError(f"Truncated WAV chunk header: {path}")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_data_start = handle.tell()
            if chunk_id == b"fmt ":
                fmt = handle.read(min(chunk_size, 16))
                if len(fmt) < 16:
                    raise ValueError(f"Truncated WAV fmt chunk: {path}")
                _, _, sample_rate_value, _, block_align_value, _ = struct.unpack(
                    "<HHIIHH", fmt
                )
                sample_rate = int(sample_rate_value)
                block_align = int(block_align_value)
            elif chunk_id == b"data":
                data_size = int(chunk_size)
            handle.seek(chunk_data_start + chunk_size + (chunk_size % 2))
        if not sample_rate or not block_align or data_size is None:
            raise ValueError(f"WAV is missing fmt or data metadata: {path}")
        frames = data_size // block_align
        return sample_rate, frames, frames / sample_rate


def audio_info(path: Path) -> Tuple[int, int, float]:
    """Return sample rate, frames, duration without decoding the full audio."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return int(info.samplerate), int(info.frames), float(info.duration)
    except ModuleNotFoundError:
        # Keep preflight usable on a minimal management node.  ``wave`` rejects
        # IEEE-float WAV (format 3), which SID-Bench uses for simulated silence.
        try:
            with wave.open(str(path), "rb") as handle:
                sample_rate = int(handle.getframerate())
                frames = int(handle.getnframes())
            return sample_rate, frames, frames / sample_rate
        except wave.Error:
            return _riff_wave_info(path)


def parse_event_timestamp(path: Path, source: str) -> Tuple[float, float]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if source == "metadata.timestamps":
        timestamps = value.get("timestamps") if isinstance(value, dict) else None
    elif source == "interrupt.json[0].timestamp":
        timestamps = (
            value[0].get("timestamp")
            if isinstance(value, list) and value and isinstance(value[0], dict)
            else None
        )
    else:
        raise ValueError(f"Unsupported timestamp source: {source}")
    if not isinstance(timestamps, list) or len(timestamps) != 2:
        raise ValueError(f"Expected a two-element timestamp in {path}")
    start, end = float(timestamps[0]), float(timestamps[1])
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"Timestamp must be finite in {path}")
    return start, end


def chunk_center_in_window(
    event_start: float,
    event_end: float,
    duration: float,
    chunk_duration: float,
) -> bool:
    chunk_count = int(math.ceil(duration / chunk_duration))
    return any(
        event_start <= (index + 0.5) * chunk_duration < event_end
        for index in range(chunk_count)
    )


def _make_sample_id(scenario: str, language: str, relative_parent: Path) -> str:
    suffix = relative_parent.as_posix().strip("/") or "root"
    return f"{scenario}/{language}/{suffix}"


def discover_fdb_samples(
    scenario: str,
    language: str,
    root: Path,
    chunk_duration: float,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]]]:
    samples: List[SampleSpec] = []
    errors: List[Dict[str, Any]] = []
    if not root.is_dir():
        return [], [
            error_record(
                "dataset_root_missing",
                f"Dataset directory does not exist: {root}",
                scenario=scenario,
                language=language,
                path=str(root),
            )
        ]

    audio_paths = sorted(path for path in root.rglob("input.wav") if path.is_file())
    if not audio_paths:
        errors.append(
            error_record(
                "dataset_empty",
                f"No exact input.wav files found under {root}",
                scenario=scenario,
                language=language,
                path=str(root),
            )
        )
        return samples, errors

    for audio_path in audio_paths:
        if scenario == "interruption" and language == "zh":
            timestamp_path = audio_path.parent / "interrupt.json"
            timestamp_source = "interrupt.json[0].timestamp"
        else:
            timestamp_path = audio_path.parent / "metadata.json"
            timestamp_source = "metadata.timestamps"
        relative_parent = audio_path.parent.relative_to(root)
        sample_id = _make_sample_id(scenario, language, relative_parent)
        if not timestamp_path.is_file():
            errors.append(
                error_record(
                    "timestamp_file_missing",
                    f"Missing timestamp file for {audio_path}",
                    sample_id=sample_id,
                    path=str(timestamp_path),
                )
            )
            continue
        try:
            start, annotated_end = parse_event_timestamp(
                timestamp_path, timestamp_source
            )
            sample_rate, frames, duration = audio_info(audio_path)
        except Exception as exc:
            errors.append(
                error_record(
                    "sample_metadata_invalid",
                    str(exc),
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                    timestamp_path=str(timestamp_path),
                )
            )
            continue
        if start < 0 or annotated_end <= start or annotated_end > duration:
            errors.append(
                error_record(
                    "timestamp_out_of_bounds",
                    f"Event [{start}, {annotated_end}) is invalid for "
                    f"{duration:.6f}s audio",
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                )
            )
            continue
        if scenario == "background_speech":
            end = duration
            timestamp_source = "metadata.timestamps[0] to audio_duration"
        else:
            end = annotated_end
        if not chunk_center_in_window(start, end, duration, chunk_duration):
            errors.append(
                error_record(
                    "empty_event_window",
                    "No streaming chunk center falls inside the event window",
                    sample_id=sample_id,
                    event_start_sec=start,
                    event_end_sec=end,
                )
            )
            continue
        samples.append(
            SampleSpec(
                sample_id=sample_id,
                scenario=scenario,
                language=language,
                label=scenario,
                audio_path=audio_path.resolve(),
                timestamp_path=timestamp_path.resolve(),
                timestamp_source=timestamp_source,
                event_start_sec=start,
                event_end_sec=end,
                duration_sec=duration,
                original_sample_rate=sample_rate,
                original_frames=frames,
            )
        )
    return samples, errors


def discover_humdial_background_speech_samples(
    language: str,
    root: Path,
    chunk_duration: float,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]]]:
    """Load Chinese HumDial clips whose second utterance is background speech."""
    samples: List[SampleSpec] = []
    errors: List[Dict[str, Any]] = []
    if language != "zh":
        return [], [
            error_record(
                "dataset_language_unsupported",
                "HumDial background-speech evaluation is configured for Chinese only",
                scenario="background_speech",
                language=language,
                path=str(root),
            )
        ]
    if not root.is_dir():
        return [], [
            error_record(
                "dataset_root_missing",
                f"Dataset directory does not exist: {root}",
                scenario="background_speech",
                language=language,
                path=str(root),
            )
        ]

    audio_paths = sorted(
        path
        for path in root.glob("*.wav")
        if path.is_file() and not path.name.startswith("clean_")
    )
    if not audio_paths:
        return [], [
            error_record(
                "dataset_empty",
                f"No non-clean WAV files found directly under {root}",
                scenario="background_speech",
                language=language,
                path=str(root),
            )
        ]

    for audio_path in audio_paths:
        annotation_path = audio_path.with_suffix(".json")
        relative = audio_path.relative_to(root).with_suffix("")
        sample_id = _make_sample_id("background_speech", language, relative)
        if not annotation_path.is_file():
            errors.append(
                error_record(
                    "timestamp_file_missing",
                    f"Missing HumDial segment annotation for {audio_path}",
                    sample_id=sample_id,
                    path=str(annotation_path),
                )
            )
            continue
        try:
            with annotation_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            segments = (
                annotation.get("speech_segments")
                if isinstance(annotation, dict)
                else None
            )
            if not isinstance(segments, list):
                raise ValueError(
                    f"Expected speech_segments list in {annotation_path}"
                )
            if len(segments) != 2:
                continue
            second_segment = segments[1]
            if not isinstance(second_segment, dict):
                raise ValueError(
                    f"Second speech segment must be an object in {annotation_path}"
                )
            second_start = float(second_segment["xmin"])
            second_end = float(second_segment["xmax"])
            sample_rate, frames, duration = audio_info(audio_path)
        except Exception as exc:
            errors.append(
                error_record(
                    "sample_metadata_invalid",
                    str(exc),
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                    timestamp_path=str(annotation_path),
                )
            )
            continue
        if (
            not math.isfinite(second_start)
            or not math.isfinite(second_end)
            or second_start < 0
            or second_end <= second_start
            or second_end > duration
        ):
            errors.append(
                error_record(
                    "timestamp_out_of_bounds",
                    f"Second segment [{second_start}, {second_end}) is invalid for "
                    f"{duration:.6f}s audio",
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                )
            )
            continue
        if not chunk_center_in_window(
            second_start, duration, duration, chunk_duration
        ):
            errors.append(
                error_record(
                    "empty_event_window",
                    "No streaming chunk center falls after background-speech onset",
                    sample_id=sample_id,
                    event_start_sec=second_start,
                    event_end_sec=duration,
                )
            )
            continue
        samples.append(
            SampleSpec(
                sample_id=sample_id,
                scenario="background_speech",
                language=language,
                label="background_speech",
                audio_path=audio_path.resolve(),
                timestamp_path=annotation_path.resolve(),
                timestamp_source="speech_segments[1].xmin to audio_duration",
                event_start_sec=second_start,
                event_end_sec=duration,
                duration_sec=duration,
                original_sample_rate=sample_rate,
                original_frames=frames,
            )
        )
    return samples, errors


def discover_easy_turn_samples(
    language: str,
    root: Path,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]]]:
    samples: List[SampleSpec] = []
    errors: List[Dict[str, Any]] = []
    if not root.is_dir():
        return [], [
            error_record(
                "dataset_root_missing",
                f"Dataset directory does not exist: {root}",
                scenario="easy_turn",
                language=language,
                path=str(root),
            )
        ]
    for label in ("complete", "incomplete"):
        label_root = root / label
        if not label_root.is_dir():
            errors.append(
                error_record(
                    "label_directory_missing",
                    f"Missing Easy-Turn label directory: {label_root}",
                    scenario="easy_turn",
                    language=language,
                    label=label,
                    path=str(label_root),
                )
            )
            continue
        audio_paths = sorted(path for path in label_root.rglob("*.wav") if path.is_file())
        if not audio_paths:
            errors.append(
                error_record(
                    "dataset_empty",
                    f"No WAV files found under {label_root}",
                    scenario="easy_turn",
                    language=language,
                    label=label,
                )
            )
            continue
        for audio_path in audio_paths:
            relative = audio_path.relative_to(root).with_suffix("")
            sample_id = _make_sample_id("easy_turn", language, relative)
            try:
                sample_rate, frames, duration = audio_info(audio_path)
            except Exception as exc:
                errors.append(
                    error_record(
                        "audio_invalid",
                        str(exc),
                        sample_id=sample_id,
                        audio_path=str(audio_path),
                    )
                )
                continue
            if frames <= 0 or sample_rate <= 0 or duration <= 0:
                errors.append(
                    error_record(
                        "audio_empty",
                        "Easy-Turn audio must contain at least one frame",
                        sample_id=sample_id,
                        audio_path=str(audio_path),
                    )
                )
                continue
            samples.append(
                SampleSpec(
                    sample_id=sample_id,
                    scenario="easy_turn",
                    language=language,
                    label=label,
                    audio_path=audio_path.resolve(),
                    timestamp_path=None,
                    timestamp_source=None,
                    event_start_sec=None,
                    event_end_sec=None,
                    duration_sec=duration,
                    original_sample_rate=sample_rate,
                    original_frames=frames,
                )
            )
    return samples, errors


def discover_humdial_talking_to_other_samples(
    language: str,
    root: Path,
    chunk_duration: float,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]]]:
    """Load flat HumDial clips and use the third utterance end as the reject boundary."""
    samples: List[SampleSpec] = []
    errors: List[Dict[str, Any]] = []
    if language != "zh":
        return [], [
            error_record(
                "dataset_language_unsupported",
                "HumDial talking-to-other evaluation is configured for Chinese only",
                scenario="talking_to_other",
                language=language,
                path=str(root),
            )
        ]
    if not root.is_dir():
        return [], [
            error_record(
                "dataset_root_missing",
                f"Dataset directory does not exist: {root}",
                scenario="talking_to_other",
                language=language,
                path=str(root),
            )
        ]

    audio_paths = sorted(
        path
        for path in root.glob("*.wav")
        if path.is_file() and not path.name.startswith("clean_")
    )
    if not audio_paths:
        return [], [
            error_record(
                "dataset_empty",
                f"No WAV files found directly under {root}",
                scenario="talking_to_other",
                language=language,
                path=str(root),
            )
        ]

    for audio_path in audio_paths:
        annotation_path = audio_path.with_suffix(".json")
        relative = audio_path.relative_to(root).with_suffix("")
        sample_id = _make_sample_id("talking_to_other", language, relative)
        if not annotation_path.is_file():
            errors.append(
                error_record(
                    "timestamp_file_missing",
                    f"Missing HumDial segment annotation for {audio_path}",
                    sample_id=sample_id,
                    path=str(annotation_path),
                )
            )
            continue
        try:
            with annotation_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            segments = (
                annotation.get("speech_segments")
                if isinstance(annotation, dict)
                else None
            )
            if not isinstance(segments, list) or len(segments) != 3:
                raise ValueError(
                    f"Expected exactly three speech_segments in {annotation_path}"
                )
            third_segment = segments[2]
            if not isinstance(third_segment, dict):
                raise ValueError(
                    f"Third speech segment must be an object in {annotation_path}"
                )
            third_start = float(third_segment["xmin"])
            third_end = float(third_segment["xmax"])
            sample_rate, frames, duration = audio_info(audio_path)
        except Exception as exc:
            errors.append(
                error_record(
                    "sample_metadata_invalid",
                    str(exc),
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                    timestamp_path=str(annotation_path),
                )
            )
            continue
        if (
            not math.isfinite(third_start)
            or not math.isfinite(third_end)
            or third_start < 0
            or third_end <= third_start
            or third_end >= duration
        ):
            errors.append(
                error_record(
                    "timestamp_out_of_bounds",
                    f"Third segment [{third_start}, {third_end}) is invalid for "
                    f"{duration:.6f}s audio",
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                )
            )
            continue
        if not chunk_center_in_window(third_end, duration, duration, chunk_duration):
            errors.append(
                error_record(
                    "empty_event_window",
                    "No streaming chunk center falls after the third speech segment",
                    sample_id=sample_id,
                    event_start_sec=third_end,
                    event_end_sec=duration,
                )
            )
            continue
        samples.append(
            SampleSpec(
                sample_id=sample_id,
                scenario="talking_to_other",
                language=language,
                label="talking_to_other",
                audio_path=audio_path.resolve(),
                timestamp_path=annotation_path.resolve(),
                timestamp_source="speech_segments[2].xmax",
                event_start_sec=third_end,
                event_end_sec=duration,
                duration_sec=duration,
                original_sample_rate=sample_rate,
                original_frames=frames,
            )
        )
    return samples, errors


def discover_sid_noise_samples(
    language: str,
    root: Path,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]]]:
    """Load SID-Bench's language-independent silence/environmental-noise split."""
    samples: List[SampleSpec] = []
    errors: List[Dict[str, Any]] = []
    if not root.is_dir():
        return [], [
            error_record(
                "dataset_root_missing",
                f"Dataset directory does not exist: {root}",
                scenario="background_noise",
                language=language,
                path=str(root),
            )
        ]

    annotation_path = root / "silence_noise_test.jsonl"
    audio_root = root / "silence_or_noise"
    if not annotation_path.is_file():
        errors.append(
            error_record(
                "annotation_file_missing",
                f"Missing SID-Bench annotation file: {annotation_path}",
                scenario="background_noise",
                language=language,
                path=str(annotation_path),
            )
        )
    if not audio_root.is_dir():
        errors.append(
            error_record(
                "dataset_root_missing",
                f"Missing SID-Bench audio directory: {audio_root}",
                scenario="background_noise",
                language=language,
                path=str(audio_root),
            )
        )
    if errors:
        return samples, errors

    annotations: List[Tuple[int, Mapping[str, Any]]] = []
    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                annotations.append((line_number, value))
    except Exception as exc:
        return [], [
            error_record(
                "annotation_file_invalid",
                str(exc),
                scenario="background_noise",
                language=language,
                path=str(annotation_path),
            )
        ]
    if not annotations:
        return [], [
            error_record(
                "dataset_empty",
                f"No annotations found in {annotation_path}",
                scenario="background_noise",
                language=language,
            )
        ]

    seen_audio: set[str] = set()
    for line_number, annotation in annotations:
        audio_name = annotation.get("audio")
        if not isinstance(audio_name, str) or not audio_name or Path(audio_name).name != audio_name:
            errors.append(
                error_record(
                    "sample_metadata_invalid",
                    "SID-Bench audio must be a plain filename",
                    line_number=line_number,
                    audio=audio_name,
                )
            )
            continue
        sample_id = _make_sample_id(
            "background_noise", language, Path(audio_name).with_suffix("")
        )
        if audio_name in seen_audio:
            errors.append(
                error_record(
                    "duplicate_annotation",
                    f"Duplicate SID-Bench audio annotation: {audio_name}",
                    sample_id=sample_id,
                    line_number=line_number,
                )
            )
            continue
        seen_audio.add(audio_name)
        audio_path = audio_root / audio_name
        try:
            if annotation.get("total_nonbreak") is not True:
                raise ValueError("total_nonbreak must be true for pure-noise samples")
            if annotation.get("text_with_break") is not None:
                raise ValueError("text_with_break must be null for pure-noise samples")
            annotated_duration = float(annotation["duration"])
            break_time = float(annotation["break_time"])
            sample_rate, frames, duration = audio_info(audio_path)
            if not math.isfinite(annotated_duration) or annotated_duration <= 0:
                raise ValueError("annotated duration must be positive and finite")
            if abs(annotated_duration - duration) > 0.05:
                raise ValueError(
                    f"annotated duration {annotated_duration:.6f}s differs from "
                    f"audio duration {duration:.6f}s"
                )
            if not math.isfinite(break_time) or abs(break_time - annotated_duration) > 0.05:
                raise ValueError("break_time must equal duration for pure-noise samples")
        except Exception as exc:
            errors.append(
                error_record(
                    "sample_metadata_invalid",
                    str(exc),
                    sample_id=sample_id,
                    line_number=line_number,
                    audio_path=str(audio_path),
                    timestamp_path=str(annotation_path),
                )
            )
            continue
        samples.append(
            SampleSpec(
                sample_id=sample_id,
                scenario="background_noise",
                language=language,
                label="background_noise",
                audio_path=audio_path.resolve(),
                timestamp_path=annotation_path.resolve(),
                timestamp_source="SID-bench silence_noise_test.jsonl whole clip",
                event_start_sec=0.0,
                event_end_sec=duration,
                duration_sec=duration,
                original_sample_rate=sample_rate,
                original_frames=frames,
            )
        )

    actual_audio = {
        path.name for path in audio_root.glob("*.wav") if path.is_file()
    }
    for audio_name in sorted(actual_audio - seen_audio):
        errors.append(
            error_record(
                "unreferenced_audio",
                f"SID-Bench WAV is not referenced by the annotation file: {audio_name}",
                scenario="background_noise",
                language=language,
            )
        )
    return samples, errors


def _language_stream_settings(config: Mapping[str, Any], language: str) -> Tuple[int, int]:
    project_root = Path(config["_project_root"])
    language_config = config["model"]["languages"][language]
    base_path = resolve_path(project_root, str(language_config["base_config"]))
    base = load_yaml(base_path)
    input_config = base["infer_config"]["input"]
    return int(input_config["sample_rate"]), int(input_config["chunk_size"])


def discover_all_samples(
    config: Mapping[str, Any],
    scenarios: Sequence[str],
    limit_per_dataset: int = 0,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    project_root = Path(config["_project_root"])
    samples: List[SampleSpec] = []
    errors: List[Dict[str, Any]] = []
    stream_settings: Dict[str, Dict[str, int]] = {}
    for language in LANGUAGES:
        try:
            sample_rate, chunk_size = _language_stream_settings(config, language)
            stream_settings[language] = {
                "sample_rate": sample_rate,
                "chunk_size": chunk_size,
            }
        except Exception as exc:
            errors.append(
                error_record(
                    "model_config_invalid",
                    str(exc),
                    language=language,
                )
            )

    datasets = config.get("datasets") or {}
    for scenario in scenarios:
        scenario_config = datasets.get(scenario)
        if not isinstance(scenario_config, dict):
            errors.append(
                error_record(
                    "dataset_config_missing",
                    f"No dataset mapping configured for {scenario}",
                    scenario=scenario,
                )
            )
            continue
        for language, root_value in sorted(scenario_config.items()):
            if language not in stream_settings:
                continue
            root = resolve_path(project_root, str(root_value))
            if scenario == "easy_turn":
                found, found_errors = discover_easy_turn_samples(language, root)
            elif scenario == "background_speech" and language == "zh":
                settings = stream_settings[language]
                chunk_duration = settings["chunk_size"] / settings["sample_rate"]
                found, found_errors = discover_humdial_background_speech_samples(
                    language, root, chunk_duration
                )
            elif scenario == "talking_to_other":
                settings = stream_settings[language]
                chunk_duration = settings["chunk_size"] / settings["sample_rate"]
                found, found_errors = discover_humdial_talking_to_other_samples(
                    language, root, chunk_duration
                )
            elif scenario == "background_noise":
                found, found_errors = discover_sid_noise_samples(language, root)
            else:
                settings = stream_settings[language]
                chunk_duration = settings["chunk_size"] / settings["sample_rate"]
                found, found_errors = discover_fdb_samples(
                    scenario, language, root, chunk_duration
                )
            if limit_per_dataset > 0:
                if scenario == "easy_turn":
                    limited: List[SampleSpec] = []
                    by_label: Dict[str, List[SampleSpec]] = defaultdict(list)
                    for sample in found:
                        by_label[sample.label].append(sample)
                    for label in sorted(by_label):
                        limited.extend(by_label[label][:limit_per_dataset])
                    found = limited
                else:
                    found = found[:limit_per_dataset]
            samples.extend(found)
            errors.extend(found_errors)

    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(s.sample_id for s in samples).items() if count > 1
    )
    for sample_id in duplicate_ids:
        errors.append(
            error_record(
                "duplicate_sample_id",
                f"Duplicate sample id: {sample_id}",
                sample_id=sample_id,
            )
        )
    return sorted(samples, key=lambda item: item.sample_id), errors, stream_settings


def validate_model_assets(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    project_root = Path(config["_project_root"])
    errors: List[Dict[str, Any]] = []
    checkpoint = resolve_path(project_root, str(config["model"]["checkpoint"]))
    if not checkpoint.is_file():
        errors.append(
            error_record(
                "checkpoint_missing",
                f"Checkpoint does not exist: {checkpoint}",
                path=str(checkpoint),
            )
        )
    for language, language_config in config["model"]["languages"].items():
        base_path = resolve_path(project_root, str(language_config["base_config"]))
        if not base_path.is_file():
            errors.append(
                error_record(
                    "model_config_missing",
                    f"Base model config does not exist: {base_path}",
                    language=language,
                    path=str(base_path),
                )
            )
            continue
        try:
            base = load_yaml(base_path)
            model_config = base["model_config"]
            for field in ("glm_tokenizer_path", "model_name"):
                asset_path = resolve_path(project_root, str(model_config[field]))
                if not asset_path.exists():
                    errors.append(
                        error_record(
                            "model_asset_missing",
                            f"Model asset does not exist: {asset_path}",
                            language=language,
                            field=field,
                            path=str(asset_path),
                        )
                    )
        except Exception as exc:
            errors.append(
                error_record(
                    "model_config_invalid",
                    str(exc),
                    language=language,
                    path=str(base_path),
                )
            )
    return errors


def prepare_runtime_configs(config: Mapping[str, Any], run_dir: Path) -> Dict[str, Path]:
    project_root = Path(config["_project_root"])
    checkpoint = resolve_path(project_root, str(config["model"]["checkpoint"]))
    far_field_threshold = float(config["model"]["far_field_threshold"])
    output: Dict[str, Path] = {}
    for language, language_config in config["model"]["languages"].items():
        base_path = resolve_path(project_root, str(language_config["base_config"]))
        runtime = load_yaml(base_path)
        runtime["model_config"]["init_ckpt_path_lora"] = str(checkpoint)
        runtime["infer_config"]["far_field_threshold"] = far_field_threshold
        runtime["infer_config"]["asr"] = dict(language_config["asr"])
        output_path = run_dir / "configs" / f"{language}_config_used.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(runtime, handle, allow_unicode=True, sort_keys=False)
        os.replace(temporary, output_path)
        output[language] = output_path
    return output


def read_and_normalize_audio(
    audio_path: Path,
    target_sample_rate: int,
    max_duration_sec: Optional[float] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Read audio, optionally limiting decoding and inference to its leading segment."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as handle:
        original_sample_rate = int(handle.samplerate)
        source_frames = int(handle.frames)
        frames_to_read = source_frames
        if max_duration_sec is not None:
            if max_duration_sec <= 0:
                raise ValueError("max_duration_sec must be positive when provided")
            frames_to_read = min(
                source_frames, int(round(max_duration_sec * original_sample_rate))
            )
        audio = handle.read(frames=frames_to_read, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    resampled = int(original_sample_rate) != int(target_sample_rate)
    if resampled:
        import soxr

        audio = soxr.resample(audio, int(original_sample_rate), int(target_sample_rate))
        audio = np.asarray(audio, dtype=np.float32)
    return audio, {
        "original_sample_rate": int(original_sample_rate),
        "target_sample_rate": int(target_sample_rate),
        "source_frames": source_frames,
        "source_duration_sec": source_frames / original_sample_rate,
        "truncated": frames_to_read < source_frames,
        "truncation_limit_sec": max_duration_sec,
        "resampled": resampled,
        "normalized_frames": int(len(audio)),
        "normalized_duration_sec": len(audio) / target_sample_rate,
    }


def advance_virtual_time(
    previous_current_time: float,
    chunk_available_time: float,
    process_time: float,
) -> Tuple[float, float, float]:
    process_start = max(chunk_available_time, previous_current_time)
    queue_delay = max(0.0, previous_current_time - chunk_available_time)
    current_time = process_start + process_time
    return process_start, queue_delay, current_time


def _cuda_synchronizer(model: Any) -> Callable[[], None]:
    try:
        import torch

        device = str(getattr(model, "device", "cpu"))
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:
        pass
    return lambda: None


def stream_audio(
    model: Any,
    audio: Any,
    chunk_size: int,
    sample_rate: int,
    timer: Callable[[], float] = time.perf_counter,
    synchronizer: Optional[Callable[[], None]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    import numpy as np

    if len(audio) <= 0:
        raise ValueError("Audio must contain at least one sample")
    sync = synchronizer or _cuda_synchronizer(model)
    chunk_duration = chunk_size / sample_rate
    previous_current = 0.0
    events: List[Dict[str, Any]] = []
    padded_samples = 0
    for chunk_index, start_sample in enumerate(range(0, len(audio), chunk_size)):
        chunk = np.asarray(audio[start_sample : start_sample + chunk_size], dtype=np.float32)
        if len(chunk) < chunk_size:
            padded_samples = chunk_size - len(chunk)
            chunk = np.pad(chunk, (0, padded_samples))
        sync()
        started = timer()
        result = model.process(chunk.astype(np.float32, copy=False))
        sync()
        process_time = max(0.0, timer() - started)
        chunk_start = chunk_index * chunk_duration
        chunk_end = (chunk_index + 1) * chunk_duration
        available = chunk_end
        virtual_start, queue_delay, current_time = advance_virtual_time(
            previous_current, available, process_time
        )
        raw_token = result.get("raw_state_token")
        events.append(
            {
                "chunk_index": chunk_index,
                "chunk_start_sec": chunk_start,
                "chunk_end_sec": chunk_end,
                "chunk_center_sec": (chunk_start + chunk_end) / 2.0,
                "process_time_sec": process_time,
                "queue_delay_sec": queue_delay,
                "virtual_process_start_sec": virtual_start,
                "current_time_sec": current_time,
                "raw_state": str(result.get("raw_state", "")),
                "raw_state_token": None if raw_token is None else str(raw_token),
                "public_state": str(result.get("state", "")),
            }
        )
        previous_current = current_time
    return events, padded_samples


def warm_up_model(
    model: Any,
    sample: SampleSpec,
    chunk_size: int,
    sample_rate: int,
    max_duration_sec: Optional[float] = None,
    trailing_silence_sec: float = 0.0,
) -> None:
    import numpy as np

    if trailing_silence_sec < 0:
        raise ValueError("trailing_silence_sec must be non-negative")
    audio, _ = read_and_normalize_audio(
        sample.audio_path, sample_rate, max_duration_sec=max_duration_sec
    )
    trailing_silence_samples = int(round(trailing_silence_sec * sample_rate))
    if trailing_silence_samples:
        audio = np.concatenate(
            (audio, np.zeros(trailing_silence_samples, dtype=np.float32))
        )
    model.reset()
    stream_audio(model, audio, chunk_size, sample_rate)
    model.reset()


def infer_sample(
    model: Any,
    sample: SampleSpec,
    chunk_size: int,
    sample_rate: int,
    project_root: Path,
    max_duration_sec: Optional[float] = None,
    trailing_silence_sec: float = 0.0,
) -> Dict[str, Any]:
    import numpy as np

    if trailing_silence_sec < 0:
        raise ValueError("trailing_silence_sec must be non-negative")
    audio, audio_metadata = read_and_normalize_audio(
        sample.audio_path, sample_rate, max_duration_sec=max_duration_sec
    )
    evaluated_audio_duration_sec = len(audio) / sample_rate
    trailing_silence_samples = int(round(trailing_silence_sec * sample_rate))
    if trailing_silence_samples:
        audio = np.concatenate(
            (audio, np.zeros(trailing_silence_samples, dtype=np.float32))
        )
    model.reset()
    events, padded_samples = stream_audio(model, audio, chunk_size, sample_rate)
    event = None
    if sample.event_start_sec is not None and sample.event_end_sec is not None:
        effective_event_end_sec = min(sample.event_end_sec, evaluated_audio_duration_sec)
        event = {
            "start_sec": sample.event_start_sec,
            "end_sec": effective_event_end_sec,
            "interval": "[start,end)",
            "chunk_membership": "chunk_center",
            "timestamp_source": sample.timestamp_source,
            "timestamp_path": relative_display(sample.timestamp_path, project_root),
        }
        for state_event in events:
            center = float(state_event["chunk_center_sec"])
            state_event["in_event_window"] = (
                sample.event_start_sec <= center < effective_event_end_sec
            )
    else:
        for state_event in events:
            state_event["in_event_window"] = None
    return {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "scenario": sample.scenario,
        "language": sample.language,
        "label": sample.label,
        "audio": {
            "path": relative_display(sample.audio_path, project_root),
            **audio_metadata,
            "inference_duration_sec": len(audio) / sample_rate,
            "trailing_silence_sec": trailing_silence_samples / sample_rate,
            "trailing_silence_samples": trailing_silence_samples,
            "right_padding_samples": padded_samples,
        },
        "event": event,
        "state_events": events,
    }


def _sample_counts(samples: Sequence[SampleSpec]) -> Dict[str, int]:
    return dict(sorted(Counter(sample.count_key for sample in samples).items()))


def _build_manifest(
    config: Mapping[str, Any],
    run_id: str,
    scenarios: Sequence[str],
    samples: Sequence[SampleSpec],
    stream_settings: Mapping[str, Any],
    limit_per_dataset: int,
    background_noise_max_duration_sec: float,
    background_noise_trailing_silence_sec: float,
    formal_scenario_evaluation: bool = False,
) -> Dict[str, Any]:
    project_root = Path(config["_project_root"])
    checkpoint = resolve_path(project_root, str(config["model"]["checkpoint"]))
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "preflight_running",
        "complete": False,
        "formal_full_evaluation": scenarios == SCENARIOS and limit_per_dataset == 0,
        "formal_scenario_evaluation": formal_scenario_evaluation,
        "evaluation_scope": (
            "full"
            if scenarios == SCENARIOS and limit_per_dataset == 0
            else "vad_only"
            if formal_scenario_evaluation
            else "diagnostic_subset"
        ),
        "selected_scenarios": list(scenarios),
        "limit_per_dataset": limit_per_dataset,
        "config_path": relative_display(Path(config["_config_path"]), project_root),
        "project_root": str(project_root),
        "model": {
            "checkpoint": relative_display(checkpoint, project_root),
            "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "far_field_threshold": float(config["model"]["far_field_threshold"]),
            "runtime_configs": {},
        },
        "sample_counts": _sample_counts(samples),
        "input_truncation": {
            "background_noise": {
                "max_duration_sec": background_noise_max_duration_sec,
                "policy": "use the first max_duration_sec of each clip when longer",
                "trailing_silence_sec": background_noise_trailing_silence_sec,
                "trailing_silence_policy": "append silence after truncation and evaluate all resulting state outputs",
            }
        },
        "num_samples": len(samples),
        "stream_settings": dict(stream_settings),
        "timing": {
            "clock": "time.perf_counter",
            "cuda_synchronize": "before_and_after_process_when_cuda",
            "data_loading_and_resampling_excluded": True,
            "warmup": "first_valid_sample_per_language_then_reset",
            "current_time_formula": "max(chunk_end, previous_current_time) + process_time",
            "chunk_window_membership": "chunk_center in [event_start,event_end)",
        },
        "not_evaluated": list(config.get("not_evaluated") or []),
        "inference_files": {
            scenario: f"inference/{scenario}.jsonl" for scenario in scenarios
        },
    }


def validate_specialized_runners(
    config: Mapping[str, Any], scenarios: Sequence[str]
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    if "easy_turn" in scenarios:
        from easy_turn.adapter import validate_easy_turn_runner

        errors.extend(validate_easy_turn_runner(config))
    return errors


def run_specialized_runner(
    scenario: str,
    config: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
    samples: Sequence[SampleSpec],
) -> List[Dict[str, Any]]:
    if scenario == "easy_turn":
        from easy_turn.adapter import run_easy_turn_runner

        sample_counts = Counter(
            (sample.language, sample.label) for sample in samples
        )
        return run_easy_turn_runner(config, run_dir, run_id, sample_counts)
    raise ValueError(f"No specialized inference runner registered for {scenario}")


def run_inference(
    config_path: str | Path,
    run_id: str,
    scenarios: Sequence[str] = SCENARIOS,
    limit_per_dataset: int = 0,
    checkpoint: Optional[str] = None,
    formal_scenario_evaluation: bool = False,
) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore and hyphen")
    invalid_scenarios = sorted(set(scenarios) - set(SUPPORTED_SCENARIOS))
    if invalid_scenarios:
        raise ValueError(f"Unknown scenarios: {invalid_scenarios}")
    if formal_scenario_evaluation and (
        tuple(scenarios) != ("easy_turn",) or limit_per_dataset != 0
    ):
        raise ValueError(
            "formal_scenario_evaluation currently requires the complete "
            "easy_turn scenario with no sample limit"
        )
    config = load_eval_config(config_path)
    if checkpoint:
        config["model"]["checkpoint"] = checkpoint
    background_noise_max_duration_sec = float(
        config.get("evaluation", {}).get("background_noise_max_duration_sec", 20.0)
    )
    background_noise_trailing_silence_sec = float(
        config.get("evaluation", {}).get(
            "background_noise_trailing_silence_sec", 2.0
        )
    )
    if background_noise_max_duration_sec <= 0:
        raise ValueError(
            "evaluation.background_noise_max_duration_sec must be positive"
        )
    if background_noise_trailing_silence_sec < 0:
        raise ValueError(
            "evaluation.background_noise_trailing_silence_sec must be non-negative"
        )
    output_root = Path(config["_output_root"])
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "inference").mkdir()
    (run_dir / "evaluation").mkdir()
    errors_path = run_dir / "errors.jsonl"
    manifest_path = run_dir / "manifest.json"

    samples, errors, stream_settings = discover_all_samples(
        config, scenarios, limit_per_dataset=limit_per_dataset
    )
    errors.extend(validate_model_assets(config))
    errors.extend(validate_specialized_runners(config, scenarios))
    manifest = _build_manifest(
        config,
        run_id,
        scenarios,
        samples,
        stream_settings,
        limit_per_dataset,
        background_noise_max_duration_sec,
        background_noise_trailing_silence_sec,
        formal_scenario_evaluation,
    )
    atomic_write_jsonl(errors_path, errors)
    if errors:
        manifest.update(
            {
                "status": "preflight_failed",
                "updated_at": utc_now(),
                "error_count": len(errors),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise RuntimeError(
            f"Preflight failed with {len(errors)} error(s); see {errors_path}"
        )

    project_root = Path(config["_project_root"])
    deployment_samples = [
        sample
        for sample in samples
        if sample.scenario not in SPECIALIZED_RUNNER_SCENARIOS
    ]
    runtime_configs = (
        prepare_runtime_configs(config, run_dir) if deployment_samples else {}
    )
    manifest["model"]["runtime_configs"] = {
        language: relative_display(path, project_root)
        for language, path in runtime_configs.items()
    }
    manifest.update({"status": "inference_running", "updated_at": utc_now()})
    atomic_write_json(manifest_path, manifest)
    rows_by_scenario: Dict[str, List[Dict[str, Any]]] = {
        scenario: [] for scenario in scenarios
    }
    os.chdir(project_root)
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if deployment_samples:
            from service.model import TurnModel

            samples_by_language: Dict[str, List[SampleSpec]] = defaultdict(list)
            for sample in deployment_samples:
                samples_by_language[sample.language].append(sample)
            for language in LANGUAGES:
                language_samples = samples_by_language.get(language, [])
                if not language_samples:
                    continue
                print(
                    f"Loading {language} model for {len(language_samples)} samples",
                    flush=True,
                )
                model = TurnModel(config_path=str(runtime_configs[language]))
                settings = stream_settings[language]
                chunk_size = int(settings["chunk_size"])
                sample_rate = int(settings["sample_rate"])
                warm_up_model(
                    model,
                    language_samples[0],
                    chunk_size,
                    sample_rate,
                    max_duration_sec=(
                        background_noise_max_duration_sec
                        if language_samples[0].scenario == "background_noise"
                        else None
                    ),
                    trailing_silence_sec=(
                        background_noise_trailing_silence_sec
                        if language_samples[0].scenario == "background_noise"
                        else 0.0
                    ),
                )
                for index, sample in enumerate(language_samples, 1):
                    row = infer_sample(
                        model, sample, chunk_size, sample_rate, project_root,
                        max_duration_sec=(
                            background_noise_max_duration_sec
                            if sample.scenario == "background_noise"
                            else None
                        ),
                        trailing_silence_sec=(
                            background_noise_trailing_silence_sec
                            if sample.scenario == "background_noise"
                            else 0.0
                        ),
                    )
                    rows_by_scenario[sample.scenario].append(row)
                    print(
                        f"[{language}] {index}/{len(language_samples)} "
                        f"{sample.sample_id}",
                        flush=True,
                    )
                del model
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        for scenario in SPECIALIZED_RUNNER_SCENARIOS:
            scenario_samples = [
                sample for sample in samples if sample.scenario == scenario
            ]
            if not scenario_samples:
                continue
            print(
                f"Loading {scenario} specialized runner for "
                f"{len(scenario_samples)} samples",
                flush=True,
            )
            rows_by_scenario[scenario].extend(
                run_specialized_runner(
                    scenario, config, run_dir, run_id, scenario_samples
                )
            )
        for scenario, rows in rows_by_scenario.items():
            atomic_write_jsonl(
                run_dir / "inference" / f"{scenario}.jsonl",
                sorted(rows, key=lambda row: row["sample_id"]),
            )
    except Exception as exc:
        inference_error = error_record(
            "inference_failed",
            str(exc),
            traceback=traceback.format_exc(),
        )
        atomic_write_jsonl(errors_path, [inference_error])
        for scenario, rows in rows_by_scenario.items():
            if rows:
                atomic_write_jsonl(
                    run_dir / "inference" / f"{scenario}.jsonl", rows
                )
        manifest.update(
            {
                "status": "inference_failed",
                "updated_at": utc_now(),
                "error_count": 1,
                "complete": False,
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise

    formal_run = bool(manifest["formal_full_evaluation"]) or bool(
        manifest["formal_scenario_evaluation"]
    )
    manifest.update(
        {
            "status": "inference_complete" if formal_run else "subset_inference_complete",
            "updated_at": utc_now(),
            "error_count": 0,
            "complete": formal_run,
        }
    )
    atomic_write_json(manifest_path, manifest)
    atomic_write_jsonl(errors_path, [])
    print(f"Inference complete. Run directory: {run_dir}", flush=True)
    return run_dir


def replace_scenario_inference(
    config_path: str | Path,
    scenario: str,
    run_id: Optional[str] = None,
    run_dir_value: Optional[str | Path] = None,
) -> Path:
    """Replace one specialized scenario using the normal discovery/runner path."""
    if scenario not in SPECIALIZED_RUNNER_SCENARIOS:
        raise ValueError(f"Scenario does not have a replaceable runner: {scenario}")
    config = load_eval_config(config_path)
    if run_dir_value is not None:
        run_dir = Path(run_dir_value).expanduser().resolve()
    elif run_id:
        run_dir = (Path(config["_output_root"]) / run_id).resolve()
    else:
        raise ValueError("Provide exactly one of run_id or run_dir_value")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("error_count") != 0:
        raise RuntimeError("Scenario replacement requires a complete zero-error run")
    scenario_was_recorded = (
        scenario in manifest.get("selected_scenarios", [])
        or scenario in manifest.get("inference_files", {})
        or any(
            key == scenario or key.startswith(f"{scenario}/")
            for key in manifest.get("sample_counts", {})
        )
    )
    if not scenario_was_recorded:
        raise RuntimeError(f"Original run did not select scenario: {scenario}")

    stored_checkpoint = manifest.get("model", {}).get("checkpoint")
    if stored_checkpoint:
        config["model"]["checkpoint"] = stored_checkpoint
    samples, errors, _stream_settings = discover_all_samples(config, (scenario,))
    errors.extend(validate_model_assets(config))
    errors.extend(validate_specialized_runners(config, (scenario,)))
    if errors:
        replacement_errors = run_dir / f"{scenario}_replacement_errors.jsonl"
        atomic_write_jsonl(replacement_errors, errors)
        raise RuntimeError(
            f"{scenario} replacement preflight failed with {len(errors)} "
            f"error(s); see {replacement_errors}"
        )

    actual_counts = _sample_counts(samples)
    expected_counts = {
        key: int(value)
        for key, value in manifest.get("sample_counts", {}).items()
        if key == scenario or key.startswith(f"{scenario}/")
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"{scenario} replacement inventory differs from the original run: "
            f"{actual_counts} != {expected_counts}"
        )

    artifacts_root = run_dir / "artifacts"
    artifact_root = artifacts_root / scenario
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    if artifacts_root.is_dir() and not any(artifacts_root.iterdir()):
        artifacts_root.rmdir()
    legacy_artifact_root = run_dir / "inference" / "vad"
    if scenario == "easy_turn" and legacy_artifact_root.exists():
        shutil.rmtree(legacy_artifact_root)
    rows = run_specialized_runner(
        scenario,
        config,
        run_dir,
        str(manifest["run_id"]),
        samples,
    )
    target = run_dir / "inference" / f"{scenario}.jsonl"
    atomic_write_jsonl(target, rows)
    replacement_errors = run_dir / f"{scenario}_replacement_errors.jsonl"
    if replacement_errors.exists():
        replacement_errors.unlink()
    return target


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "eval_config.yaml"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument(
        "--checkpoint",
        help="Override model.checkpoint from the evaluation config for this inference run.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="Diagnostic subset only; a limited run is never marked complete.",
    )
    return parser


def run_cli(scenarios: Sequence[str], description: str) -> None:
    args = build_arg_parser(description).parse_args()
    if args.limit_per_dataset < 0:
        raise ValueError("--limit-per-dataset must be non-negative")
    run_inference(
        config_path=args.config,
        run_id=args.run_id,
        scenarios=tuple(scenarios),
        limit_per_dataset=args.limit_per_dataset,
        checkpoint=args.checkpoint,
    )
