from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np


EVAL_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


infer_common = load_module("soulx_infer_common", EVAL_ROOT / "infer" / "common.py")
eval_common = load_module("soulx_metric_common", EVAL_ROOT / "eval" / "common.py")
stage3_search = load_module(
    "soulx_stage3_search",
    EVAL_ROOT.parent / "scripts" / "search_stage3_loss_weights.py",
)


def write_wav(path: Path, duration_sec: float = 3.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def state_event(
    index: int,
    center: float,
    current: float,
    public: str = "idle",
    raw: str = "idle",
    upstream: str | None = None,
):
    return {
        "chunk_index": index,
        "chunk_start_sec": center - 0.08,
        "chunk_end_sec": center + 0.08,
        "chunk_center_sec": center,
        "process_time_sec": 0.01,
        "queue_delay_sec": 0.0,
        "virtual_process_start_sec": current - 0.01,
        "current_time_sec": current,
        "raw_state": raw,
        "raw_state_token": f"<|user_{raw}|>",
        "public_state": public,
        "upstream_state": upstream or public,
    }


def inference_row(
    scenario: str, events, label: str | None = None, language: str = "en"
):
    row = {
        "schema_version": 1,
        "sample_id": f"{scenario}/{language}/1",
        "scenario": scenario,
        "language": language,
        "label": label or scenario,
        "audio": {"normalized_duration_sec": 3.0},
        "state_events": events,
    }
    if scenario == "easy_turn":
        row["event"] = None
    else:
        row["event"] = {
            "start_sec": 1.0,
            "end_sec": 2.5,
            "interval": "[start,end)",
            "chunk_membership": "chunk_center",
        }
    return row


class DatasetAdapterTests(unittest.TestCase):
    def test_noise_inference_cap_uses_first_twenty_seconds(self):
        class FakeModel:
            def __init__(self):
                self.chunks = []

            def reset(self):
                pass

            def process(self, chunk):
                self.chunks.append(chunk.copy())
                state = "speak" if np.all(chunk == 0.0) else "idle"
                return {"raw_state": state, "state": state}

        sample = infer_common.SampleSpec(
            sample_id="background_noise/neutral/long",
            scenario="background_noise",
            language="neutral",
            label="background_noise",
            audio_path=Path("long.wav"),
            timestamp_path=None,
            timestamp_source=None,
            event_start_sec=0.0,
            event_end_sec=30.0,
            duration_sec=30.0,
            original_sample_rate=100,
            original_frames=3000,
        )
        metadata = {
            "source_duration_sec": 30.0,
            "normalized_duration_sec": 20.0,
            "truncated": True,
        }
        with mock.patch.object(
            infer_common,
            "read_and_normalize_audio",
            return_value=(np.ones(2000, dtype=np.float32), metadata),
        ) as reader:
            model = FakeModel()
            row = infer_common.infer_sample(
                model,
                sample,
                100,
                100,
                Path.cwd(),
                max_duration_sec=20.0,
                trailing_silence_sec=2.0,
            )
        reader.assert_called_once_with(sample.audio_path, 100, max_duration_sec=20.0)
        self.assertEqual(row["event"]["end_sec"], 20.0)
        self.assertEqual(row["audio"]["normalized_duration_sec"], 20.0)
        self.assertEqual(row["audio"]["inference_duration_sec"], 22.0)
        self.assertEqual(row["audio"]["trailing_silence_sec"], 2.0)
        self.assertEqual(row["audio"]["trailing_silence_samples"], 200)
        self.assertEqual(len(row["state_events"]), 22)
        self.assertTrue(np.all(model.chunks[-1] == 0.0))
        self.assertTrue(row["audio"]["truncated"])
        result = eval_common.evaluate_record(row, 1.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_speak_chunk_index"], 20)

    def test_optional_speech_scenarios_are_not_formal(self):
        self.assertNotIn("background_speech", infer_common.SCENARIOS)
        self.assertIn("background_speech", infer_common.OPTIONAL_SCENARIOS)
        self.assertIn("background_speech", infer_common.SUPPORTED_SCENARIOS)
        self.assertNotIn("talking_to_other", infer_common.SCENARIOS)
        self.assertIn("talking_to_other", infer_common.OPTIONAL_SCENARIOS)
        self.assertIn("talking_to_other", infer_common.SUPPORTED_SCENARIOS)

    def test_vad_only_manifest_is_formal_but_not_full_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            config_path = root / "eval_config.yaml"
            config_path.write_text("version: 1\n", encoding="utf-8")
            config = {
                "_project_root": str(root),
                "_config_path": str(config_path),
                "model": {
                    "checkpoint": str(checkpoint),
                    "far_field_threshold": 0.02,
                },
                "not_evaluated": [],
            }
            manifest = infer_common._build_manifest(
                config,
                "vad-only-test",
                ("easy_turn",),
                [],
                {},
                0,
                20.0,
                2.0,
                formal_scenario_evaluation=True,
            )
        self.assertFalse(manifest["formal_full_evaluation"])
        self.assertTrue(manifest["formal_scenario_evaluation"])
        self.assertEqual(manifest["evaluation_scope"], "vad_only")

    def test_riff_metadata_parser_supports_float_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float.wav"
            frames = 160
            data = b"\x00" * (frames * 4)
            fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
            riff_size = 4 + 8 + len(fmt) + 8 + len(data)
            path.write_bytes(
                b"RIFF"
                + struct.pack("<I", riff_size)
                + b"WAVEfmt "
                + struct.pack("<I", len(fmt))
                + fmt
                + b"data"
                + struct.pack("<I", len(data))
                + data
            )
            sample_rate, parsed_frames, duration = infer_common._riff_wave_info(path)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(parsed_frames, frames)
            self.assertAlmostEqual(duration, 0.01)

    def test_both_timestamp_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"
            metadata.write_text('{"timestamps": [1.25, 2.5]}', encoding="utf-8")
            interrupt = root / "interrupt.json"
            interrupt.write_text('[{"timestamp": [0.5, 1.75]}]', encoding="utf-8")
            self.assertEqual(
                infer_common.parse_event_timestamp(metadata, "metadata.timestamps"),
                (1.25, 2.5),
            )
            self.assertEqual(
                infer_common.parse_event_timestamp(
                    interrupt, "interrupt.json[0].timestamp"
                ),
                (0.5, 1.75),
            )

    def test_fdb_discovers_only_exact_input_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "nested" / "1"
            write_wav(sample / "input.wav")
            write_wav(sample / "clean_input.wav")
            write_wav(sample / "context.wav")
            (sample / "metadata.json").write_text(
                '{"timestamps": [1.0, 2.0]}', encoding="utf-8"
            )
            rows, errors = infer_common.discover_fdb_samples(
                "backchannel", "en", root, 0.16
            )
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].audio_path.name, "input.wav")

    def test_english_background_window_extends_from_onset_to_clip_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "background_speech" / "1"
            write_wav(sample / "input.wav", duration_sec=6.0)
            (sample / "metadata.json").write_text(
                '{"timestamps": [2.0, 3.5]}', encoding="utf-8"
            )
            rows, errors = infer_common.discover_fdb_samples(
                "background_speech", "en", root, 0.16
            )
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0].event_start_sec, 2.0)
            self.assertAlmostEqual(rows[0].event_end_sec, 6.0)
            self.assertEqual(
                rows[0].timestamp_source,
                "metadata.timestamps[0] to audio_duration",
            )

    def test_chinese_background_uses_second_segment_onset_and_ignores_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_wav(root / "sample.wav", duration_sec=8.0)
            write_wav(root / "clean_sample.wav", duration_sec=3.0)
            write_wav(root / "three_segments.wav", duration_sec=8.0)
            (root / "sample.json").write_text(
                json.dumps(
                    {
                        "final_duration": 8.0,
                        "speech_segments": [
                            {"xmin": 0.1, "xmax": 1.0, "text": "user"},
                            {"xmin": 3.25, "xmax": 4.5, "text": "background"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "three_segments.json").write_text(
                json.dumps(
                    {
                        "final_duration": 8.0,
                        "speech_segments": [
                            {"xmin": 0.1, "xmax": 1.0, "text": "user"},
                            {"xmin": 3.0, "xmax": 4.0, "text": "background"},
                            {"xmin": 3.5, "xmax": 5.0, "text": "extra"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows, errors = infer_common.discover_humdial_background_speech_samples(
                "zh", root, 0.16
            )
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sample_id, "background_speech/zh/sample")
            self.assertAlmostEqual(rows[0].event_start_sec, 3.25)
            self.assertAlmostEqual(rows[0].event_end_sec, 8.0)
            self.assertEqual(
                rows[0].timestamp_source,
                "speech_segments[1].xmin to audio_duration",
            )

    def test_chinese_interruption_adapter_and_easy_turn_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interruption = root / "interruption" / "1"
            write_wav(interruption / "input.wav")
            (interruption / "interrupt.json").write_text(
                '[{"timestamp": [1.0, 2.0]}]', encoding="utf-8"
            )
            rows, errors = infer_common.discover_fdb_samples(
                "interruption", "zh", root / "interruption", 0.16
            )
            self.assertEqual(errors, [])
            self.assertEqual(rows[0].timestamp_source, "interrupt.json[0].timestamp")

            easy = root / "easy"
            write_wav(easy / "complete" / "a.wav", duration_sec=0.5)
            write_wav(easy / "incomplete" / "b.wav", duration_sec=0.5)
            easy_rows, easy_errors = infer_common.discover_easy_turn_samples(
                "zh", easy
            )
            self.assertEqual(easy_errors, [])
            self.assertEqual({row.label for row in easy_rows}, {"complete", "incomplete"})

    def test_timestamp_bounds_and_empty_window_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "1"
            write_wav(sample / "input.wav", duration_sec=1.0)
            (sample / "metadata.json").write_text(
                '{"timestamps": [1.0, 1.2]}', encoding="utf-8"
            )
            rows, errors = infer_common.discover_fdb_samples(
                "backchannel", "en", root, 0.16
            )
            self.assertEqual(rows, [])
            self.assertEqual(errors[0]["code"], "timestamp_out_of_bounds")

    def test_humdial_talking_to_other_uses_third_segment_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_wav(root / "sample_add.wav", duration_sec=8.0)
            write_wav(root / "clean_sample_add.wav", duration_sec=6.0)
            (root / "sample_add.json").write_text(
                json.dumps(
                    {
                        "final_duration": 8.0,
                        "speech_segments": [
                            {"xmin": 0.1, "xmax": 1.0, "text": "first"},
                            {"xmin": 2.0, "xmax": 3.0, "text": "second"},
                            {"xmin": 4.0, "xmax": 5.25, "text": "other"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "clean_sample_add.json").write_text(
                json.dumps(
                    {
                        "final_duration": 6.0,
                        "speech_segments": [
                            {"xmin": 0.1, "xmax": 1.0, "text": "second"},
                            {"xmin": 2.0, "xmax": 3.0, "text": "other"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows, errors = infer_common.discover_humdial_talking_to_other_samples(
                "zh", root, 0.16
            )
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sample_id, "talking_to_other/zh/sample_add")
            self.assertEqual(rows[0].timestamp_source, "speech_segments[2].xmax")
            self.assertAlmostEqual(rows[0].event_start_sec, 5.25)
            self.assertAlmostEqual(rows[0].event_end_sec, 8.0)

    def test_sid_noise_adapter_requires_pure_noise_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_root = root / "silence_or_noise"
            write_wav(audio_root / "noise.wav", duration_sec=2.0)
            (root / "silence_noise_test.jsonl").write_text(
                json.dumps(
                    {
                        "audio": "noise.wav",
                        "total_nonbreak": True,
                        "duration": 2.0,
                        "break_time": 2.0,
                        "text_with_break": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows, errors = infer_common.discover_sid_noise_samples("neutral", root)
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sample_id, "background_noise/neutral/noise")
            self.assertEqual(rows[0].event_start_sec, 0.0)
            self.assertAlmostEqual(rows[0].event_end_sec, 2.0)


class TimingTests(unittest.TestCase):
    def test_chunk_center_half_open_window(self):
        self.assertTrue(infer_common.chunk_center_in_window(0.08, 0.09, 1.0, 0.16))
        self.assertFalse(infer_common.chunk_center_in_window(0.0, 0.08, 1.0, 0.16))

    def test_virtual_time_without_and_with_backlog(self):
        start, queue, current = infer_common.advance_virtual_time(0.0, 0.16, 0.03)
        self.assertAlmostEqual(start, 0.16)
        self.assertAlmostEqual(queue, 0.0)
        self.assertAlmostEqual(current, 0.19)
        start, queue, current = infer_common.advance_virtual_time(0.40, 0.32, 0.05)
        self.assertAlmostEqual(start, 0.40)
        self.assertAlmostEqual(queue, 0.08)
        self.assertAlmostEqual(current, 0.45)

    def test_fake_turn_model_stream_schema_and_padding(self):
        class FakeModel:
            device = "cpu"

            def __init__(self):
                self.index = 0

            def process(self, chunk):
                states = [
                    ("backchannel", "idle"),
                    ("nonidle", "nonidle"),
                ]
                raw, public = states[self.index]
                self.index += 1
                return {
                    "raw_state": raw,
                    "raw_state_token": f"<|user_{raw}|>",
                    "state": public,
                }

        times = iter([0.0, 0.01, 0.01, 0.21])
        events, padding = infer_common.stream_audio(
            FakeModel(),
            np.zeros(3000, dtype=np.float32),
            chunk_size=2560,
            sample_rate=16000,
            timer=lambda: next(times),
            synchronizer=lambda: None,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(padding, 2120)
        self.assertEqual(events[0]["raw_state"], "backchannel")
        self.assertEqual(events[1]["public_state"], "nonidle")
        self.assertAlmostEqual(events[0]["current_time_sec"], 0.17)
        self.assertAlmostEqual(events[1]["current_time_sec"], 0.52)


class MetricTests(unittest.TestCase):
    def test_interruption_excludes_pre_onset_and_tracks_strict_one_second(self):
        events = [
            state_event(0, 0.88, 0.97, public="nonidle", raw="nonidle"),
            state_event(1, 1.08, 1.999, public="nonidle", raw="nonidle"),
        ]
        passed = eval_common.evaluate_record(inference_row("interruption", events), 1.0)
        self.assertTrue(passed["passed"])
        self.assertTrue(passed["within_latency_threshold"])
        self.assertAlmostEqual(passed["detection_latency_sec"], 0.999)
        self.assertEqual(passed["first_nonidle_chunk_index"], 1)

        events[1] = state_event(1, 1.08, 2.0, public="nonidle", raw="nonidle")
        exact = eval_common.evaluate_record(inference_row("interruption", events), 1.0)
        self.assertTrue(exact["passed"])
        self.assertFalse(exact["within_latency_threshold"])
        self.assertEqual(exact["outcome"], "late")

    def test_interruption_late_and_miss(self):
        late = eval_common.evaluate_record(
            inference_row(
                "interruption",
                [state_event(1, 1.2, 2.2, public="nonidle", raw="nonidle")],
            ),
            1.0,
        )
        miss = eval_common.evaluate_record(
            inference_row("interruption", [state_event(1, 1.2, 1.3)]), 1.0
        )
        self.assertEqual(late["outcome"], "late")
        self.assertTrue(late["passed"])
        self.assertFalse(late["within_latency_threshold"])
        self.assertEqual(miss["outcome"], "miss")
        self.assertFalse(miss["passed"])
        summary = eval_common.interruption_latency_summary([late, miss])
        self.assertEqual(summary["detected"], 1)
        self.assertEqual(summary["missed"], 1)
        self.assertEqual(summary["detection_rate"], 0.5)
        self.assertEqual(summary["on_time_rate"], 0.0)
        self.assertAlmostEqual(summary["mean_sec"], 1.2)

    def test_backchannel_requires_raw_detection_and_no_public_nonidle(self):
        accepted = eval_common.evaluate_record(
            inference_row(
                "backchannel",
                [state_event(1, 1.2, 1.3, public="idle", raw="backchannel")],
            ),
            1.0,
        )
        false_interrupt = eval_common.evaluate_record(
            inference_row(
                "backchannel",
                [state_event(1, 1.2, 1.3, public="nonidle", raw="backchannel")],
            ),
            1.0,
        )
        self.assertTrue(accepted["passed"])
        self.assertFalse(false_interrupt["passed"])

    def test_background_rejects_only_speak_and_easy_turn_accepts_nonidle(self):
        background = eval_common.evaluate_record(
            inference_row(
                "background_speech",
                [
                    state_event(1, 1.2, 1.3, public="nonidle", raw="nonidle"),
                    state_event(2, 1.36, 1.46, public="blank"),
                ],
            ),
            1.0,
        )
        false_speak = eval_common.evaluate_record(
            inference_row(
                "background_speech",
                [state_event(1, 1.2, 1.3, public="speak", raw="complete")],
            ),
            1.0,
        )
        easy = eval_common.evaluate_record(
            inference_row(
                "easy_turn",
                [state_event(0, 0.08, 0.17), state_event(1, 0.24, 0.33, public="nonidle", raw="nonidle")],
                label="complete",
            ),
            1.0,
        )
        self.assertTrue(background["passed"])
        self.assertFalse(false_speak["passed"])
        self.assertEqual(false_speak["outcome"], "false_speak")
        self.assertTrue(easy["passed"])

    def test_talking_to_other_rejects_only_speak_after_third_segment(self):
        accepted = eval_common.evaluate_record(
            inference_row(
                "talking_to_other",
                [
                    state_event(0, 0.8, 0.9, public="speak", raw="complete"),
                    state_event(1, 1.2, 1.3, public="nonidle", raw="nonidle"),
                    state_event(2, 1.4, 1.5, public="idle", raw="idle"),
                ],
            ),
            1.0,
        )
        false_speak = eval_common.evaluate_record(
            inference_row(
                "talking_to_other",
                [state_event(1, 1.2, 1.3, public="speak", raw="complete")],
            ),
            1.0,
        )
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["outcome"], "rejected")
        self.assertFalse(false_speak["passed"])
        self.assertEqual(false_speak["outcome"], "false_speak")
        self.assertEqual(false_speak["first_speak_chunk_index"], 1)

    def test_background_noise_rejects_any_speak_across_whole_clip(self):
        accepted = eval_common.evaluate_record(
            inference_row(
                "background_noise",
                [
                    state_event(0, 0.08, 0.17, public="idle", raw="idle"),
                    state_event(1, 0.24, 0.33, public="nonidle", raw="nonidle"),
                ],
            ),
            1.0,
        )
        false_speak = eval_common.evaluate_record(
            inference_row(
                "background_noise",
                [state_event(0, 0.08, 0.17, public="speak", raw="complete")],
            ),
            1.0,
        )
        self.assertTrue(accepted["passed"])
        self.assertFalse(false_speak["passed"])
        self.assertEqual(false_speak["outcome"], "false_speak")

    def test_weighted_rate_uses_samples_not_category_means(self):
        rows = [{"passed": True}] * 9 + [{"passed": False}] + [{"passed": False}] * 2
        summary = eval_common.rate_summary(rows)
        self.assertEqual(summary["passed"], 9)
        self.assertEqual(summary["total"], 12)
        self.assertAlmostEqual(summary["rate"], 0.75)

    def test_language_macro_does_not_weight_larger_language_split_more(self):
        rows = (
            [{"language": "en", "passed": True}] * 2
            + [{"language": "en", "passed": False}] * 8
            + [{"language": "zh", "passed": True}] * 90
            + [{"language": "zh", "passed": False}] * 10
        )
        macro = eval_common.language_macro_rate_summary(rows)
        self.assertEqual(macro["aggregation"], "macro_average_by_language")
        self.assertAlmostEqual(macro["components"]["en"]["rate"], 0.2)
        self.assertAlmostEqual(macro["components"]["zh"]["rate"], 0.9)
        self.assertAlmostEqual(macro["rate"], 0.55)
        self.assertAlmostEqual(eval_common.rate_summary(rows)["rate"], 0.8363636363636363)

    def test_hierarchical_macro_equal_weights_scenarios_after_languages(self):
        backchannel = [{"language": "en", "passed": False}] * 2 + [{"language": "zh", "passed": True}] * 20
        noise = [{"language": "neutral", "passed": True}] * 500
        macro = eval_common.scenario_language_macro_rate_summary(
            {"backchannel": backchannel, "background_noise": noise}
        )
        self.assertEqual(
            macro["aggregation"],
            "hierarchical_macro_average_by_scenario_then_language",
        )
        self.assertAlmostEqual(macro["components"]["backchannel"]["rate"], 0.5)
        self.assertAlmostEqual(macro["components"]["background_noise"]["rate"], 1.0)
        self.assertAlmostEqual(macro["rate"], 0.75)

    def test_inference_parser_accepts_checkpoint_override(self):
        parser = infer_common.build_arg_parser("test")
        args = parser.parse_args(
            ["--run-id", "checkpoint-test", "--checkpoint", "ckpt/model.ckpt"]
        )
        self.assertEqual(args.checkpoint, "ckpt/model.ckpt")

    def test_incomplete_manifest_blocks_formal_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "manifest.json").write_text(
                json.dumps({"complete": False, "status": "inference_failed", "error_count": 1}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                eval_common.load_complete_manifest(run_dir)

    def test_vad_only_manifest_completes_after_vad_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics_path = run_dir / "evaluation" / "vad" / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text("{}\n", encoding="utf-8")
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "evaluation_scope": "vad_only",
                        "status": "inference_complete",
                        "evaluations": {},
                    }
                ),
                encoding="utf-8",
            )
            eval_common.update_manifest_evaluation(run_dir, "vad", metrics_path)
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["status"], "evaluation_complete")

    def test_stage3_search_reads_formal_vad_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "scene": "vad_easy_turn",
                        "aggregates": {
                            "complete_accuracy": {"rate": 0.8},
                            "incomplete_accuracy": {"rate": 0.7},
                            "avg_accuracy": 0.75,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = stage3_search.load_metrics(path)
        self.assertEqual(metrics["acc_complete"], 0.8)
        self.assertEqual(metrics["acc_incomplete"], 0.7)
        self.assertEqual(metrics["avg_acc"], 0.75)
        self.assertAlmostEqual(metrics["incomplete_error_rate"], 0.3)

    def test_stage3_diagnostics_read_framework_state_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "easy_turn.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "state_events": [
                            {"raw_state": "nonidle", "public_state": "nonidle"},
                            {"raw_state": "complete", "public_state": "speak"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            diagnostics = stage3_search.prediction_diagnostics(path)
        self.assertEqual(diagnostics["num_prediction_samples"], 1)
        self.assertEqual(diagnostics["raw_complete_chunks"], 1)
        self.assertEqual(diagnostics["raw_complete_but_not_speak_rate"], 0.0)


class EndToEndEvidenceTests(unittest.TestCase):
    def test_saved_jsonl_generates_both_reports_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "output" / "fake-run"
            inference_dir = run_dir / "inference"
            inference_dir.mkdir(parents=True)
            config = {
                "paths": {"project_root": ".", "output_root": "output"},
                "evaluation": {
                    "interruption_latency_threshold_sec": 1.0,
                    "thresholds": {
                        "interruption_effective_false_rejection_lt": 0.02,
                        "interruption_invalid_rejection_gt": 0.85,
                        "rejection_effective_false_rejection_lt": 0.02,
                        "rejection_invalid_rejection_gt": 0.90,
                        "vad_pause_accuracy_gt": 0.90,
                        "vad_normal_endpoint_accuracy_gt": 0.90,
                    },
                    "vad_endpoint_latency_threshold_sec": 0.3,
                    "vad_endpoint_annotations": "annotations.jsonl",
                },
                "not_evaluated": [],
            }
            config_path = root / "eval_config.yaml"
            config_path.write_text(
                __import__("yaml").safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            (root / "annotations.jsonl").write_text(
                json.dumps(
                    {
                        "oracle": "silero_vad_oracle_v1",
                        "sample_id": "easy_turn/en/1",
                        "label": "complete",
                        "status": "ok",
                        "speech_end_sample": 16000,
                        "speech_end_sec": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = {
                "interruption": inference_row(
                    "interruption",
                    [state_event(1, 1.2, 1.4, public="nonidle", raw="nonidle")],
                ),
                "backchannel": inference_row(
                    "backchannel",
                    [state_event(1, 1.2, 1.4, public="idle", raw="backchannel")],
                ),
                "background_speech": inference_row(
                    "background_speech", [state_event(1, 1.2, 1.4)]
                ),
                "background_noise": inference_row(
                    "background_noise",
                    [state_event(1, 1.2, 1.4)],
                    language="neutral",
                ),
                "easy_turn": inference_row(
                    "easy_turn",
                    [state_event(0, 0.08, 0.17, public="nonidle", raw="complete")],
                    label="complete",
                ),
            }
            for scenario, row in rows.items():
                (inference_dir / f"{scenario}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
            easy_turn_incomplete = inference_row(
                "easy_turn",
                [state_event(0, 0.08, 0.17, public="speak", raw="incomplete")],
                label="incomplete",
            )
            easy_turn_incomplete["sample_id"] = "easy_turn/en/2"
            with (inference_dir / "easy_turn.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(easy_turn_incomplete) + "\n")
            vad_rows = []
            for language in ("en", "zh"):
                for label in ("complete", "incomplete"):
                    legacy_event = state_event(
                        0,
                        0.08,
                        0.17,
                        public="nonidle",
                        raw=label,
                    )
                    legacy_event.pop("upstream_state")
                    row = inference_row(
                        "easy_turn",
                        [legacy_event],
                        label=label, language=language,
                    )
                    row["sample_id"] = f"easy_turn/{language}/{label}/1"
                    vad_rows.append(row)
            (inference_dir / "easy_turn.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in vad_rows), encoding="utf-8"
            )
            manifest = {
                "run_id": "fake-run",
                "complete": True,
                "status": "inference_complete",
                "error_count": 0,
                "sample_counts": {
                    "interruption/en": 1,
                    "backchannel/en": 1,
                    "background_speech/en": 1,
                    "background_noise/neutral": 1,
                    "easy_turn/en/complete": 1,
                    "easy_turn/en/incomplete": 1,
                    "easy_turn/zh/complete": 1,
                    "easy_turn/zh/incomplete": 1,
                },
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            scripts = [
                EVAL_ROOT / "eval" / "eval_interruption.py",
                EVAL_ROOT / "eval" / "eval_rejection.py",
                EVAL_ROOT / "eval" / "eval_vad.py",
            ]
            for script in scripts:
                subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--config",
                        str(config_path),
                        "--run-dir",
                        str(run_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            metrics_path = run_dir / "evaluation" / "interruption" / "metrics.json"
            first_metrics = metrics_path.read_text(encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(scripts[0]),
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first_metrics, metrics_path.read_text(encoding="utf-8"))
            self.assertTrue(
                (run_dir / "evaluation" / "interruption" / "report.md").is_file()
            )
            self.assertTrue(
                (run_dir / "evaluation" / "rejection" / "report.md").is_file()
            )
            rejection_metrics = json.loads(
                (run_dir / "evaluation" / "rejection" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            effective = rejection_metrics["aggregates"][
                "effective_intent_acceptance_rate"
            ]
            self.assertEqual(effective["total"], 3)
            self.assertNotIn(
                "background_speech/en", rejection_metrics["per_category"]
            )
            self.assertNotIn(
                "background_speech",
                rejection_metrics["aggregates"]["invalid_intent_rejection_rate"][
                    "components"
                ],
            )
            self.assertEqual(effective["passed"], 3)
            self.assertEqual(effective["aggregation"], "hierarchical_macro_average_by_scenario_then_language")
            self.assertTrue((run_dir / "evaluation" / "vad" / "report.md").is_file())
            vad_metrics = json.loads(
                (run_dir / "evaluation" / "vad" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(vad_metrics["protocol"]["primary_rule"], "last-terminal-v1")
            self.assertEqual(
                vad_metrics["aggregates"]["complete_accuracy"]["aggregation"],
                "macro_average_by_language",
            )
            self.assertIn("sample_weighted_aggregates", vad_metrics)
            self.assertEqual(vad_metrics["aggregates"]["complete_accuracy"]["passed"], 2)
            self.assertEqual(vad_metrics["aggregates"]["incomplete_accuracy"]["passed"], 2)
            vad_report = (
                run_dir / "evaluation" / "vad" / "report.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Run ID: `fake-run`", vad_report)
            self.assertIn("## Terminal-state protocol", vad_report)
            self.assertIn("## Per-category results", vad_report)
            self.assertIn("| Category | Correct | Incorrect | Total | Accuracy |", vad_report)
            self.assertIn("## Primary language-macro metrics", vad_report)
            self.assertIn("## Sample-weighted reference metrics", vad_report)
            self.assertIn("## Technical-target checks", vad_report)
            final_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_manifest["status"], "evaluation_complete")


class SpecializedRunnerIntegrationTests(unittest.TestCase):
    def test_easy_turn_readout_uses_last_terminal_state(self):
        previous_common = sys.modules.get("common")
        sys.modules["common"] = eval_common
        try:
            eval_vad = load_module(
                "soulx_eval_vad_unit", EVAL_ROOT / "eval" / "eval_vad.py"
            )
        finally:
            if previous_common is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = previous_common

        readout = eval_vad.classify_state_events(
            [
                {"upstream_state": "speak"},
                {"upstream_state": "backchannel"},
                {"upstream_state": "wait"},
            ]
        )
        self.assertEqual(readout["prediction"], "incomplete")
        self.assertEqual(readout["terminal_count"], 2)
        self.assertEqual(readout["selected_terminal"]["upstream_state"], "wait")

        legacy_readout = eval_vad.classify_state_events(
            [
                {"public_state": "nonidle", "raw_state": "complete"},
                {"public_state": "nonidle", "raw_state": "incomplete"},
            ]
        )
        self.assertEqual(legacy_readout["prediction"], "incomplete")
        self.assertEqual(legacy_readout["terminal_count"], 2)
        self.assertEqual(
            legacy_readout["selected_terminal"]["resolved_terminal_state"],
            "wait",
        )

    def test_easy_turn_command_has_no_audit_artifact_arguments(self):
        infer_root = str(EVAL_ROOT / "infer")
        if infer_root not in sys.path:
            sys.path.insert(0, infer_root)
        from easy_turn.adapter import runner_command

        command = runner_command(
            Path("runner.py"),
            "run-1",
            "en",
            "complete",
            1,
            Path("dataset"),
            Path("official"),
            Path("config.yaml"),
            Path("asr"),
            Path("artifacts"),
        )
        self.assertNotIn("--trace-dir", command)
        self.assertNotIn("--asr-cache", command)
        self.assertNotIn("--continuation-checkpoint", command)
        self.assertIn("--diagnostic-limit", command)

    def test_easy_turn_runner_keeps_no_intermediate_artifact_directory(self):
        infer_root = str(EVAL_ROOT / "infer")
        if infer_root not in sys.path:
            sys.path.insert(0, infer_root)
        from easy_turn import adapter

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            runner_root = project_root / "infer" / "easy_turn"
            runner_root.mkdir(parents=True)
            (runner_root / "runner.py").write_text("", encoding="utf-8")
            (project_root / "asr" / "en").mkdir(parents=True)
            (project_root / "datasets" / "en").mkdir(parents=True)
            run_dir = project_root / "output" / "run-1"
            run_dir.mkdir(parents=True)
            config = {
                "_project_root": project_root,
                "model": {"checkpoint": "checkpoint.ckpt"},
                "datasets": {"easy_turn": {"en": "datasets/en"}},
                "scenario_runners": {
                    "easy_turn": {
                        "implementation": "easy_turn",
                        "runner_root": "infer/easy_turn",
                        "official_root": "official",
                        "asr_model_dirs": {"en": "asr/en"},
                    }
                },
            }
            temporary_work_roots = []

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                temporary_work_roots.append(output.parents[1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "records": [
                                {
                                    "sample_id": "easy_turn/en/complete/a",
                                    "wav_path": "a.wav",
                                    "audio_duration_seconds": 1.0,
                                    "trace": [
                                        {
                                            "state": "speak",
                                            "timestamp": [0.0, 0.16],
                                        }
                                    ],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(adapter.subprocess, "run", side_effect=fake_run):
                rows = adapter.run_easy_turn_runner(
                    config,
                    run_dir,
                    "run-1",
                    {("en", "complete"): 1},
                )

            self.assertEqual(len(rows), 1)
            self.assertNotIn("vad_readout", rows[0])
            self.assertNotIn("vad_protocol", rows[0])
            self.assertFalse((run_dir / "artifacts" / "easy_turn").exists())
            self.assertTrue(temporary_work_roots)
            self.assertTrue(all(not path.exists() for path in temporary_work_roots))

    def test_easy_turn_artifacts_can_recover_final_inference(self):
        infer_root = str(EVAL_ROOT / "infer")
        if infer_root not in sys.path:
            sys.path.insert(0, infer_root)
        from recover_easy_turn_artifacts import recover_easy_turn_inference

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run-1"
            results_dir = run_dir / "artifacts" / "easy_turn" / "results"
            results_dir.mkdir(parents=True)
            sample_counts = {}
            for language in ("en", "zh"):
                for label, state in (
                    ("complete", "speak"),
                    ("incomplete", "wait"),
                ):
                    sample_counts[f"easy_turn/{language}/{label}"] = 1
                    payload = {
                        "status": "running",
                        "records": [
                            {
                                "sample_id": f"easy_turn/{language}/{label}/a",
                                "wav_path": f"{language}_{label}.wav",
                                "audio_duration_seconds": 1.0,
                                "trace": [
                                    {"state": state, "timestamp": [0.0, 0.16]}
                                ],
                            }
                        ],
                    }
                    (results_dir / f"{language}_{label}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
            (run_dir / "manifest.json").write_text(
                json.dumps({"sample_counts": sample_counts}), encoding="utf-8"
            )
            target = run_dir / "inference" / "easy_turn.jsonl"
            target.parent.mkdir(parents=True)
            target.write_text("old evidence\n", encoding="utf-8")

            recovered = recover_easy_turn_inference(run_dir)

            rows = [
                json.loads(line)
                for line in recovered.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["state_events"][-1]["upstream_state"] for row in rows},
                {"speak", "wait"},
            )
            self.assertTrue((run_dir / "artifacts" / "easy_turn").is_dir())

    def test_easy_turn_dispatches_through_registered_framework_runner(self):
        infer_root = str(EVAL_ROOT / "infer")
        if infer_root not in sys.path:
            sys.path.insert(0, infer_root)
        from easy_turn import adapter

        samples = [
            infer_common.SampleSpec(
                sample_id="easy_turn/en/complete/a",
                scenario="easy_turn",
                language="en",
                label="complete",
                audio_path=Path("a.wav"),
                timestamp_path=None,
                timestamp_source=None,
                event_start_sec=None,
                event_end_sec=None,
                duration_sec=1.0,
                original_sample_rate=16000,
                original_frames=16000,
            ),
            infer_common.SampleSpec(
                sample_id="easy_turn/en/incomplete/b",
                scenario="easy_turn",
                language="en",
                label="incomplete",
                audio_path=Path("b.wav"),
                timestamp_path=None,
                timestamp_source=None,
                event_start_sec=None,
                event_end_sec=None,
                duration_sec=1.0,
                original_sample_rate=16000,
                original_frames=16000,
            ),
        ]
        expected = [{"sample_id": "runner-row"}]
        with mock.patch.object(
            adapter, "run_easy_turn_runner", return_value=expected
        ) as runner:
            actual = infer_common.run_specialized_runner(
                "easy_turn", {}, Path("run"), "run-1", samples
            )
        self.assertEqual(actual, expected)
        runner.assert_called_once_with(
            {},
            Path("run"),
            "run-1",
            {("en", "complete"): 1, ("en", "incomplete"): 1},
        )

    def test_easy_turn_is_part_of_the_normal_full_scenario_set(self):
        self.assertIn("easy_turn", infer_common.SCENARIOS)
        self.assertIn("easy_turn", infer_common.SPECIALIZED_RUNNER_SCENARIOS)
        run_all_source = (EVAL_ROOT / "infer" / "run_all.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_cli(SCENARIOS", run_all_source)
        shell_source = (EVAL_ROOT / "run_all.sh").read_text(encoding="utf-8")
        self.assertNotIn("infer_easy_turn.py", shell_source)


if __name__ == "__main__":
    unittest.main()
