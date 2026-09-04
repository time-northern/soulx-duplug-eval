"""Run SoulX-Duplug inference for one Easy-Turn language/label split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Callable, Sequence

from dataset import SAMPLE_ORDER, discover_samples


SENSEVOICE_EMOTION_MARKERS = (
    "😊",
    "😔",
    "😡",
    "😰",
    "🤢",
    "😮",
    "🎼",
    "👏",
    "😀",
    "😭",
    "🤧",
    "😷",
)
SENSEVOICE_EMOTION_PATTERN = re.compile(
    "[" + re.escape("".join(SENSEVOICE_EMOTION_MARKERS)) + "]"
)


def clean_sensevoice_text(value: str) -> str:
    """Apply the text cleanup used by the pinned official ASR wrapper."""
    if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]", value):
        return ""
    return SENSEVOICE_EMOTION_PATTERN.sub("", value)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Write a result without exposing a partially serialized JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ParaformerASR:
    """Use the configured local Paraformer model as the cascade ASR."""

    def __init__(self, model_dir: Path):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=str(model_dir),
            device="cuda",
            disable_pbar=True,
            disable_update=True,
        )

    def recognize(self, audio_chunk, sample_rate=16_000, **_kwargs):
        import numpy as np
        import soxr

        audio = np.asarray(audio_chunk)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            audio = soxr.resample(audio, sample_rate, 16_000)
        try:
            return self.pipeline(audio)[0]["text"].strip()
        except Exception as exc:
            # Preserve the fallback used by the official ASR wrapper.
            print(f"ASR recognition failed: {exc}")
            return ""


class SensevoiceASR:
    """Use the configured local SenseVoice model as the cascade ASR."""

    def __init__(self, model_dir: Path, language: str = "en"):
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self.model = AutoModel(
            model=str(model_dir),
            trust_remote_code=False,
            device="cuda",
            disable_pbar=True,
            disable_update=True,
        )
        self.language = language
        self.postprocess = rich_transcription_postprocess

    def recognize(self, audio_chunk, sample_rate=16_000, language=None, **_kwargs):
        import numpy as np
        import soxr

        audio = np.asarray(audio_chunk)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            audio = soxr.resample(audio, sample_rate, 16_000)
        try:
            value = self.postprocess(
                self.model.generate(
                    input=audio,
                    cache={},
                    language=language or self.language,
                    use_itn=True,
                    batch_size=16,
                )[0]["text"]
            ).strip()
            return clean_sensevoice_text(value)
        except Exception as exc:
            # Preserve the fallback used by the official ASR wrapper.
            print(f"ASR recognition failed: {exc}")
            return ""


class CachedASR:
    """Avoid recomputing identical cascade-ASR requests during one split."""

    def __init__(self, recognizer_factory: Callable[[], Any]) -> None:
        self.recognizer_factory = recognizer_factory
        self.recognizer = None
        self.values: dict[str, str] = {}

    def initialize(self) -> None:
        if self.recognizer is None:
            self.recognizer = self.recognizer_factory()

    def recognize(self, audio, sample_rate=16_000, **kwargs):
        import numpy as np

        array = np.ascontiguousarray(audio)
        digest = hashlib.sha256()
        digest.update(str(sample_rate).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        key = digest.hexdigest()
        if key not in self.values:
            self.initialize()
            text = self.recognizer.recognize(
                audio, sample_rate=sample_rate, **kwargs
            )
            if not isinstance(text, str):
                raise TypeError("cascade ASR returned non-text output")
            self.values[key] = text
        return self.values[key]


def load_upstream_module(official_root: Path):
    """Load the inference implementation from the configured training tree."""
    path = official_root / "scripts/duplex_inference.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(official_root))
    specification = importlib.util.spec_from_file_location(
        "easy_turn_upstream_duplex_inference", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import upstream inference: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_trace(trace: Any) -> list[dict[str, Any]]:
    """Validate the state sequence required by the VAD readout."""
    if not isinstance(trace, list) or not trace:
        raise ValueError("upstream state trace is empty or invalid")
    valid_states = {"speak", "wait", "backchannel", "idle", "nonidle", "unknown"}
    for index, item in enumerate(trace):
        if not isinstance(item, dict) or item.get("state") not in valid_states:
            raise ValueError(f"invalid state at trace index {index}: {item}")
        timestamp = item.get("timestamp")
        expected = [index * 0.16, (index + 1) * 0.16]
        if (
            not isinstance(timestamp, list)
            or len(timestamp) != 2
            or abs(float(timestamp[0]) - expected[0]) > 1e-8
            or abs(float(timestamp[1]) - expected[1]) > 1e-8
        ):
            raise ValueError(f"invalid timestamp at trace index {index}: {timestamp}")
    return trace


def evaluate_one(
    upstream: Any,
    cfg: Any,
    model: Any,
    asr: CachedASR,
    sample: Any,
    work_dir: Path,
    sequence_index: int,
) -> dict[str, Any]:
    """Run one WAV and return only fields consumed by the framework adapter."""
    import soundfile as sf

    safe_name = f"{sequence_index:04d}-{sample.wav_path.stem}"
    staged_wav = work_dir / f"{safe_name}.wav"
    state_path = work_dir / f"{safe_name}_states.json"
    staged_wav.symlink_to(sample.wav_path.resolve(strict=True))

    upstream.duplex_predict_160_cascade_asr(cfg, model, str(staged_wav), asr)
    if not state_path.is_file():
        raise RuntimeError(f"upstream did not write state trace: {state_path}")
    trace = validate_trace(json.loads(state_path.read_text(encoding="utf-8")))
    info = sf.info(sample.wav_path)
    return {
        "sample_id": sample.sample_id,
        "wav_path": str(sample.wav_path),
        "audio_duration_seconds": info.frames / info.samplerate,
        "sample_order": SAMPLE_ORDER[sample.language],
        "trace": trace,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Easy-Turn VAD language/label split."
    )
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--label", choices=("complete", "incomplete"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--diagnostic-limit",
        type=int,
        help="Run only a prefix of this split.",
    )
    parser.add_argument(
        "--diagnostic-start-index",
        type=int,
        default=0,
        help="Zero-based prefix start; requires --diagnostic-limit.",
    )
    return parser


def validate_runtime_config(cfg: Any, language: str) -> None:
    """Reject settings that would change the Easy-Turn inference protocol."""
    if int(cfg.infer_config.seed) != 42:
        raise RuntimeError("Easy-Turn seed must be 42")
    expected = {
        "chunk_size": 2560,
        "audio_back_size": 15360,
        "audio_ahead_size": 640,
        "sample_rate": 16000,
        "chunk_token_len_small": 2,
        "max_wait_num": 5,
        "max_mistake_num": 5,
        "single_round": False,
        "precision": "bf16",
        "enable_cascade_asr": True,
        "asr_model_name": "sensevoice" if language == "en" else "paraformer",
        "asr_language": language,
    }
    actual = {
        "chunk_size": int(cfg.infer_config.input.chunk_size),
        "audio_back_size": int(cfg.infer_config.input.audio_back_size),
        "audio_ahead_size": int(cfg.infer_config.input.audio_ahead_size),
        "sample_rate": int(cfg.infer_config.input.sample_rate),
        "chunk_token_len_small": int(cfg.infer_config.input.chunk_token_len_small),
        "max_wait_num": int(cfg.infer_config.max_wait_num),
        "max_mistake_num": int(cfg.infer_config.max_mistake_num),
        "single_round": bool(cfg.infer_config.single_round),
        "precision": str(cfg.infer_config.precision),
        "enable_cascade_asr": bool(cfg.model_config.enable_cascade_asr),
        "asr_model_name": str(cfg.infer_config.asr.model_name),
        "asr_language": str(cfg.infer_config.asr.language),
    }
    if actual != expected:
        raise RuntimeError(f"Easy-Turn config drift: {actual} != {expected}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.diagnostic_limit is not None and args.diagnostic_limit <= 0:
        raise ValueError("--diagnostic-limit must be positive")
    if args.diagnostic_start_index < 0:
        raise ValueError("--diagnostic-start-index must be non-negative")
    if args.diagnostic_limit is None and args.diagnostic_start_index != 0:
        raise ValueError(
            "--diagnostic-start-index is forbidden without --diagnostic-limit"
        )

    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    official_root = args.official_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    asr_model_dir = args.asr_model_dir.resolve(strict=True)
    all_samples = discover_samples(
        args.dataset_root,
        args.language,
        args.label,
    )
    if args.diagnostic_limit is None:
        samples = all_samples
    else:
        stop = args.diagnostic_start_index + args.diagnostic_limit
        if stop > len(all_samples):
            raise ValueError(
                "diagnostic slice exceeds class inventory: "
                f"{args.diagnostic_start_index}:{stop} > {len(all_samples)}"
            )
        samples = all_samples[args.diagnostic_start_index:stop]

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import WhisperFeatureExtractor

    upstream = load_upstream_module(official_root)
    from config.config import RunConfig
    from models.state_prediction_model import State_Prediction_Model

    cfg = OmegaConf.merge(RunConfig(), OmegaConf.load(config_path))
    validate_runtime_config(cfg, args.language)
    Path(cfg.model_config.init_ckpt_path_lora).resolve(strict=True)

    seed = int(cfg.infer_config.seed)
    upstream.pl.seed_everything(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = State_Prediction_Model(cfg)
    model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
        cfg.model_config.glm_tokenizer_path
    )
    model.eval().to("cuda")

    recognizer_factory: Callable[[], Any]
    if args.language == "zh":
        recognizer_factory = lambda: ParaformerASR(asr_model_dir)
    else:
        recognizer_factory = lambda: SensevoiceASR(asr_model_dir, "en")
    asr = CachedASR(recognizer_factory)
    asr.initialize()

    payload: dict[str, Any] = {
        "status": "running",
        "run_id": args.run_id,
        "language": args.language,
        "label": args.label,
        "records": [],
    }
    atomic_json_write(output, payload)

    with tempfile.TemporaryDirectory(
        prefix=f"easy-turn-{args.language}-{args.label}-"
    ) as temporary:
        work_dir = Path(temporary)
        for index, sample in enumerate(samples):
            record = evaluate_one(
                upstream, cfg, model, asr, sample, work_dir, index
            )
            payload["records"].append(record)
            atomic_json_write(output, payload)
            print(
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "progress": f"{index + 1}/{len(samples)}",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    payload["status"] = "complete"
    atomic_json_write(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
