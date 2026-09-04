"""Easy-Turn sample discovery for inference.

This module is independent from the model runtime and owns only input-dataset
validation and traversal order.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

EXPECTED_COUNTS = {
    ("en", "complete"): 318,
    ("en", "incomplete"): 299,
    ("zh", "complete"): 300,
    ("zh", "incomplete"): 300,
}

SAMPLE_ORDER = {
    "en": "bundled-release-order-v1",
    "zh": "official-list-v1",
}
ENGLISH_ORDER_ROOT = Path(__file__).resolve().with_name("orders")


@dataclass(frozen=True)
class EasyTurnSample:
    sample_id: str
    language: str
    label: str
    wav_path: Path


def _safe_dataset_path(dataset_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe dataset path: {relative}")
    resolved_root = dataset_root.resolve(strict=True)
    resolved = (resolved_root / relative).resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"dataset path escapes root: {relative}")
    return resolved


def discover_samples(
    dataset_root: Path,
    language: str,
    label: str,
) -> list[EasyTurnSample]:
    """Discover one class in the order used by the official inference script."""
    if language not in {"en", "zh"}:
        raise ValueError(f"unsupported language: {language}")
    if label not in {"complete", "incomplete"}:
        raise ValueError(f"unsupported label: {label}")
    root = dataset_root.resolve(strict=True)
    paths: list[Path] = []
    if language == "en":
        # The historical upstream runner used filesystem-dependent os.walk()
        # order.  Freeze the audited release order in a small bundled manifest
        # so inference does not depend on either directory-entry order or the
        # continued presence of the original release ZIP.
        order_file = ENGLISH_ORDER_ROOT / f"en_{label}.txt"
        if not order_file.is_file():
            raise FileNotFoundError(
                f"English Easy-Turn order manifest is missing: {order_file}"
            )
        for line_number, name in enumerate(
            order_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            name = name.strip()
            if not name:
                continue
            relative = Path(name)
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or relative.suffix.lower() != ".wav"
            ):
                raise ValueError(
                    f"invalid English order row {order_file}:{line_number}: {name}"
                )
            paths.append(_safe_dataset_path(root, Path(label) / relative))
    else:
        list_name = (
            "complete/complete_test.list"
            if label == "complete"
            else "incomplete/incomplete_real_test.list"
        )
        list_path = root / list_name
        if not list_path.is_file():
            raise FileNotFoundError(list_path)
        for line_number, line in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                relative = Path(record["wav"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid list row {list_path}:{line_number}") from exc
            path = _safe_dataset_path(root, Path(str(relative).removeprefix("./")))
            if path.relative_to(root).parts[0] != label:
                raise ValueError(
                    f"label mismatch in {list_path}:{line_number}: {relative}"
                )
            paths.append(path)

    expected = EXPECTED_COUNTS[(language, label)]
    if len(paths) != expected:
        raise ValueError(
            f"unexpected Easy-Turn inventory for {language}/{label}: "
            f"{len(paths)} != {expected}"
        )
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate WAV paths for {language}/{label}")

    samples = []
    seen_ids = set()
    for path in paths:
        # Keep identifiers compatible with the frozen Table-3 evidence bundle.
        sample_id = f"{language}:{label}:{path.stem}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        samples.append(EasyTurnSample(sample_id, language, label, path))
    return samples
