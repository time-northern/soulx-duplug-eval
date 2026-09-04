from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER_ROOT = Path(__file__).resolve().parents[1] / "infer" / "easy_turn"
sys.path.insert(0, str(RUNNER_ROOT))

import dataset  # noqa: E402
import runner  # noqa: E402


class EasyTurnDatasetTests(unittest.TestCase):
    def test_english_order_is_bundled_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Easy-Turn-Testset-en"
            complete = root / "complete"
            incomplete = root / "incomplete"
            complete.mkdir(parents=True)
            incomplete.mkdir()
            for name in ("2.wav", "10.wav", "1.wav"):
                (complete / name).write_bytes(b"wav")
            order_root = Path(directory) / "orders"
            order_root.mkdir()
            (order_root / "en_complete.txt").write_text(
                "2.wav\n10.wav\n1.wav\n", encoding="utf-8"
            )

            expected = dataset.EXPECTED_COUNTS[("en", "complete")]
            dataset.EXPECTED_COUNTS[("en", "complete")] = 3
            try:
                with mock.patch.object(dataset, "ENGLISH_ORDER_ROOT", order_root):
                    samples = dataset.discover_samples(root, "en", "complete")
            finally:
                dataset.EXPECTED_COUNTS[("en", "complete")] = expected

            self.assertEqual(
                [sample.wav_path.name for sample in samples],
                ["2.wav", "10.wav", "1.wav"],
            )
            self.assertEqual(
                dataset.SAMPLE_ORDER["en"],
                "bundled-release-order-v1",
            )

    def test_english_order_requires_bundled_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Easy-Turn-Testset-en"
            (root / "complete").mkdir(parents=True)
            with mock.patch.object(
                dataset, "ENGLISH_ORDER_ROOT", Path(directory) / "missing"
            ):
                with self.assertRaisesRegex(FileNotFoundError, "order manifest"):
                    dataset.discover_samples(root, "en", "complete")


class SenseVoiceCleanupTests(unittest.TestCase):
    def test_removes_official_emotion_markers(self):
        self.assertEqual(
            runner.clean_sensevoice_text("😊 Hello 👏 world 😭"),
            " Hello  world ",
        )

    def test_rejects_non_lexical_output(self):
        self.assertEqual(runner.clean_sensevoice_text("😊🎼👏"), "")

    def test_keeps_english_and_chinese_text(self):
        self.assertEqual(runner.clean_sensevoice_text("English text"), "English text")
        self.assertEqual(runner.clean_sensevoice_text("中文文本"), "中文文本")


if __name__ == "__main__":
    unittest.main()
