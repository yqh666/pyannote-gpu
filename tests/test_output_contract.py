# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyannote_dual.pipeline import Segment, has_overlap  # noqa: E402


def load_segments(path: Path) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Segment(item["start"], item["end"], item["speaker"]) for item in payload["segments"]]


class ExampleOutputContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.complete = load_segments(ROOT / "example" / "output" / "complete" / "segments.json")
        cls.non_conflict = load_segments(
            ROOT / "example" / "output" / "non_conflict" / "segments.json"
        )
        cls.complete_payload = json.loads(
            (ROOT / "example" / "output" / "complete" / "segments.json").read_text(encoding="utf-8")
        )
        cls.non_conflict_payload = json.loads(
            (ROOT / "example" / "output" / "non_conflict" / "segments.json").read_text(encoding="utf-8")
        )

    def test_complete_output_contains_overlap(self) -> None:
        self.assertTrue(has_overlap(self.complete))
        self.assertGreater(self.complete_payload["overlap_seconds"], 0)

    def test_non_conflict_output_has_no_overlap(self) -> None:
        self.assertFalse(has_overlap(self.non_conflict))
        self.assertEqual(self.non_conflict_payload["overlap_seconds"], 0)

    def test_segments_are_valid(self) -> None:
        for segments, payload in (
            (self.complete, self.complete_payload),
            (self.non_conflict, self.non_conflict_payload),
        ):
            for segment in segments:
                self.assertGreater(segment.end, segment.start)
                self.assertGreaterEqual(segment.start, 0)
                self.assertLessEqual(segment.end, payload["duration_seconds"] + 1e-6)

    def test_speaker_sets_are_consistent(self) -> None:
        complete_speakers = {segment.speaker for segment in self.complete}
        non_conflict_speakers = {segment.speaker for segment in self.non_conflict}
        self.assertEqual(complete_speakers, non_conflict_speakers)


if __name__ == "__main__":
    unittest.main()
