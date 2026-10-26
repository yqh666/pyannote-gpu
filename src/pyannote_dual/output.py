# -*- coding: utf-8 -*-
"""JSON, TXT and RTTM writers for both diarization views."""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import DiarizationBundle, Segment, _union_duration


def _segments_payload(bundle: DiarizationBundle, mode: str) -> dict:
    segments = bundle.complete if mode == "complete" else bundle.non_conflict
    overlap = bundle.complete_overlap_seconds if mode == "complete" else bundle.non_conflict_overlap_seconds
    return {
        "mode": mode,
        "model": bundle.model,
        "device": bundle.device,
        "audio": str(bundle.audio),
        "duration_seconds": round(bundle.duration, 6),
        "num_speakers": len(bundle.speakers),
        "speakers": bundle.speakers,
        "segment_count": len(segments),
        "overlap_seconds": round(overlap, 6),
        "speech_seconds": round(_union_duration(segments), 6),
        "segments": [segment.as_dict() for segment in segments],
    }


def _write_text(path: Path, segments: list[Segment]) -> None:
    lines = [
        f"{segment.start:.3f}\t{segment.end:.3f}\t{segment.speaker}"
        for segment in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_rttm(path: Path, uri: str, segments: list[Segment]) -> None:
    lines = [
        "SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>".format(
            uri=uri,
            start=segment.start,
            duration=segment.duration,
            speaker=segment.speaker,
        )
        for segment in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_mode(bundle: DiarizationBundle, root: Path, mode: str) -> None:
    target = root / mode
    target.mkdir(parents=True, exist_ok=True)
    segments = bundle.complete if mode == "complete" else bundle.non_conflict
    (target / "segments.json").write_text(
        json.dumps(_segments_payload(bundle, mode), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_text(target / "diarization.txt", segments)
    _write_rttm(target / "diarization.rttm", bundle.audio.stem, segments)


def write_bundle(
    bundle: DiarizationBundle,
    output_dir: str | Path,
    inference_seconds: float | None = None,
) -> dict:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_mode(bundle, root, "complete")
    _write_mode(bundle, root, "non_conflict")
    summary = bundle.summary(inference_seconds=inference_seconds)
    summary["output_dir"] = str(root)
    summary["artifacts"] = {
        "complete": {
            "segments_json": str(root / "complete" / "segments.json"),
            "txt": str(root / "complete" / "diarization.txt"),
            "rttm": str(root / "complete" / "diarization.rttm"),
        },
        "non_conflict": {
            "segments_json": str(root / "non_conflict" / "segments.json"),
            "txt": str(root / "non_conflict" / "diarization.txt"),
            "rttm": str(root / "non_conflict" / "diarization.rttm"),
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
