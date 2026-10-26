# -*- coding: utf-8 -*-
"""Reusable wrapper for pyannote/speaker-diarization-community-1.

Audio files are decoded through FFmpeg and TorchCodec. The Community-1 pipeline
returns both complete diarization and the official exclusive diarization from a
single inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import activate_runtime

MODEL_ID = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "speaker": self.speaker,
        }


def _annotation_segments(annotation: Any) -> list[Segment]:
    return [
        Segment(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _union_duration(segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    intervals = sorted((segment.start, segment.end) for segment in segments)
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def overlap_seconds(segments: list[Segment]) -> float:
    """Return the total amount of time covered by two or more turns."""
    speech_seconds = _union_duration(segments)
    total_seconds = sum(segment.duration for segment in segments)
    return max(0.0, total_seconds - speech_seconds)


def has_overlap(segments: list[Segment], tolerance: float = 1e-6) -> bool:
    """Return True when two turns overlap by more than tolerance seconds."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    furthest_end = float("-inf")
    for segment in ordered:
        if segment.start < furthest_end - tolerance:
            return True
        furthest_end = max(furthest_end, segment.end)
    return False


@dataclass
class DiarizationBundle:
    audio: Path
    duration: float
    complete: list[Segment]
    non_conflict: list[Segment]
    model: str
    device: str
    backend: str = "torch"

    @property
    def speakers(self) -> list[str]:
        return sorted({segment.speaker for segment in self.complete})

    @property
    def complete_overlap_seconds(self) -> float:
        return overlap_seconds(self.complete)

    @property
    def non_conflict_overlap_seconds(self) -> float:
        return overlap_seconds(self.non_conflict)

    def summary(self, inference_seconds: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "device": self.device,
            "backend": self.backend,
            "audio": str(self.audio),
            "duration_seconds": round(self.duration, 6),
            "num_speakers": len(self.speakers),
            "speakers": self.speakers,
            "complete": {
                "description": "Complete output; overlapping speech is preserved.",
                "segment_count": len(self.complete),
                "overlap_seconds": round(self.complete_overlap_seconds, 6),
                "has_overlap": has_overlap(self.complete),
            },
            "non_conflict": {
                "description": "Official exclusive output; at most one speaker at any time.",
                "segment_count": len(self.non_conflict),
                "overlap_seconds": round(self.non_conflict_overlap_seconds, 6),
                "has_overlap": has_overlap(self.non_conflict),
            },
        }
        if inference_seconds is not None:
            payload["inference_seconds"] = round(inference_seconds, 6)
            payload["rtf"] = round(inference_seconds / self.duration, 6) if self.duration else None
        return payload


class Community1Diarizer:
    """Load Community-1 once and decode audio through FFmpeg/TorchCodec."""

    def __init__(self, pipeline: Any, model: str, device: str, backend: str = "torch") -> None:
        self._pipeline = pipeline
        self.model = model
        self.device = device
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = MODEL_ID,
        token: str | None = None,
        device: str = "auto",
        offline: bool = False,
        ffmpeg_bin: str | Path | None = None,
        backend: str = "torch",
        embedding_batch_size: int = 32,
        chunk_batch_size: int = 16,
        engine_path: str | Path | None = None,
    ) -> "Community1Diarizer":
        import os

        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        activate_runtime(ffmpeg_bin=ffmpeg_bin)

        import torch
        from pyannote.audio import Pipeline

        if backend not in {"torch", "shared-frames", "tensorrt"}:
            raise ValueError("backend must be torch, shared-frames or tensorrt")
        if embedding_batch_size < 1 or chunk_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        selected_model = str(model)
        selected_path = Path(selected_model)
        checkpoint: str | Path = selected_path if selected_path.exists() else selected_model
        pipeline = Pipeline.from_pretrained(checkpoint, token=token)
        if pipeline is None:
            raise RuntimeError(
                f"Could not load {selected_model!r}. Accept the model license and provide --token, "
                "or pass a local model directory with --model."
            )

        if device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in {"cpu", "cuda"}:
            selected_device = device
        else:
            raise ValueError("device must be one of: auto, cpu, cuda")

        if selected_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this Python environment.")
        if selected_device != "cpu":
            pipeline.to(torch.device(selected_device))
        pipeline.embedding_batch_size = embedding_batch_size
        if backend in {"shared-frames", "tensorrt"}:
            from .shared_frames import enable_shared_frames
            enable_shared_frames(pipeline, chunk_batch_size)
        if backend == "tensorrt":
            if selected_device != 'cuda':
                raise ValueError('TensorRT requires --device cuda')
            from .trt_backend import TensorRTFrames, DEFAULT_ENGINE
            encoder = TensorRTFrames(engine_path or DEFAULT_ENGINE, pipeline._embedding.model_)
            if chunk_batch_size > encoder.metadata['max_batch']:
                raise ValueError('chunk_batch_size exceeds the TensorRT engine profile')
            pipeline.shared_frame_encoder = encoder
        return cls(pipeline=pipeline, model=selected_model, device=selected_device, backend=backend)

    def process(
        self,
        audio: str | Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        streaming: bool = False,
    ) -> DiarizationBundle:
        audio_path = Path(audio).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        for value in (num_speakers, min_speakers, max_speakers):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("speaker counts must be positive integers")
        if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
            raise ValueError("num_speakers cannot be combined with speaker bounds")
        if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
            raise ValueError("min_speakers must not exceed max_speakers")

        kwargs: dict[str, int] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = int(num_speakers)
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                kwargs["max_speakers"] = int(max_speakers)

        output = self._pipeline(str(audio_path), preload=not streaming, **kwargs)
        complete_annotation = getattr(output, "speaker_diarization", output)
        exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive_annotation is None:
            raise RuntimeError(
                "This pyannote pipeline does not return exclusive_speaker_diarization. "
                "Install pyannote.audio 4.0.x and use the Community-1 pipeline."
            )

        from torchcodec.decoders import AudioDecoder

        metadata = AudioDecoder(str(audio_path)).metadata
        duration = float(metadata.duration_seconds_from_header)
        return DiarizationBundle(
            audio=audio_path,
            duration=duration,
            complete=_annotation_segments(complete_annotation),
            non_conflict=_annotation_segments(exclusive_annotation),
            model=self.model,
            device=self.device,
            backend=self.backend,
        )
