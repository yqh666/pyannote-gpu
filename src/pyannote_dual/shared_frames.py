"""Reuse WeSpeaker frame features across local speakers in a chunk.

The official model applies masks only in statistics pooling. Computing the
same convolutional features three times is unnecessary. Mask selection,
pooling, PLDA, clustering and both reconstructions remain upstream operations.
"""
from __future__ import annotations

import math
import types


def enable_shared_frames(pipeline, chunk_batch_size: int = 16) -> None:
    import torch
    from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34

    model = pipeline._embedding.model_
    if not isinstance(model, WeSpeakerResNet34) or model.training:
        raise ValueError("shared-frames requires the evaluation-mode Community-1 WeSpeakerResNet34")
    if chunk_batch_size < 1:
        raise ValueError("chunk_batch_size must be positive")
    pipeline.get_embeddings = types.MethodType(_get_embeddings, pipeline)
    pipeline.shared_frames_batch_size = chunk_batch_size


def _get_embeddings(self, file, binary_segmentations, exclude_overlap=False, hook=None):
    import numpy as np
    import torch

    if self.training:
        raise RuntimeError("shared-frames is an inference-only backend")
    model = self._embedding.model_
    duration = binary_segmentations.sliding_window.duration
    data = np.nan_to_num(binary_segmentations.data, nan=0.0).astype(np.float32)
    chunks, _, speakers = data.shape
    # Keep the official overlap-exclusion and short-clean-speech fallback rule.
    if exclude_overlap:
        min_samples = self._embedding.min_num_samples
        min_frames = math.ceil(data.shape[1] * min_samples / (duration * self._embedding.sample_rate))
        clean = data * (np.sum(data, axis=2, keepdims=True) < 2)
        enough = np.sum(clean, axis=1, keepdims=True) > min_frames
        masks = np.where(enough, clean, data)
    else:
        masks = data
    batch_size = self.shared_frames_batch_size
    total = math.ceil(chunks / batch_size)
    if hook:
        hook("embeddings", None, total=total, completed=0)
    results = []
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, chunks, batch_size), 1):
            stop = min(chunks, start + batch_size)
            waveforms = [self._audio.crop(file, binary_segmentations.sliding_window[i], mode="pad")[0]
                         for i in range(start, stop)]
            waves = torch.stack(waveforms).to(self._embedding.device)
            weights = torch.from_numpy(masks[start:stop].transpose(0, 2, 1).copy()).to(self._embedding.device)
            encoder = getattr(self, 'shared_frame_encoder', model.forward_frames)
            frames = encoder(waves)
            # Pool one speaker at a time without repeating the large frame tensor.
            embeddings = torch.stack([model.forward_embedding(frames, weights=weights[:, s])
                                      for s in range(speakers)], dim=1).cpu().numpy()
            results.append(embeddings)
            if hook:
                hook("embeddings", embeddings, total=total, completed=batch_index)
    return np.concatenate(results, axis=0)
