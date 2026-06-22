#!/usr/bin/env python3
"""Hardware-shaped PDM VAD score models used for experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_WINDOWS = 160


def pcm_to_density(pcm_i16: np.ndarray) -> np.ndarray:
    normalized = np.clip(pcm_i16.astype(np.float32) / 32768.0, -1.0, 1.0)
    ones = np.rint((normalized + 1.0) * 32.0).astype(np.int16)
    return np.clip(ones, 0, 64)


def rtl_energy_from_density(ones_per_window: np.ndarray) -> np.ndarray:
    """Current RTL-equivalent energy estimator."""
    magnitudes = np.abs(ones_per_window.astype(np.int16) - 32)
    energy = np.zeros(len(magnitudes), dtype=np.uint8)
    state = 0
    for idx, magnitude in enumerate(magnitudes):
        state = (state - (state >> 2) + int(magnitude)) & 0xFF
        energy[idx] = state
    return energy


def rtl_energy_from_pcm(pcm_i16: np.ndarray) -> np.ndarray:
    return rtl_frame_score_from_pcm(pcm_i16)


def rtl_frame_score_from_density(ones_per_window: np.ndarray) -> np.ndarray:
    """Current RTL-equivalent frequency-scanner score."""
    density = ones_per_window.astype(np.int16) - 32
    usable = (len(density) // FRAME_WINDOWS) * FRAME_WINDOWS
    density = density[:usable]
    frames = density.reshape((-1, FRAME_WINDOWS)).astype(np.int16)
    raw = np.abs(frames).sum(axis=1).astype(np.int32)

    band_sum = np.zeros(len(frames), dtype=np.int32)
    for step in (8, 9, 24):
        phase = (np.arange(FRAME_WINDOWS, dtype=np.int32) * step) & 0xFF
        sign = np.where(phase < 128, 1, -1).astype(np.int16)
        band_sum += np.abs((frames * sign).sum(axis=1)).astype(np.int32)

    frame_scores = np.zeros(len(raw), dtype=np.uint8)
    score = 0
    prev_raw = 0
    for idx, raw_sum in enumerate(raw):
        raw_int = int(raw_sum)
        feature = (raw_int >> 10) + (abs(raw_int - prev_raw) >> 4) + (int(band_sum[idx]) >> 7)
        feature = min(feature, 255)
        score = score - (score >> 6) + feature
        score = min(score, 255)
        frame_scores[idx] = score
        prev_raw = raw_int
    return np.repeat(frame_scores, FRAME_WINDOWS)


def rtl_frame_score_from_pcm(pcm_i16: np.ndarray) -> np.ndarray:
    return rtl_frame_score_from_density(pcm_to_density(pcm_i16))


@dataclass(frozen=True)
class CandidateParams:
    name: str
    raw_weight: int
    delta_weight: int
    leak_shift: int
    noise_shift: int | None
    noise_margin: int


def candidate_score_from_density(ones_per_window: np.ndarray, params: CandidateParams) -> np.ndarray:
    """Simulate small tile-friendly score variants.

    `raw_weight` and `delta_weight` are power-of-two shift divisors:
    feature = raw >> raw_weight + delta >> delta_weight.

    If `noise_shift` is set, an asymmetric baseline tracks the feature and the
    output score is max(0, smoothed_feature - noise - margin).
    """
    density = ones_per_window.astype(np.int16) - 32
    raw = np.abs(density)
    delta = np.abs(np.diff(density, prepend=density[0]))
    score = np.zeros(len(raw), dtype=np.uint8)
    fast = 0
    noise = 0

    for idx, (raw_mag, delta_mag) in enumerate(zip(raw, delta)):
        feature = int(raw_mag >> params.raw_weight) + int(delta_mag >> params.delta_weight)
        feature = min(feature, 255)
        fast = fast - (fast >> params.leak_shift) + feature
        fast = min(fast, 255)

        if params.noise_shift is not None:
            if fast > noise:
                noise += (fast - noise) >> params.noise_shift
            else:
                noise -= (noise - fast) >> 2
            noise = max(0, min(noise, 255))
            value = max(0, fast - noise - params.noise_margin)
        else:
            value = fast

        score[idx] = min(value, 255)

    return score


def candidate_score_from_pcm(pcm_i16: np.ndarray, params: CandidateParams) -> np.ndarray:
    return candidate_score_from_density(pcm_to_density(pcm_i16), params)
