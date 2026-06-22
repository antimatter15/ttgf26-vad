#!/usr/bin/env python3
"""Compare the Tiny Tapeout PDM energy estimator against Silero VAD labels."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from tiny_vad_models import rtl_energy_from_pcm


ROOT = Path(__file__).resolve().parent
WAV_PATH = ROOT / "jfk_moon_120s_16k.wav"
SUMMARY_PATH = ROOT / "jfk_vad_comparison_summary.json"
THRESHOLD_CSV_PATH = ROOT / "jfk_vad_threshold_sweep.csv"
FRAME_CSV_PATH = ROOT / "jfk_vad_frames.csv"

SAMPLE_RATE = 16_000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def read_wav_i16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32)


def simulate_energy_estimator(pcm_i16: np.ndarray) -> np.ndarray:
    return rtl_energy_from_pcm(pcm_i16)


def silero_frame_labels(audio_len: int, speech_timestamps: list[dict[str, int]]) -> np.ndarray:
    frame_count = audio_len // FRAME_SAMPLES
    labels = np.zeros(frame_count, dtype=bool)
    for segment in speech_timestamps:
        start_frame = max(0, segment["start"] // FRAME_SAMPLES)
        end_frame = min(frame_count, (segment["end"] + FRAME_SAMPLES - 1) // FRAME_SAMPLES)
        labels[start_frame:end_frame] = True
    return labels


def frame_scores(energy: np.ndarray, mode: str) -> np.ndarray:
    usable = (len(energy) // FRAME_SAMPLES) * FRAME_SAMPLES
    frames = energy[:usable].reshape((-1, FRAME_SAMPLES))
    if mode == "mean":
        return frames.mean(axis=1)
    if mode == "max":
        return frames.max(axis=1)
    raise ValueError(mode)


def score_threshold(labels: np.ndarray, scores: np.ndarray, threshold: int) -> dict[str, float]:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def sweep_thresholds(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    return [score_threshold(labels, scores, threshold) for threshold in range(256)]


def main() -> None:
    pcm = read_wav_i16(WAV_PATH)

    model = load_silero_vad()
    audio = torch.from_numpy(np.clip(pcm / 32768.0, -1.0, 1.0).astype(np.float32))
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )

    labels = silero_frame_labels(len(pcm), speech_timestamps)
    energy = simulate_energy_estimator(pcm)
    scores_by_mode = {
        "mean": frame_scores(energy, "mean")[: len(labels)],
        "max": frame_scores(energy, "max")[: len(labels)],
    }

    sweeps = {
        mode: sweep_thresholds(labels, scores)
        for mode, scores in scores_by_mode.items()
    }
    best = {
        mode: max(rows, key=lambda row: (row["f1"], row["accuracy"]))
        for mode, rows in sweeps.items()
    }

    with THRESHOLD_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "threshold",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ],
        )
        writer.writeheader()
        for mode, rows in sweeps.items():
            for row in rows:
                writer.writerow({"mode": mode, **row})

    with FRAME_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "silero_speech", "energy_mean", "energy_max"])
        writer.writeheader()
        for idx in range(len(labels)):
            writer.writerow(
                {
                    "time_s": round(idx * FRAME_MS / 1000.0, 3),
                    "silero_speech": int(labels[idx]),
                    "energy_mean": round(float(scores_by_mode["mean"][idx]), 3),
                    "energy_max": round(float(scores_by_mode["max"][idx]), 3),
                }
            )

    summary = {
        "audio": str(WAV_PATH),
        "duration_s": len(pcm) / SAMPLE_RATE,
        "sample_rate_hz": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "silero_segments": len(speech_timestamps),
        "silero_speech_fraction": float(np.mean(labels)),
        "energy_min": int(np.min(energy)),
        "energy_max": int(np.max(energy)),
        "energy_mean": float(np.mean(energy)),
        "best_thresholds": best,
        "speech_timestamps_samples": speech_timestamps,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
