#!/usr/bin/env python3
"""Compare the PDM energy estimator against AVA-Speech labels."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
from tiny_vad_models import rtl_energy_from_pcm


ROOT = Path(__file__).resolve().parent
LABEL_PATH = ROOT / "external" / "VAD_Benchmark" / "dataset" / "ava_speech_labels_v1.csv"
SUBSET_DIR = ROOT / "ava_subset"
SUMMARY_PATH = ROOT / "ava_subset_comparison_summary.json"
THRESHOLD_CSV_PATH = ROOT / "ava_subset_threshold_sweep.csv"
FRAME_CSV_PATH = ROOT / "ava_subset_frames.csv"

SAMPLE_RATE = 16_000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
SPEECH_LABELS = {"CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE"}


def read_labels(label_path: Path) -> dict[str, list[tuple[float, float, str]]]:
    labels: dict[str, list[tuple[float, float, str]]] = {}
    with label_path.open(newline="") as f:
        for video_id, start, end, label in csv.reader(f):
            labels.setdefault(video_id, []).append((float(start), float(end), label))
    return labels


def read_wav_i16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32)


def simulate_energy_estimator(pcm_i16: np.ndarray) -> np.ndarray:
    return rtl_energy_from_pcm(pcm_i16)


def frame_scores(energy: np.ndarray) -> np.ndarray:
    usable = (len(energy) // FRAME_SAMPLES) * FRAME_SAMPLES
    return energy[:usable].reshape((-1, FRAME_SAMPLES)).mean(axis=1)


def frame_labels(
    clip_labels: list[tuple[float, float, str]],
    clip_start_s: float,
    frame_count: int,
) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=bool)
    for start, end, label in clip_labels:
        if label not in SPEECH_LABELS:
            continue
        local_start = start - clip_start_s
        local_end = end - clip_start_s
        if local_end <= 0 or local_start >= frame_count * FRAME_MS / 1000.0:
            continue
        start_frame = max(0, int(np.floor(local_start * 1000 / FRAME_MS)))
        end_frame = min(frame_count, int(np.ceil(local_end * 1000 / FRAME_MS)))
        labels[start_frame:end_frame] = True
    return labels


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


def main() -> None:
    labels_by_video = read_labels(LABEL_PATH)
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_clip: list[dict[str, object]] = []

    wavs = sorted(SUBSET_DIR.glob("*_16k.wav"))
    if not wavs:
        raise SystemExit(f"no wavs found in {SUBSET_DIR}")

    with FRAME_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip", "time_s", "ava_speech", "energy_mean"],
        )
        writer.writeheader()

        for wav_path in wavs:
            video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
            clip_start_s = float(start_s_text)
            pcm = read_wav_i16(wav_path)
            scores = frame_scores(simulate_energy_estimator(pcm))
            labels = frame_labels(labels_by_video[video_id], clip_start_s, len(scores))
            all_scores.append(scores)
            all_labels.append(labels)
            per_clip.append({
                "clip": wav_path.name,
                "duration_s": len(scores) * FRAME_MS / 1000.0,
                "speech_fraction": float(np.mean(labels)),
                "energy_mean": float(np.mean(scores)),
                "energy_max": float(np.max(scores)),
            })
            for idx, (label, score) in enumerate(zip(labels, scores)):
                writer.writerow({
                    "clip": wav_path.name,
                    "time_s": round(idx * FRAME_MS / 1000.0, 3),
                    "ava_speech": int(label),
                    "energy_mean": round(float(score), 3),
                })

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    rows = [score_threshold(labels, scores, threshold) for threshold in range(256)]
    best = max(rows, key=lambda row: (row["f1"], row["accuracy"]))

    with THRESHOLD_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "clips": per_clip,
        "clip_count": len(wavs),
        "total_duration_s": len(labels) * FRAME_MS / 1000.0,
        "frame_ms": FRAME_MS,
        "speech_fraction": float(np.mean(labels)),
        "energy_min": float(np.min(scores)),
        "energy_max": float(np.max(scores)),
        "energy_mean": float(np.mean(scores)),
        "best_threshold": best,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
