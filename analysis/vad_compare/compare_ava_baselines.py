#!/usr/bin/env python3
"""Compare the PDM energy estimator with lightweight VAD baselines."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import webrtcvad
from silero_vad import get_speech_timestamps, load_silero_vad

from compare_ava_subset import (
    FRAME_MS,
    FRAME_SAMPLES,
    LABEL_PATH,
    SAMPLE_RATE,
    SPEECH_LABELS,
    SUBSET_DIR,
    frame_labels,
    frame_scores,
    read_labels,
    score_threshold,
    simulate_energy_estimator,
)


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "ava_baseline_comparison_summary.json"
CSV_PATH = ROOT / "ava_baseline_comparison.csv"
CHART_PATH = ROOT / "ava_baseline_comparison.png"


def read_wav_i16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2")


def metrics_from_predictions(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def silero_predictions(model, pcm_i16: np.ndarray, frame_count: int) -> np.ndarray:
    audio = torch.from_numpy(np.clip(pcm_i16.astype(np.float32) / 32768.0, -1.0, 1.0))
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )
    predictions = np.zeros(frame_count, dtype=bool)
    for segment in timestamps:
        start_frame = max(0, segment["start"] // FRAME_SAMPLES)
        end_frame = min(frame_count, (segment["end"] + FRAME_SAMPLES - 1) // FRAME_SAMPLES)
        predictions[start_frame:end_frame] = True
    return predictions


def webrtc_predictions(vad: webrtcvad.Vad, pcm_i16: np.ndarray, frame_count: int) -> np.ndarray:
    predictions = np.zeros(frame_count, dtype=bool)
    usable = min(len(pcm_i16), frame_count * FRAME_SAMPLES)
    pcm_i16 = pcm_i16[:usable]
    for idx in range(frame_count):
        frame = pcm_i16[idx * FRAME_SAMPLES : (idx + 1) * FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            break
        predictions[idx] = vad.is_speech(frame.astype("<i2", copy=False).tobytes(), SAMPLE_RATE)
    return predictions


def plot_metrics(rows: list[dict[str, object]]) -> None:
    names = [str(row["model"]) for row in rows]
    metric_names = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(11, 6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for offset, metric in enumerate(metric_names):
        values = [float(row[metric]) for row in rows]
        plt.bar(x + (offset - 1.5) * width, values, width, label=metric, color=colors[offset])

    plt.title("PDM Frame Score vs VAD_Benchmark Baselines on AVA Subset")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, names, rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=180)


def main() -> None:
    labels_by_video = read_labels(LABEL_PATH)
    wavs = sorted(SUBSET_DIR.glob("*_16k.wav"))
    if not wavs:
        raise SystemExit(f"no wavs found in {SUBSET_DIR}")

    all_labels: list[np.ndarray] = []
    all_energy_scores: list[np.ndarray] = []
    all_silero: list[np.ndarray] = []
    all_webrtc: list[np.ndarray] = []

    silero_model = load_silero_vad()
    webrtc_model = webrtcvad.Vad(2)

    for wav_path in wavs:
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        clip_start_s = float(start_s_text)
        pcm = read_wav_i16(wav_path)
        scores = frame_scores(simulate_energy_estimator(pcm.astype(np.float32)))
        labels = frame_labels(labels_by_video[video_id], clip_start_s, len(scores))
        all_labels.append(labels)
        all_energy_scores.append(scores)
        all_silero.append(silero_predictions(silero_model, pcm, len(scores)))
        all_webrtc.append(webrtc_predictions(webrtc_model, pcm, len(scores)))

    labels = np.concatenate(all_labels)
    energy_scores = np.concatenate(all_energy_scores)
    silero = np.concatenate(all_silero)
    webrtc = np.concatenate(all_webrtc)

    rows: list[dict[str, object]] = []
    for model_name, threshold in [("Tile score t=0", 0), ("Tile score best t=73", 73)]:
        row = score_threshold(labels, energy_scores, threshold)
        rows.append({"model": model_name, "threshold": threshold, **row})

    rows.append({"model": "WebRTC mode 2", "threshold": "", **metrics_from_predictions(labels, webrtc)})
    rows.append({"model": "Silero", "threshold": "", **metrics_from_predictions(labels, silero)})

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "clip_count": len(wavs),
        "duration_s": len(labels) * FRAME_MS / 1000.0,
        "speech_fraction": float(np.mean(labels)),
        "rows": rows,
        "chart": str(CHART_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    plot_metrics(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
