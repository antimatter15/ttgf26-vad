#!/usr/bin/env python3
"""Plot precision-recall curves for the tile estimator and VAD baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import webrtcvad
from silero_vad import get_speech_timestamps, load_silero_vad

from compare_ava_baselines import read_wav_i16, webrtc_predictions
from compare_ava_subset import (
    FRAME_MS,
    FRAME_SAMPLES,
    LABEL_PATH,
    SAMPLE_RATE,
    SUBSET_DIR,
    frame_labels,
    frame_scores,
    read_labels,
    score_threshold,
    simulate_energy_estimator,
)


ROOT = Path(__file__).resolve().parent
CURVE_CSV_PATH = ROOT / "ava_pr_curves.csv"
SUMMARY_PATH = ROOT / "ava_pr_curves_summary.json"
CHART_PATH = ROOT / "ava_pr_curves.png"


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


def silero_predictions(model, pcm_i16: np.ndarray, frame_count: int, threshold: float) -> np.ndarray:
    audio = torch.from_numpy(np.clip(pcm_i16.astype(np.float32) / 32768.0, -1.0, 1.0))
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
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


def pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep points that are not dominated in precision and recall."""
    frontier: list[dict[str, object]] = []
    for row in rows:
        precision = float(row["precision"])
        recall = float(row["recall"])
        dominated = any(
            float(other["precision"]) >= precision
            and float(other["recall"]) >= recall
            and (
                float(other["precision"]) > precision
                or float(other["recall"]) > recall
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: float(row["recall"]))


def load_subset() -> tuple[list[np.ndarray], list[np.ndarray]]:
    labels_by_video = read_labels(LABEL_PATH)
    labels_per_clip: list[np.ndarray] = []
    pcm_per_clip: list[np.ndarray] = []
    wavs = sorted(SUBSET_DIR.glob("*_16k.wav"))
    if not wavs:
        raise SystemExit(f"no wavs found in {SUBSET_DIR}")

    for wav_path in wavs:
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        clip_start_s = float(start_s_text)
        pcm = read_wav_i16(wav_path)
        scores = frame_scores(simulate_energy_estimator(pcm.astype(np.float32)))
        labels = frame_labels(labels_by_video[video_id], clip_start_s, len(scores))
        labels_per_clip.append(labels)
        pcm_per_clip.append(pcm[: len(scores) * FRAME_SAMPLES])

    return labels_per_clip, pcm_per_clip


def main() -> None:
    labels_per_clip, pcm_per_clip = load_subset()
    labels = np.concatenate(labels_per_clip)

    energy_scores = np.concatenate([
        frame_scores(simulate_energy_estimator(pcm.astype(np.float32)))
        for pcm in pcm_per_clip
    ])
    tile_rows = [
        {"model": "Tile frame score", **score_threshold(labels, energy_scores, threshold)}
        for threshold in range(256)
    ]

    silero_model = load_silero_vad()
    silero_rows: list[dict[str, object]] = []
    for threshold in [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]:
        predictions = np.concatenate([
            silero_predictions(silero_model, pcm, len(label), threshold)
            for pcm, label in zip(pcm_per_clip, labels_per_clip)
        ])
        silero_rows.append({
            "model": "Silero",
            "threshold": threshold,
            **metrics_from_predictions(labels, predictions),
        })

    webrtc_rows: list[dict[str, object]] = []
    for mode in range(4):
        vad = webrtcvad.Vad(mode)
        predictions = np.concatenate([
            webrtc_predictions(vad, pcm, len(label))
            for pcm, label in zip(pcm_per_clip, labels_per_clip)
        ])
        webrtc_rows.append({
            "model": "WebRTC",
            "threshold": mode,
            **metrics_from_predictions(labels, predictions),
        })

    rows = tile_rows + silero_rows + webrtc_rows
    fieldnames = ["model", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"]
    with CURVE_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_model = {
        "Tile frame score": tile_rows,
        "Silero": silero_rows,
        "WebRTC": webrtc_rows,
    }
    frontiers = {name: pareto_frontier(model_rows) for name, model_rows in by_model.items()}
    best_f1 = {
        name: max(model_rows, key=lambda row: (float(row["f1"]), float(row["accuracy"])))
        for name, model_rows in by_model.items()
    }

    plt.figure(figsize=(9, 7))
    styles = {
        "Tile frame score": {"color": "#4c78a8", "marker": "o"},
        "Silero": {"color": "#54a24b", "marker": "s"},
        "WebRTC": {"color": "#f58518", "marker": "^"},
    }
    for name, frontier in frontiers.items():
        recalls = [float(row["recall"]) for row in frontier]
        precisions = [float(row["precision"]) for row in frontier]
        plt.plot(
            recalls,
            precisions,
            label=name,
            linewidth=2.0,
            markersize=4.5,
            **styles[name],
        )

        best = best_f1[name]
        plt.scatter(
            [float(best["recall"])],
            [float(best["precision"])],
            color=styles[name]["color"],
            edgecolor="black",
            s=95,
            zorder=4,
        )
        text_offset = (-110, 7) if float(best["recall"]) > 0.9 else (8, 7)
        plt.annotate(
            f"best F1 {float(best['f1']):.3f}",
            (float(best["recall"]), float(best["precision"])),
            textcoords="offset points",
            xytext=text_offset,
            fontsize=9,
        )

    plt.title("Precision-Recall Pareto Frontiers on AVA Subset")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.28)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=180)

    summary = {
        "clip_count": len(labels_per_clip),
        "duration_s": len(labels) * FRAME_MS / 1000.0,
        "speech_fraction": float(np.mean(labels)),
        "best_f1": best_f1,
        "frontier_point_counts": {name: len(points) for name, points in frontiers.items()},
        "chart": str(CHART_PATH),
        "csv": str(CURVE_CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
