#!/usr/bin/env python3
"""Plot precision-recall curves for the tiny VAD circuit variants tried so far."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import webrtcvad
from silero_vad import get_speech_timestamps, load_silero_vad

from compare_ava_baselines import webrtc_predictions
from compare_ava_subset import (
    FRAME_MS,
    FRAME_SAMPLES,
    LABEL_PATH,
    SAMPLE_RATE,
    SUBSET_DIR,
    frame_labels,
    frame_scores,
    read_labels,
    read_wav_i16,
    score_threshold,
)
from tiny_vad_models import pcm_to_density, rtl_energy_from_pcm


ROOT = Path(__file__).resolve().parent
CURVE_CSV_PATH = ROOT / "ava_variant_pr_curves.csv"
SUMMARY_PATH = ROOT / "ava_variant_pr_curves_summary.json"
CHART_PATH = ROOT / "ava_variant_pr_curves.png"
ZOOM_CHART_PATH = ROOT / "ava_variant_pr_curves_zoom.png"
MIN_RECALL = 0.2


def pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier = []
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


def visible_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if float(row["recall"]) >= MIN_RECALL]


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


def load_subset() -> tuple[list[np.ndarray], list[np.ndarray]]:
    labels_by_video = read_labels(LABEL_PATH)
    labels_per_clip = []
    pcm_per_clip = []
    wavs = sorted(SUBSET_DIR.glob("*_16k.wav"))
    if not wavs:
        raise SystemExit(f"no wavs found in {SUBSET_DIR}")

    for wav_path in wavs:
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        pcm = read_wav_i16(wav_path)
        frame_count = len(pcm) // FRAME_SAMPLES
        labels = frame_labels(read_labels(LABEL_PATH)[video_id], float(start_s_text), frame_count)
        labels_per_clip.append(labels)
        pcm_per_clip.append(pcm[: frame_count * FRAME_SAMPLES])
    return labels_per_clip, pcm_per_clip


def frame_density(pcm_i16: np.ndarray) -> np.ndarray:
    usable = (len(pcm_i16) // FRAME_SAMPLES) * FRAME_SAMPLES
    density = pcm_to_density(pcm_i16[:usable]).astype(np.int16) - 32
    return density.reshape((-1, FRAME_SAMPLES)).astype(np.int16)


def old_window_iir_scores(pcm_i16: np.ndarray) -> np.ndarray:
    density = pcm_to_density(pcm_i16).astype(np.int16) - 32
    magnitudes = np.abs(density)
    state = 0
    sample_scores = np.zeros(len(magnitudes), dtype=np.uint8)
    for idx, magnitude in enumerate(magnitudes):
        state = (state - (state >> 2) + int(magnitude)) & 0xFF
        sample_scores[idx] = state
    return frame_scores(sample_scores)


def frame_raw_scores(pcm_i16: np.ndarray) -> np.ndarray:
    frames = frame_density(pcm_i16)
    raw = np.abs(frames).sum(axis=1).astype(np.int32)
    score = 0
    out = np.zeros(len(raw), dtype=np.uint8)
    for idx, raw_sum in enumerate(raw):
        feature = min(int(raw_sum) >> 6, 255)
        score = min(255, score - (score >> 4) + feature)
        out[idx] = score
    return out


def raw_delta_scores(pcm_i16: np.ndarray, leak_shift: int = 5) -> np.ndarray:
    frames = frame_density(pcm_i16)
    raw = np.abs(frames).sum(axis=1).astype(np.int32)
    score = 0
    prev_raw = 0
    out = np.zeros(len(raw), dtype=np.uint8)
    for idx, raw_sum in enumerate(raw):
        raw_int = int(raw_sum)
        feature = (raw_int >> 7) + (abs(raw_int - prev_raw) >> 4)
        feature = min(feature, 255)
        score = min(255, score - (score >> leak_shift) + feature)
        out[idx] = score
        prev_raw = raw_int
    return out


def level_crossing_scores(pcm_i16: np.ndarray, adaptive_floor: bool) -> np.ndarray:
    frames = frame_density(pcm_i16)
    raw = np.abs(frames).sum(axis=1).astype(np.int32)
    zero_crossings = (frames[:, :-1] * frames[:, 1:] < 0).sum(axis=1).astype(np.int32)
    score = 0
    floor = 0
    out = np.zeros(len(raw), dtype=np.uint8)
    for idx, (raw_sum, zc_count) in enumerate(zip(raw, zero_crossings)):
        feature = min((int(raw_sum) >> 7) + (int(zc_count) << 1), 255)
        score = min(255, score - (score >> 6) + feature)
        if adaptive_floor:
            if feature > floor:
                floor += (feature - floor) >> 6
            else:
                floor -= (floor - feature) >> 2
            value = max(0, score - floor)
        else:
            value = score
        out[idx] = value
    return out


def square_mixer_band(frames: np.ndarray, step: int) -> np.ndarray:
    phase = (np.arange(FRAME_SAMPLES, dtype=np.int32) * step) & 0xFF
    sign = np.where(phase < 128, 1, -1).astype(np.int16)
    return np.abs((frames * sign).sum(axis=1)).astype(np.int32)


def frequency_scanner_scores(pcm_i16: np.ndarray, *, validation_peeked: bool) -> np.ndarray:
    frames = frame_density(pcm_i16)
    raw = np.abs(frames).sum(axis=1).astype(np.int32)
    if validation_peeked:
        band_feature = np.maximum.reduce([
            square_mixer_band(frames, 4),
            square_mixer_band(frames, 16),
            square_mixer_band(frames, 32),
        ])
        raw_shift = 7
        delta_shift = None
        band_shift = 5
        leak_shift = 5
    else:
        band_feature = (
            square_mixer_band(frames, 4)
            + square_mixer_band(frames, 8)
            + square_mixer_band(frames, 16)
        )
        raw_shift = 8
        delta_shift = 4
        band_shift = 6
        leak_shift = 6

    score = 0
    prev_raw = 0
    out = np.zeros(len(raw), dtype=np.uint8)
    for idx, raw_sum in enumerate(raw):
        raw_int = int(raw_sum)
        feature = raw_int >> raw_shift
        if delta_shift is not None:
            feature += abs(raw_int - prev_raw) >> delta_shift
        feature += int(band_feature[idx]) >> band_shift
        feature = min(feature, 255)
        score = min(255, score - (score >> leak_shift) + feature)
        out[idx] = score
        prev_raw = raw_int
    return out


def current_adaptive_scores(pcm_i16: np.ndarray) -> np.ndarray:
    return frame_scores(rtl_energy_from_pcm(pcm_i16))


def score_model(labels: np.ndarray, model_name: str, scores: np.ndarray) -> list[dict[str, object]]:
    return [
        {"model": model_name, **score_threshold(labels, scores, threshold)}
        for threshold in range(256)
    ]


def main() -> None:
    labels_per_clip, pcm_per_clip = load_subset()
    labels = np.concatenate(labels_per_clip)

    model_fns = {
        "Old window IIR": old_window_iir_scores,
        "Frame raw energy": frame_raw_scores,
        "Raw + delta onset": raw_delta_scores,
        "Level crossing": lambda pcm: level_crossing_scores(pcm, adaptive_floor=False),
        "Adaptive floor RTL": current_adaptive_scores,
        "Freq scan dev-selected": lambda pcm: frequency_scanner_scores(pcm, validation_peeked=False),
        "Freq scan val-peeked": lambda pcm: frequency_scanner_scores(pcm, validation_peeked=True),
    }

    rows = []
    by_model = {}
    for name, fn in model_fns.items():
        scores = np.concatenate([fn(pcm)[: len(label)] for pcm, label in zip(pcm_per_clip, labels_per_clip)])
        model_rows = score_model(labels, name, scores)
        by_model[name] = model_rows
        rows.extend(model_rows)

    silero_model = load_silero_vad()
    silero_rows = []
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
    by_model["Silero"] = silero_rows
    rows.extend(silero_rows)

    webrtc_rows = []
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
    by_model["WebRTC"] = webrtc_rows
    rows.extend(webrtc_rows)

    by_model = {name: visible_rows(model_rows) for name, model_rows in by_model.items()}
    rows = [row for model_rows in by_model.values() for row in model_rows]

    fieldnames = ["model", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"]
    with CURVE_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frontiers = {name: pareto_frontier(model_rows) for name, model_rows in by_model.items()}
    best_f1 = {
        name: max(model_rows, key=lambda row: (float(row["f1"]), float(row["accuracy"])))
        for name, model_rows in by_model.items()
    }

    styles = {
        "Old window IIR": {"color": "#8e6c8a", "marker": "o"},
        "Frame raw energy": {"color": "#9d755d", "marker": "v"},
        "Raw + delta onset": {"color": "#4c78a8", "marker": "s"},
        "Level crossing": {"color": "#f58518", "marker": "^"},
        "Adaptive floor RTL": {"color": "#54a24b", "marker": "o"},
        "Freq scan dev-selected": {"color": "#b279a2", "marker": "D"},
        "Freq scan val-peeked": {"color": "#e45756", "marker": "P"},
        "Silero": {"color": "#72b7b2", "marker": "X"},
        "WebRTC": {"color": "#ff9da6", "marker": "*"},
    }

    def draw_chart(path: Path, *, zoom: bool) -> None:
        plt.figure(figsize=(10.5, 7.5))
        for name, frontier in frontiers.items():
            recalls = [float(row["recall"]) for row in frontier]
            precisions = [float(row["precision"]) for row in frontier]
            plt.plot(
                recalls,
                precisions,
                label=f"{name} (F1 {float(best_f1[name]['f1']):.3f})",
                linewidth=2.0,
                markersize=4.0,
                alpha=0.92,
                **styles[name],
            )
            best = best_f1[name]
            plt.scatter(
                [float(best["recall"])],
                [float(best["precision"])],
                color=styles[name]["color"],
                edgecolor="black",
                s=65,
                zorder=4,
            )

        title_suffix = "Zoom" if zoom else "Full"
        plt.title(f"Precision-Recall Frontiers for Tiny VAD Variants on AVA Subset ({title_suffix})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        if zoom:
            plt.xlim(0.72, 1.01)
            plt.ylim(0.48, 0.66)
        else:
            plt.xlim(MIN_RECALL - 0.02, 1.02)
            plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.28)
        plt.legend(loc="lower left", fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    draw_chart(CHART_PATH, zoom=False)
    draw_chart(ZOOM_CHART_PATH, zoom=True)

    summary = {
        "clip_count": len(labels_per_clip),
        "duration_s": len(labels) * FRAME_MS / 1000.0,
        "speech_fraction": float(np.mean(labels)),
        "min_recall": MIN_RECALL,
        "best_f1": best_f1,
        "frontier_point_counts": {name: len(points) for name, points in frontiers.items()},
        "chart": str(CHART_PATH),
        "zoom_chart": str(ZOOM_CHART_PATH),
        "csv": str(CURVE_CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
