#!/usr/bin/env python3
"""Measure VAD robustness under synthetic additive white noise."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import webrtcvad
from silero_vad import load_silero_vad

from compare_ava_baselines import webrtc_predictions
from compare_ava_subset import FRAME_MS, FRAME_SAMPLES, score_threshold
from plot_variant_pr_curves import (
    current_adaptive_scores,
    frame_raw_scores,
    frequency_scanner_scores,
    level_crossing_scores,
    load_subset,
    metrics_from_predictions,
    old_window_iir_scores,
    raw_delta_scores,
    silero_predictions,
)


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "ava_noise_robustness.csv"
SUMMARY_PATH = ROOT / "ava_noise_robustness_summary.json"
CHART_PATH = ROOT / "ava_noise_robustness.png"

SNR_LEVELS_DB: tuple[float | None, ...] = (None, 30.0, 20.0, 10.0, 5.0, 0.0, -5.0)
RNG_SEED = 26_026


def add_white_noise(pcm_i16: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    audio = pcm_i16.astype(np.float32)
    if snr_db is None:
        return np.clip(np.rint(audio), -32768, 32767).astype(np.int16)

    signal_rms = float(np.sqrt(np.mean(audio * audio)))
    if signal_rms <= 0.0:
        return np.zeros(len(audio), dtype=np.int16)

    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(noise * noise)))
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noisy = audio + noise * (target_noise_rms / noise_rms)
    return np.clip(np.rint(noisy), -32768, 32767).astype(np.int16)


def snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def main() -> None:
    labels_per_clip, pcm_per_clip = load_subset()
    labels = np.concatenate(labels_per_clip)

    circuit_models = {
        "Committed baseline": (old_window_iir_scores, 0),
        "Frame raw energy": (frame_raw_scores, 16),
        "Raw + delta onset": (raw_delta_scores, 37),
        "Level crossing": (lambda pcm: level_crossing_scores(pcm, adaptive_floor=False), 74),
        "Current RTL freq scan": (current_adaptive_scores, 68),
        "Freq scan val-peeked": (lambda pcm: frequency_scanner_scores(pcm, validation_peeked=True), 32),
    }

    silero_model = load_silero_vad()
    webrtc_model = webrtcvad.Vad(3)
    rows: list[dict[str, object]] = []

    for snr_db in SNR_LEVELS_DB:
        rng = np.random.default_rng(RNG_SEED + (0 if snr_db is None else int((snr_db + 100.0) * 10.0)))
        noisy_clips = [add_white_noise(pcm, snr_db, rng) for pcm in pcm_per_clip]
        snr = snr_label(snr_db)

        for name, (fn, threshold) in circuit_models.items():
            scores = np.concatenate([
                fn(pcm)[: len(label)] for pcm, label in zip(noisy_clips, labels_per_clip)
            ])
            metrics = score_threshold(labels, scores, threshold)
            rows.append({
                "model": name,
                "snr_db": "" if snr_db is None else snr_db,
                "snr": snr,
                "operating_point": threshold,
                **metrics,
            })

        with torch.no_grad():
            silero_pred = np.concatenate([
                silero_predictions(silero_model, pcm, len(label), 0.25)
                for pcm, label in zip(noisy_clips, labels_per_clip)
            ])
        rows.append({
            "model": "Silero",
            "snr_db": "" if snr_db is None else snr_db,
            "snr": snr,
            "operating_point": 0.25,
            "threshold": 0.25,
            **metrics_from_predictions(labels, silero_pred),
        })

        webrtc_pred = np.concatenate([
            webrtc_predictions(webrtc_model, pcm, len(label))
            for pcm, label in zip(noisy_clips, labels_per_clip)
        ])
        rows.append({
            "model": "WebRTC",
            "snr_db": "" if snr_db is None else snr_db,
            "snr": snr,
            "operating_point": 3,
            "threshold": 3,
            **metrics_from_predictions(labels, webrtc_pred),
        })

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot(rows)

    summary = {
        "clip_count": len(labels_per_clip),
        "duration_s": len(labels) * FRAME_MS / 1000.0,
        "speech_fraction": float(np.mean(labels)),
        "noise": "additive white Gaussian noise over the whole clip",
        "operating_points": {
            row["model"]: row["operating_point"]
            for row in rows
            if row["snr"] == "clean"
        },
        "rows": rows,
        "chart": str(CHART_PATH),
        "csv": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def plot(rows: list[dict[str, object]]) -> None:
    snr_positions = {"clean": 35.0, "30 dB": 30.0, "20 dB": 20.0, "10 dB": 10.0, "5 dB": 5.0, "0 dB": 0.0, "-5 dB": -5.0}
    styles = {
        "Committed baseline": {"color": "#8e6c8a", "marker": "o"},
        "Frame raw energy": {"color": "#9d755d", "marker": "v"},
        "Raw + delta onset": {"color": "#4c78a8", "marker": "s"},
        "Level crossing": {"color": "#f58518", "marker": "^"},
        "Current RTL freq scan": {"color": "#54a24b", "marker": "o"},
        "Freq scan val-peeked": {"color": "#e45756", "marker": "P"},
        "Silero": {"color": "#72b7b2", "marker": "X"},
        "WebRTC": {"color": "#ff9da6", "marker": "*"},
    }

    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for metric, axis in zip(("accuracy", "f1"), axes):
        for model, model_rows in by_model.items():
            ordered = sorted(model_rows, key=lambda row: snr_positions[str(row["snr"])], reverse=True)
            axis.plot(
                [snr_positions[str(row["snr"])] for row in ordered],
                [float(row[metric]) for row in ordered],
                label=model,
                linewidth=2.0,
                markersize=4.5,
                alpha=0.92,
                **styles[model],
            )
        axis.set_ylabel(metric.upper() if metric == "f1" else metric.title())
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.28)

    axes[0].set_title("AVA Subset Robustness with Synthetic White Noise")
    axes[1].set_xlabel("Input SNR")
    axes[1].set_xticks([35, 30, 20, 10, 5, 0, -5])
    axes[1].set_xticklabels(["clean", "30 dB", "20 dB", "10 dB", "5 dB", "0 dB", "-5 dB"])
    axes[0].legend(loc="lower left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
