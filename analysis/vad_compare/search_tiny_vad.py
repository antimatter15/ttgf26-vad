#!/usr/bin/env python3
"""Search small hardware-friendly PDM VAD score variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from compare_ava_subset import (
    FRAME_MS,
    FRAME_SAMPLES,
    LABEL_PATH,
    SUBSET_DIR,
    frame_labels,
    frame_scores,
    read_labels,
    read_wav_i16,
    score_threshold,
)
from tiny_vad_models import CandidateParams, pcm_to_density, rtl_energy_from_pcm


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "tiny_vad_search.csv"
SUMMARY_PATH = ROOT / "tiny_vad_search_summary.json"


def load_dataset() -> tuple[np.ndarray, list[np.ndarray]]:
    labels_by_video = read_labels(LABEL_PATH)
    labels_per_clip: list[np.ndarray] = []
    pcm_per_clip: list[np.ndarray] = []
    for wav_path in sorted(SUBSET_DIR.glob("*_16k.wav")):
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        clip_start_s = float(start_s_text)
        pcm = read_wav_i16(wav_path)
        baseline_scores = frame_scores(rtl_energy_from_pcm(pcm))
        labels = frame_labels(labels_by_video[video_id], clip_start_s, len(baseline_scores))
        labels_per_clip.append(labels)
        pcm_per_clip.append(pcm[: len(baseline_scores) * FRAME_SAMPLES])
    return np.concatenate(labels_per_clip), pcm_per_clip


def frame_sum(values: np.ndarray) -> np.ndarray:
    usable = (len(values) // FRAME_SAMPLES) * FRAME_SAMPLES
    return values[:usable].reshape((-1, FRAME_SAMPLES)).sum(axis=1)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    bins = np.clip(np.floor(scores), 0, 255).astype(np.uint8)
    pos_hist = np.bincount(bins[labels], minlength=256)
    neg_hist = np.bincount(bins[~labels], minlength=256)
    pos_ge = np.cumsum(pos_hist[::-1])[::-1]
    neg_ge = np.cumsum(neg_hist[::-1])[::-1]
    total_pos = int(np.sum(pos_hist))
    total_neg = int(np.sum(neg_hist))
    rows = []
    for threshold in range(256):
        tp = int(pos_ge[threshold])
        fp = int(neg_ge[threshold])
        fn = total_pos - tp
        tn = total_neg - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (total_pos + total_neg)
        rows.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        })
    return max(rows, key=lambda row: (row["f1"], row["accuracy"]))


def simulate_frame_candidate(
    raw_frames: list[np.ndarray],
    params: CandidateParams,
) -> np.ndarray:
    scores_per_clip: list[np.ndarray] = []
    for raw in raw_frames:
        fast = 0
        noise = 0
        prev_raw_value = 0
        scores = np.zeros(len(raw), dtype=np.uint8)
        for idx, raw_value in enumerate(raw):
            raw_int = int(raw_value)
            delta_int = abs(raw_int - prev_raw_value)
            feature = (raw_int >> params.raw_weight) + (delta_int >> params.delta_weight)
            feature = min(feature, 255)
            fast = fast - (fast >> params.leak_shift) + feature
            fast = min(fast, 255)
            if params.noise_shift is not None:
                if fast > noise:
                    noise += (fast - noise) >> params.noise_shift
                else:
                    noise -= (noise - fast) >> 2
                scores[idx] = max(0, fast - noise - params.noise_margin)
            else:
                scores[idx] = fast
            prev_raw_value = raw_int
        scores_per_clip.append(scores)
    return np.concatenate(scores_per_clip)


def score_frame_model(
    labels: np.ndarray,
    raw_frames: list[np.ndarray],
    params: CandidateParams,
) -> dict[str, object]:
    scores = simulate_frame_candidate(raw_frames, params)
    best = best_threshold(labels, scores)
    return {
        "name": params.name,
        "raw_weight": params.raw_weight,
        "delta_weight": params.delta_weight,
        "leak_shift": params.leak_shift,
        "noise_shift": "" if params.noise_shift is None else params.noise_shift,
        "noise_margin": params.noise_margin,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        **best,
    }


def candidate_grid() -> list[CandidateParams]:
    candidates: list[CandidateParams] = []
    raw_weights = [5, 6, 7, 8]
    delta_weights = [3, 4, 5, 6, 7]
    leak_shifts = [4, 5, 6]
    noise_options: list[int | None] = [None, 5, 6, 7]
    margins = [0, 4, 8, 16]

    for raw_weight in raw_weights:
        for delta_weight in delta_weights:
            for leak_shift in leak_shifts:
                for noise_shift in noise_options:
                    for margin in ([0] if noise_shift is None else margins):
                        name = (
                            f"raw{raw_weight}_delta{delta_weight}_"
                            f"leak{leak_shift}_noise{noise_shift}_margin{margin}"
                        )
                        candidates.append(CandidateParams(
                            name=name,
                            raw_weight=raw_weight,
                            delta_weight=delta_weight,
                            leak_shift=leak_shift,
                            noise_shift=noise_shift,
                            noise_margin=margin,
                        ))
    return candidates


def main() -> None:
    labels, pcm_per_clip = load_dataset()
    raw_frames: list[np.ndarray] = []
    for pcm in pcm_per_clip:
        density = pcm_to_density(pcm).astype(np.int16) - 32
        raw_frames.append(frame_sum(np.abs(density)))

    baseline_scores = np.concatenate([
        frame_scores(rtl_energy_from_pcm(pcm))
        for pcm in pcm_per_clip
    ])
    baseline = {
        "name": "current_rtl_energy",
        "raw_weight": "",
        "delta_weight": "",
        "leak_shift": "",
        "noise_shift": "",
        "noise_margin": "",
        "score_mean": float(np.mean(baseline_scores)),
        "score_max": float(np.max(baseline_scores)),
        **best_threshold(labels, baseline_scores),
    }

    rows = [baseline]
    for params in candidate_grid():
        rows.append(score_frame_model(labels, raw_frames, params))

    rows = sorted(rows, key=lambda row: (float(row["f1"]), float(row["accuracy"])), reverse=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "clip_count": len(pcm_per_clip),
        "duration_s": len(labels) * FRAME_MS / 1000.0,
        "speech_fraction": float(np.mean(labels)),
        "tested_models": len(rows),
        "best": rows[0],
        "top_20": rows[:20],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
