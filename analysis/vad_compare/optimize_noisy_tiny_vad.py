#!/usr/bin/env python3
"""Search tile-friendly VAD features for high synthetic-noise conditions."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compare_ava_subset import (
    FRAME_MS,
    FRAME_SAMPLES,
    LABEL_PATH,
    SUBSET_DIR,
    frame_labels,
    read_labels,
    read_wav_i16,
    score_threshold,
)
from heldout_validate_frequency_scan import BAND_SETS, STEPS, split_video_ids
from noise_robustness_sweep import add_white_noise
from plot_variant_pr_curves import (
    current_adaptive_scores,
    frequency_scanner_scores,
    raw_delta_scores,
)
from search_tiny_vad import best_threshold
from tiny_vad_models import pcm_to_density


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "tiny_vad_noisy_optimization.csv"
SUMMARY_PATH = ROOT / "tiny_vad_noisy_optimization_summary.json"
HIGH_NOISE_SNRS_DB = (5.0, 0.0, -5.0)
RNG_SEED = 74_101


@dataclass(frozen=True)
class Clip:
    path: str
    video_id: str
    labels: np.ndarray
    pcm: np.ndarray


@dataclass(frozen=True)
class FeatureSet:
    labels: np.ndarray
    raw: np.ndarray
    delta: np.ndarray
    zc: np.ndarray
    bands: np.ndarray


def load_clips() -> list[Clip]:
    labels_by_video = read_labels(LABEL_PATH)
    clips = []
    for wav_path in sorted(SUBSET_DIR.glob("*_16k.wav")):
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        pcm = read_wav_i16(wav_path)
        frame_count = len(pcm) // FRAME_SAMPLES
        labels = frame_labels(labels_by_video[video_id], float(start_s_text), frame_count)
        clips.append(Clip(
            path=wav_path.name,
            video_id=video_id,
            labels=labels,
            pcm=pcm[: frame_count * FRAME_SAMPLES],
        ))
    return clips


def noisy_versions(clips: list[Clip], snrs: tuple[float, ...]) -> list[tuple[Clip, float, np.ndarray]]:
    versions = []
    for clip_idx, clip in enumerate(clips):
        for snr in snrs:
            seed = RNG_SEED + clip_idx * 1000 + int((snr + 100.0) * 10.0)
            rng = np.random.default_rng(seed)
            versions.append((clip, snr, add_white_noise(clip.pcm, snr, rng)))
    return versions


def features_from_versions(versions: list[tuple[Clip, float, np.ndarray]]) -> FeatureSet:
    labels = []
    raw = []
    delta = []
    zc = []
    bands = []
    signs = []
    for clip, _snr, pcm in versions:
        frame_count = len(clip.labels)
        density = pcm_to_density(pcm).astype(np.int16) - 32
        frames = density[: frame_count * FRAME_SAMPLES].reshape((-1, FRAME_SAMPLES))
        frame_raw = np.abs(frames).sum(axis=1).astype(np.int32)
        raw.append(frame_raw)
        delta.append(np.abs(np.diff(frame_raw, prepend=frame_raw[0])).astype(np.int32))
        zc.append((frames[:, :-1] * frames[:, 1:] < 0).sum(axis=1).astype(np.int32))
        frame_bands = []
        for step in STEPS:
            if step not in signs:
                pass
            phase = (np.arange(FRAME_SAMPLES, dtype=np.int32) * step) & 0xFF
            sign = np.where(phase < 128, 1, -1).astype(np.int16)
            frame_bands.append(np.abs((frames * sign).sum(axis=1)).astype(np.int32))
        bands.append(np.stack(frame_bands, axis=1))
        labels.append(clip.labels)
    return FeatureSet(
        labels=np.concatenate(labels),
        raw=np.concatenate(raw),
        delta=np.concatenate(delta),
        zc=np.concatenate(zc),
        bands=np.concatenate(bands),
    )


def fast_best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = np.clip(scores, 0, 255).astype(np.uint8)
    pos_hist = np.bincount(scores[labels], minlength=256)
    neg_hist = np.bincount(scores[~labels], minlength=256)
    tp = np.cumsum(pos_hist[::-1])[::-1]
    fp = np.cumsum(neg_hist[::-1])[::-1]
    fn = tp[0] - tp
    tn = fp[0] - fp
    precision = np.divide(tp, tp + fp, out=np.zeros(256, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(256, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros(256, dtype=float), where=(precision + recall) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros(256, dtype=float), where=(tn + fp) != 0)
    balanced_accuracy = (recall + specificity) / 2.0
    accuracy = (tp + tn) / len(labels)
    # High-noise work should avoid both the all-speech and all-silence traps.
    idx = int(np.lexsort((accuracy, f1, balanced_accuracy))[-1])
    return {
        "threshold": idx,
        "tp": int(tp[idx]),
        "fp": int(fp[idx]),
        "fn": int(fn[idx]),
        "tn": int(tn[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "specificity": float(specificity[idx]),
        "balanced_accuracy": float(balanced_accuracy[idx]),
        "f1": float(f1[idx]),
        "accuracy": float(accuracy[idx]),
    }


def add_balanced_metrics(row: dict[str, float]) -> dict[str, float]:
    specificity = row["tn"] / (row["tn"] + row["fp"]) if row["tn"] + row["fp"] else 0.0
    row["specificity"] = specificity
    row["balanced_accuracy"] = (row["recall"] + specificity) / 2.0
    return row


def feature_scores(
    data: FeatureSet,
    band_indices: tuple[int, ...],
    band_mode: str,
    band_shift: int,
    raw_shift: int | None,
    delta_shift: int | None,
    zc_shift: int | None,
) -> np.ndarray:
    selected = data.bands[:, band_indices]
    band_max = selected.max(axis=1)
    band_min = selected.min(axis=1)

    if band_mode == "max":
        band_feature = band_max
    elif band_mode == "spread":
        band_feature = band_max - band_min
    elif band_mode == "excess_half":
        band_feature = np.maximum(0, band_max - (data.raw >> 1))
    elif band_mode == "excess_quarter":
        band_feature = np.maximum(0, band_max - (data.raw >> 2))
    elif band_mode == "spread_plus_excess":
        band_feature = (band_max - band_min) + np.maximum(0, band_max - (data.raw >> 2))
    elif band_mode == "voice_sum_minus_high_sum":
        voice = data.bands[:, [0, 1, 2]].sum(axis=1)
        high = data.bands[:, [3, 4]].sum(axis=1)
        band_feature = np.maximum(0, voice - high)
    elif band_mode == "voice_max_minus_high_max":
        voice = data.bands[:, [0, 1, 2]].max(axis=1)
        high = data.bands[:, [3, 4]].max(axis=1)
        band_feature = np.maximum(0, voice - high)
    elif band_mode == "voice_spread_minus_high_spread":
        voice = data.bands[:, [0, 1, 2]]
        high = data.bands[:, [3, 4]]
        band_feature = np.maximum(0, (voice.max(axis=1) - voice.min(axis=1)) - (high.max(axis=1) - high.min(axis=1)))
    else:
        raise ValueError(band_mode)

    score = band_feature >> band_shift
    if raw_shift is not None:
        score = score + (data.raw >> raw_shift)
    if delta_shift is not None:
        score = score + (data.delta >> delta_shift)
    if zc_shift is not None:
        score = score - (data.zc >> zc_shift)
    return np.clip(score, 0, 255).astype(np.uint8)


def baseline_row(
    name: str,
    fn,
    development_versions: list[tuple[Clip, float, np.ndarray]],
    validation_versions: list[tuple[Clip, float, np.ndarray]],
) -> dict[str, object]:
    development_labels = np.concatenate([clip.labels for clip, _snr, _pcm in development_versions])
    validation_labels = np.concatenate([clip.labels for clip, _snr, _pcm in validation_versions])
    development_scores = np.concatenate([
        fn(pcm)[: len(clip.labels)] for clip, _snr, pcm in development_versions
    ])
    validation_scores = np.concatenate([
        fn(pcm)[: len(clip.labels)] for clip, _snr, pcm in validation_versions
    ])
    tuned = best_threshold(development_labels, development_scores)
    tuned = add_balanced_metrics(tuned)
    validation = add_balanced_metrics(score_threshold(validation_labels, validation_scores, int(tuned["threshold"])))
    return {
        "name": name,
        "kind": "baseline",
        "selected_threshold": int(tuned["threshold"]),
        "development_f1": float(tuned["f1"]),
        "development_precision": float(tuned["precision"]),
        "development_recall": float(tuned["recall"]),
        "development_specificity": float(tuned["specificity"]),
        "development_balanced_accuracy": float(tuned["balanced_accuracy"]),
        "development_accuracy": float(tuned["accuracy"]),
        "validation_f1": float(validation["f1"]),
        "validation_precision": float(validation["precision"]),
        "validation_recall": float(validation["recall"]),
        "validation_specificity": float(validation["specificity"]),
        "validation_balanced_accuracy": float(validation["balanced_accuracy"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_tp": int(validation["tp"]),
        "validation_fp": int(validation["fp"]),
        "validation_fn": int(validation["fn"]),
        "validation_tn": int(validation["tn"]),
    }


def main() -> None:
    clips = load_clips()
    video_ids = sorted({clip.video_id for clip in clips})
    development_ids, validation_ids = split_video_ids(video_ids)
    development_clips = [clip for clip in clips if clip.video_id in development_ids]
    validation_clips = [clip for clip in clips if clip.video_id in validation_ids]

    development_versions = noisy_versions(development_clips, HIGH_NOISE_SNRS_DB)
    validation_versions = noisy_versions(validation_clips, HIGH_NOISE_SNRS_DB)
    development = features_from_versions(development_versions)
    validation = features_from_versions(validation_versions)

    rows = [
        baseline_row("current_adaptive_rtl", current_adaptive_scores, development_versions, validation_versions),
        baseline_row("raw_delta_onset", raw_delta_scores, development_versions, validation_versions),
        baseline_row(
            "freq_scan_dev_selected",
            lambda pcm: frequency_scanner_scores(pcm, validation_peeked=False),
            development_versions,
            validation_versions,
        ),
    ]

    band_modes = [
        "max",
        "spread",
        "excess_half",
        "excess_quarter",
        "spread_plus_excess",
        "voice_sum_minus_high_sum",
        "voice_max_minus_high_max",
        "voice_spread_minus_high_spread",
    ]
    for band_indices, band_mode, band_shift, raw_shift, delta_shift, zc_shift in itertools.product(
        BAND_SETS,
        band_modes,
        [3, 4, 5, 6, 7],
        [None, 9, 10, 11],
        [None, 6, 7],
        [None, 2, 3, 4],
    ):
        development_scores = feature_scores(
            development, band_indices, band_mode, band_shift, raw_shift, delta_shift, zc_shift
        )
        tuned = fast_best_threshold(development.labels, development_scores)
        validation_scores = feature_scores(
            validation, band_indices, band_mode, band_shift, raw_shift, delta_shift, zc_shift
        )
        validation_metrics = add_balanced_metrics(score_threshold(validation.labels, validation_scores, int(tuned["threshold"])))
        rows.append({
            "name": (
                f"noisy_{band_mode}_band{band_shift}_raw{raw_shift}_"
                f"delta{delta_shift}_zc{zc_shift}_steps"
                f"{'-'.join(str(STEPS[idx]) for idx in band_indices)}"
            ),
            "kind": "candidate",
            "band_mode": band_mode,
            "band_shift": band_shift,
            "raw_shift": "" if raw_shift is None else raw_shift,
            "delta_shift": "" if delta_shift is None else delta_shift,
            "zc_shift": "" if zc_shift is None else zc_shift,
            "band_steps": "-".join(str(STEPS[idx]) for idx in band_indices),
            "selected_threshold": int(tuned["threshold"]),
            "development_f1": float(tuned["f1"]),
            "development_precision": float(tuned["precision"]),
            "development_recall": float(tuned["recall"]),
            "development_specificity": float(tuned["specificity"]),
            "development_balanced_accuracy": float(tuned["balanced_accuracy"]),
            "development_accuracy": float(tuned["accuracy"]),
            "validation_f1": float(validation_metrics["f1"]),
            "validation_precision": float(validation_metrics["precision"]),
            "validation_recall": float(validation_metrics["recall"]),
            "validation_specificity": float(validation_metrics["specificity"]),
            "validation_balanced_accuracy": float(validation_metrics["balanced_accuracy"]),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_tp": int(validation_metrics["tp"]),
            "validation_fp": int(validation_metrics["fp"]),
            "validation_fn": int(validation_metrics["fn"]),
            "validation_tn": int(validation_metrics["tn"]),
        })

    selected = max(rows, key=lambda row: (float(row["development_balanced_accuracy"]), float(row["development_f1"]), float(row["development_accuracy"])))
    validation_best = max(rows, key=lambda row: (float(row["validation_balanced_accuracy"]), float(row["validation_f1"]), float(row["validation_accuracy"])))
    rows = sorted(rows, key=lambda row: (float(row["development_balanced_accuracy"]), float(row["development_f1"]), float(row["development_accuracy"])), reverse=True)

    with CSV_PATH.open("w", newline="") as f:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "synthetic white-noise candidates selected by development balanced accuracy over 5, 0, and -5 dB SNR",
        "split": {
            "development_video_ids": sorted(development_ids),
            "validation_video_ids": sorted(validation_ids),
            "development_duration_s": len(development.labels) * FRAME_MS / 1000.0,
            "validation_duration_s": len(validation.labels) * FRAME_MS / 1000.0,
        },
        "snr_levels_db": HIGH_NOISE_SNRS_DB,
        "frequency_step_map": {step: round(step * 16000 / 256, 2) for step in STEPS},
        "tested_rows": len(rows),
        "selected_by_development": selected,
        "best_if_peeking_at_validation": validation_best,
        "top_20_by_development": rows[:20],
        "csv": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
