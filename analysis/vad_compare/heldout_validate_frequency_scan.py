#!/usr/bin/env python3
"""Held-out validation for a tiny square-wave frequency-scanner VAD."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from compare_ava_subset import (
    FRAME_SAMPLES,
    LABEL_PATH,
    SUBSET_DIR,
    frame_labels,
    frame_scores,
    read_labels,
    read_wav_i16,
    score_threshold,
)
from search_tiny_vad import best_threshold
from tiny_vad_models import pcm_to_density, rtl_energy_from_pcm


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "tiny_vad_frequency_scan_summary.json"
CSV_PATH = ROOT / "tiny_vad_frequency_scan.csv"

# 8-bit phase increments at the 16 kHz density-window rate. These approximate
# 250, 500, 1000, 2000, and 3000 Hz square-wave mixers.
STEPS = [4, 8, 16, 32, 48]
BAND_SETS = [
    (0,), (1,), (2,), (3,), (4,),
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 3),
    (0, 2, 3), (1, 2, 3), (2, 3, 4), (0, 1, 2),
]


def split_video_ids(video_ids: list[str], validation_fraction: float = 0.25) -> tuple[set[str], set[str]]:
    if len(video_ids) < 2:
        raise ValueError("need at least two videos for a held-out split")
    validation_count = max(1, round(len(video_ids) * validation_fraction))
    validation_count = min(validation_count, len(video_ids) - 1)
    ranked = sorted(video_ids, key=lambda video_id: hashlib.sha256(video_id.encode()).hexdigest())
    validation_ids = set(ranked[:validation_count])
    return set(video_ids) - validation_ids, validation_ids


def load_clips() -> list[dict[str, object]]:
    labels_by_video = read_labels(LABEL_PATH)
    clips: list[dict[str, object]] = []
    for wav_path in sorted(SUBSET_DIR.glob("*_16k.wav")):
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        pcm = read_wav_i16(wav_path)
        rtl_scores = frame_scores(rtl_energy_from_pcm(pcm))
        labels = frame_labels(labels_by_video[video_id], float(start_s_text), len(rtl_scores))
        pcm = pcm[: len(rtl_scores) * FRAME_SAMPLES]
        density = pcm_to_density(pcm).astype(np.int16) - 32
        frames = density.reshape((-1, FRAME_SAMPLES)).astype(np.int16)
        raw = np.abs(frames).sum(axis=1).astype(np.int32)
        bands = []
        for step in STEPS:
            phase = (np.arange(FRAME_SAMPLES, dtype=np.int32) * step) & 0xFF
            sign = np.where(phase < 128, 1, -1).astype(np.int16)
            bands.append(np.abs((frames * sign).sum(axis=1)).astype(np.int32))
        clips.append({
            "path": wav_path.name,
            "video_id": video_id,
            "labels": labels,
            "raw": raw,
            "bands": np.stack(bands, axis=1),
            "rtl": rtl_scores,
        })
    return clips


def concat(clips: list[dict[str, object]], key: str) -> np.ndarray:
    return np.concatenate([clip[key] for clip in clips])  # type: ignore[index]


def simulate_score(
    raw: np.ndarray,
    band_feature: np.ndarray,
    raw_shift: int,
    delta_shift: int | None,
    band_shift: int,
    leak_shift: int,
) -> np.ndarray:
    score = 0
    prev_raw = 0
    out = np.zeros(len(raw), dtype=np.uint8)
    for idx, raw_value in enumerate(raw):
        raw_int = int(raw_value)
        feature = raw_int >> raw_shift
        if delta_shift is not None:
            feature += abs(raw_int - prev_raw) >> delta_shift
        feature += int(band_feature[idx]) >> band_shift
        feature = min(feature, 255)
        score = score - (score >> leak_shift) + feature
        score = min(score, 255)
        out[idx] = score
        prev_raw = raw_int
    return out


def simulate_candidate(
    clips: list[dict[str, object]],
    raw_shift: int,
    delta_shift: int | None,
    band_indices: tuple[int, ...],
    band_shift: int,
    band_mode: str,
    leak_shift: int,
) -> np.ndarray:
    outputs = []
    for clip in clips:
        selected = clip["bands"][:, band_indices]  # type: ignore[index]
        if band_mode == "sum":
            band_feature = selected.sum(axis=1)
        elif band_mode == "max":
            band_feature = selected.max(axis=1)
        elif band_mode == "spread":
            band_feature = selected.max(axis=1) - selected.min(axis=1)
        else:
            raise ValueError(band_mode)
        outputs.append(simulate_score(
            clip["raw"],  # type: ignore[arg-type]
            band_feature,
            raw_shift,
            delta_shift,
            band_shift,
            leak_shift,
        ))
    return np.concatenate(outputs)


def evaluate_at(labels: np.ndarray, scores: np.ndarray, threshold: int) -> dict[str, float]:
    return score_threshold(labels, scores, threshold)


def candidate_rows(
    development_clips: list[dict[str, object]],
    validation_clips: list[dict[str, object]],
) -> list[dict[str, object]]:
    development_labels = concat(development_clips, "labels")
    validation_labels = concat(validation_clips, "labels")
    rows = []
    for raw_shift, delta_shift, band_shift, band_mode, leak_shift, band_indices in itertools.product(
        [7, 8, 9],
        [None, 4, 5, 6],
        [5, 6, 7, 8, 9],
        ["sum", "max", "spread"],
        [5, 6],
        BAND_SETS,
    ):
        development_scores = simulate_candidate(
            development_clips,
            raw_shift,
            delta_shift,
            band_indices,
            band_shift,
            band_mode,
            leak_shift,
        )
        tuned = best_threshold(development_labels, development_scores)
        validation_scores = simulate_candidate(
            validation_clips,
            raw_shift,
            delta_shift,
            band_indices,
            band_shift,
            band_mode,
            leak_shift,
        )
        validation = evaluate_at(validation_labels, validation_scores, int(tuned["threshold"]))
        rows.append({
            "name": (
                f"freq_raw{raw_shift}_delta{delta_shift}_band{band_shift}_"
                f"{band_mode}_leak{leak_shift}_steps"
                f"{'-'.join(str(STEPS[idx]) for idx in band_indices)}"
            ),
            "raw_shift": raw_shift,
            "delta_shift": "" if delta_shift is None else delta_shift,
            "band_shift": band_shift,
            "band_mode": band_mode,
            "leak_shift": leak_shift,
            "band_steps": "-".join(str(STEPS[idx]) for idx in band_indices),
            "selected_threshold": int(tuned["threshold"]),
            "development_f1": float(tuned["f1"]),
            "development_precision": float(tuned["precision"]),
            "development_recall": float(tuned["recall"]),
            "development_accuracy": float(tuned["accuracy"]),
            "validation_f1": float(validation["f1"]),
            "validation_precision": float(validation["precision"]),
            "validation_recall": float(validation["recall"]),
            "validation_accuracy": float(validation["accuracy"]),
            "validation_tp": int(validation["tp"]),
            "validation_fp": int(validation["fp"]),
            "validation_fn": int(validation["fn"]),
            "validation_tn": int(validation["tn"]),
        })
    return rows


def current_rtl_row(
    development_clips: list[dict[str, object]],
    validation_clips: list[dict[str, object]],
) -> dict[str, object]:
    development_labels = concat(development_clips, "labels")
    validation_labels = concat(validation_clips, "labels")
    tuned = best_threshold(development_labels, concat(development_clips, "rtl"))
    validation = evaluate_at(validation_labels, concat(validation_clips, "rtl"), int(tuned["threshold"]))
    return {
        "name": "current_rtl",
        "selected_threshold": int(tuned["threshold"]),
        "development_f1": float(tuned["f1"]),
        "development_precision": float(tuned["precision"]),
        "development_recall": float(tuned["recall"]),
        "development_accuracy": float(tuned["accuracy"]),
        "validation_f1": float(validation["f1"]),
        "validation_precision": float(validation["precision"]),
        "validation_recall": float(validation["recall"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_tp": int(validation["tp"]),
        "validation_fp": int(validation["fp"]),
        "validation_fn": int(validation["fn"]),
        "validation_tn": int(validation["tn"]),
    }


def main() -> None:
    clips = load_clips()
    video_ids = sorted({str(clip["video_id"]) for clip in clips})
    development_ids, validation_ids = split_video_ids(video_ids)
    development_clips = [clip for clip in clips if clip["video_id"] in development_ids]
    validation_clips = [clip for clip in clips if clip["video_id"] in validation_ids]

    current = current_rtl_row(development_clips, validation_clips)
    rows = candidate_rows(development_clips, validation_clips)
    selected = max(rows, key=lambda row: (float(row["development_f1"]), float(row["development_accuracy"])))
    validation_best = max(rows, key=lambda row: (float(row["validation_f1"]), float(row["validation_accuracy"])))
    rows = sorted(rows, key=lambda row: (float(row["development_f1"]), float(row["development_accuracy"])), reverse=True)

    with CSV_PATH.open("w", newline="") as f:
        fieldnames = sorted(set().union(*(row.keys() for row in rows), current.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(current)
        writer.writerows(rows)

    summary = {
        "method": "square-wave frequency scanner candidates selected on development clips only",
        "split": {
            "development_video_ids": sorted(development_ids),
            "validation_video_ids": sorted(validation_ids),
        },
        "frequency_step_map": {step: round(step * 16000 / 256, 2) for step in STEPS},
        "current_rtl": current,
        "tested_frequency_candidates": len(rows),
        "selected_frequency_candidate": selected,
        "best_frequency_candidate_if_peeking_at_validation": validation_best,
        "rtl_port_recommended": float(selected["validation_f1"]) > float(current["validation_f1"]),
        "top_20_by_development": rows[:20],
        "csv": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
