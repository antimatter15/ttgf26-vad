#!/usr/bin/env python3
"""Coordinate-search square-wave mixer steps for the tiny VAD."""

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
    read_labels,
    read_wav_i16,
    score_threshold,
)
from heldout_validate_frequency_scan import split_video_ids
from noise_robustness_sweep import add_white_noise
from search_tiny_vad import best_threshold
from tiny_vad_models import pcm_to_density


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "spectral_step_tuning.csv"
SUMMARY_PATH = ROOT / "spectral_step_tuning_summary.json"

SAMPLE_RATE = 16_000
SNRS = (0.0, -5.0)
CURRENT_STEPS = (4, 8, 16)
STEP_RANGES = (
    range(2, 9),
    range(5, 14),
    range(10, 25),
)
RAW_SHIFT = 10
DELTA_SHIFT = 4
BAND_SHIFT = 7
LEAK_SHIFT = 6
RNG_SEED = 91_260


def load_versions(video_ids: set[str]) -> list[dict[str, object]]:
    labels_by_video = read_labels(LABEL_PATH)
    versions = []
    for clip_index, wav_path in enumerate(sorted(SUBSET_DIR.glob("*_16k.wav"))):
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        if video_id not in video_ids:
            continue
        pcm = read_wav_i16(wav_path)
        frame_count = len(pcm) // FRAME_SAMPLES
        pcm = pcm[: frame_count * FRAME_SAMPLES]
        labels = frame_labels(labels_by_video[video_id], float(start_s_text), frame_count)
        for snr in SNRS:
            rng = np.random.default_rng(RNG_SEED + clip_index * 1000 + int((snr + 100.0) * 10.0))
            noisy = add_white_noise(pcm, snr, rng)
            density = pcm_to_density(noisy).astype(np.int16) - 32
            frames = density.reshape((-1, FRAME_SAMPLES)).astype(np.int16)
            raw = np.abs(frames).sum(axis=1).astype(np.int32)
            versions.append({
                "video_id": video_id,
                "clip": wav_path.name,
                "snr": snr,
                "labels": labels,
                "raw": raw,
                "frames": frames,
            })
    return versions


def square_band(frames: np.ndarray, step: int) -> np.ndarray:
    phase = (np.arange(FRAME_SAMPLES, dtype=np.int32) * step) & 0xFF
    sign = np.where(phase < 128, 1, -1).astype(np.int16)
    return np.abs((frames * sign).sum(axis=1)).astype(np.int32)


def scores_for_versions(versions: list[dict[str, object]], steps: tuple[int, int, int]) -> np.ndarray:
    outputs = []
    for version in versions:
        raw = version["raw"]  # type: ignore[assignment]
        frames = version["frames"]  # type: ignore[assignment]
        band_sum = sum(square_band(frames, step) for step in steps)
        score = 0
        prev_raw = 0
        out = np.zeros(len(raw), dtype=np.uint8)
        for idx, (raw_value, band_value) in enumerate(zip(raw, band_sum)):
            raw_int = int(raw_value)
            feature = (
                (raw_int >> RAW_SHIFT)
                + (abs(raw_int - prev_raw) >> DELTA_SHIFT)
                + (int(band_value) >> BAND_SHIFT)
            )
            feature = min(feature, 255)
            score = score - (score >> LEAK_SHIFT) + feature
            score = min(score, 255)
            out[idx] = score
            prev_raw = raw_int
        outputs.append(out)
    return np.concatenate(outputs)


def labels_for_versions(versions: list[dict[str, object]]) -> np.ndarray:
    return np.concatenate([version["labels"] for version in versions])  # type: ignore[list-item]


def evaluate(versions: list[dict[str, object]], labels: np.ndarray, steps: tuple[int, int, int]) -> dict[str, object]:
    scores = scores_for_versions(versions, steps)
    tuned = best_threshold(labels, scores)
    return {
        "steps": "-".join(str(step) for step in steps),
        "freq_hz": "-".join(f"{step * SAMPLE_RATE / 256:.1f}" for step in steps),
        "threshold": int(tuned["threshold"]),
        "precision": float(tuned["precision"]),
        "recall": float(tuned["recall"]),
        "f1": float(tuned["f1"]),
        "accuracy": float(tuned["accuracy"]),
    }


def evaluate_at_threshold(
    versions: list[dict[str, object]],
    labels: np.ndarray,
    steps: tuple[int, int, int],
    threshold: int,
) -> dict[str, object]:
    scores = scores_for_versions(versions, steps)
    row = score_threshold(labels, scores, threshold)
    return {
        "steps": "-".join(str(step) for step in steps),
        "freq_hz": "-".join(f"{step * SAMPLE_RATE / 256:.1f}" for step in steps),
        "threshold": threshold,
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "accuracy": float(row["accuracy"]),
    }


def coordinate_search(versions: list[dict[str, object]], labels: np.ndarray) -> tuple[tuple[int, int, int], list[dict[str, object]]]:
    current = CURRENT_STEPS
    rows = []
    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1
        current_eval = evaluate(versions, labels, current)
        current_score = (float(current_eval["f1"]), float(current_eval["accuracy"]))
        rows.append({"iteration": iteration, "kind": "incumbent", **current_eval})

        for band_index, candidates in enumerate(STEP_RANGES):
            best_steps = current
            best_eval = current_eval
            best_score = current_score
            for step in candidates:
                proposal = list(current)
                proposal[band_index] = step
                proposal_tuple = tuple(proposal)
                if len(set(proposal_tuple)) != len(proposal_tuple) or proposal_tuple != tuple(sorted(proposal_tuple)):
                    continue
                result = evaluate(versions, labels, proposal_tuple)
                score = (float(result["f1"]), float(result["accuracy"]))
                rows.append({
                    "iteration": iteration,
                    "kind": f"try_band_{band_index}",
                    **result,
                })
                if score > best_score:
                    best_steps = proposal_tuple
                    best_eval = result
                    best_score = score
            if best_steps != current:
                current = best_steps
                current_eval = best_eval
                current_score = best_score
                improved = True

    return current, rows


def main() -> None:
    all_video_ids = sorted({path.stem.rsplit("_", 3)[0] for path in SUBSET_DIR.glob("*_16k.wav")})
    development_ids, validation_ids = split_video_ids(all_video_ids)
    development_versions = load_versions(development_ids)
    validation_versions = load_versions(validation_ids)
    pooled_versions = load_versions(set(all_video_ids))
    development_labels = labels_for_versions(development_versions)
    validation_labels = labels_for_versions(validation_versions)
    pooled_labels = labels_for_versions(pooled_versions)

    selected_steps, rows = coordinate_search(development_versions, development_labels)
    selected_development = evaluate(development_versions, development_labels, selected_steps)
    selected_validation = evaluate_at_threshold(
        validation_versions,
        validation_labels,
        selected_steps,
        int(selected_development["threshold"]),
    )
    current_development = evaluate(development_versions, development_labels, CURRENT_STEPS)
    current_validation = evaluate_at_threshold(
        validation_versions,
        validation_labels,
        CURRENT_STEPS,
        int(current_development["threshold"]),
    )
    selected_pooled = evaluate(pooled_versions, pooled_labels, selected_steps)
    current_pooled = evaluate(pooled_versions, pooled_labels, CURRENT_STEPS)

    with CSV_PATH.open("w", newline="") as f:
        fieldnames = ["iteration", "kind", "steps", "freq_hz", "threshold", "precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "coordinate search over 8-bit square-wave phase steps, selected on development clips at 0 and -5 dB",
        "step_resolution_hz": SAMPLE_RATE / 256,
        "weights": {
            "raw_shift": RAW_SHIFT,
            "delta_shift": DELTA_SHIFT,
            "band_shift": BAND_SHIFT,
            "leak_shift": LEAK_SHIFT,
        },
        "split": {
            "development_video_ids": sorted(development_ids),
            "validation_video_ids": sorted(validation_ids),
            "snr_levels_db": SNRS,
            "development_duration_s": len(development_labels) * FRAME_MS / 1000.0,
            "validation_duration_s": len(validation_labels) * FRAME_MS / 1000.0,
        },
        "current_development": current_development,
        "current_validation_at_development_threshold": current_validation,
        "selected_development": selected_development,
        "selected_validation_at_development_threshold": selected_validation,
        "current_pooled_best_threshold": current_pooled,
        "selected_pooled_best_threshold": selected_pooled,
        "csv": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
