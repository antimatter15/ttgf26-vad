#!/usr/bin/env python3
"""Tune tiny VAD candidates on development clips and evaluate held-out clips."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
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
from search_tiny_vad import best_threshold, candidate_grid, frame_sum, simulate_frame_candidate
from tiny_vad_models import CandidateParams, pcm_to_density, rtl_energy_from_pcm


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "tiny_vad_heldout_validation.csv"
SUMMARY_PATH = ROOT / "tiny_vad_heldout_validation_summary.json"


@dataclass(frozen=True)
class Clip:
    path: str
    video_id: str
    duration_s: float
    speech_fraction: float
    labels: np.ndarray
    raw_frames: np.ndarray
    rtl_scores: np.ndarray


def split_video_ids(video_ids: list[str], validation_fraction: float = 0.25) -> tuple[set[str], set[str]]:
    """Stable split by video id, with no frame-level leakage."""
    if len(video_ids) < 2:
        raise ValueError("need at least two videos for a held-out split")
    validation_count = max(1, round(len(video_ids) * validation_fraction))
    validation_count = min(validation_count, len(video_ids) - 1)
    ranked = sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
    )
    validation_ids = set(ranked[:validation_count])
    development_ids = set(video_ids) - validation_ids
    return development_ids, validation_ids


def load_clips() -> list[Clip]:
    labels_by_video = read_labels(LABEL_PATH)
    clips: list[Clip] = []
    for wav_path in sorted(SUBSET_DIR.glob("*_16k.wav")):
        video_id, start_s_text, _duration_text, _sr = wav_path.stem.rsplit("_", 3)
        clip_start_s = float(start_s_text)
        pcm = read_wav_i16(wav_path)
        rtl_scores = frame_scores(rtl_energy_from_pcm(pcm))
        frame_count = len(rtl_scores)
        labels = frame_labels(labels_by_video[video_id], clip_start_s, frame_count)
        pcm = pcm[: frame_count * FRAME_SAMPLES]
        density = pcm_to_density(pcm).astype(np.int16) - 32
        raw_frames = frame_sum(np.abs(density))
        clips.append(Clip(
            path=wav_path.name,
            video_id=video_id,
            duration_s=frame_count * FRAME_MS / 1000.0,
            speech_fraction=float(np.mean(labels)),
            labels=labels,
            raw_frames=raw_frames,
            rtl_scores=rtl_scores,
        ))
    return clips


def concatenate_labels(clips: list[Clip]) -> np.ndarray:
    return np.concatenate([clip.labels for clip in clips])


def simulate_candidate_for_clips(clips: list[Clip], params: CandidateParams) -> np.ndarray:
    return simulate_frame_candidate([clip.raw_frames for clip in clips], params)


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: int) -> dict[str, float]:
    return score_threshold(labels, scores, threshold)


def row_for_candidate(
    params: CandidateParams,
    development_clips: list[Clip],
    validation_clips: list[Clip],
) -> dict[str, object]:
    development_labels = concatenate_labels(development_clips)
    validation_labels = concatenate_labels(validation_clips)
    development_scores = simulate_candidate_for_clips(development_clips, params)
    validation_scores = simulate_candidate_for_clips(validation_clips, params)
    tuned = best_threshold(development_labels, development_scores)
    validation = evaluate(validation_labels, validation_scores, int(tuned["threshold"]))
    return {
        "name": params.name,
        "raw_weight": params.raw_weight,
        "delta_weight": params.delta_weight,
        "leak_shift": params.leak_shift,
        "noise_shift": "" if params.noise_shift is None else params.noise_shift,
        "noise_margin": params.noise_margin,
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


def row_for_current_rtl(development_clips: list[Clip], validation_clips: list[Clip]) -> dict[str, object]:
    development_labels = concatenate_labels(development_clips)
    validation_labels = concatenate_labels(validation_clips)
    development_scores = np.concatenate([clip.rtl_scores for clip in development_clips])
    validation_scores = np.concatenate([clip.rtl_scores for clip in validation_clips])
    tuned = best_threshold(development_labels, development_scores)
    validation = evaluate(validation_labels, validation_scores, int(tuned["threshold"]))
    return {
        "name": "current_rtl",
        "raw_weight": "raw>>7",
        "delta_weight": "zero_crossings<<1",
        "leak_shift": 6,
        "noise_shift": "",
        "noise_margin": 0,
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
    video_ids = sorted({clip.video_id for clip in clips})
    development_ids, validation_ids = split_video_ids(video_ids)
    development_clips = [clip for clip in clips if clip.video_id in development_ids]
    validation_clips = [clip for clip in clips if clip.video_id in validation_ids]

    rows = [row_for_current_rtl(development_clips, validation_clips)]
    rows.extend(row_for_candidate(params, development_clips, validation_clips) for params in candidate_grid())
    selected = max(rows, key=lambda row: (float(row["development_f1"]), float(row["development_accuracy"])))
    validation_best = max(rows, key=lambda row: (float(row["validation_f1"]), float(row["validation_accuracy"])))
    rows = sorted(rows, key=lambda row: (float(row["development_f1"]), float(row["validation_f1"])), reverse=True)

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "parameters and threshold selected on development clips only; selected model evaluated once on held-out validation clips",
        "split": {
            "development_video_ids": sorted(development_ids),
            "validation_video_ids": sorted(validation_ids),
            "development_duration_s": sum(clip.duration_s for clip in development_clips),
            "validation_duration_s": sum(clip.duration_s for clip in validation_clips),
        },
        "clips": [asdict(clip) | {"labels": None, "raw_frames": None, "rtl_scores": None} for clip in clips],
        "tested_models": len(rows),
        "selected_by_development": selected,
        "best_if_peeking_at_validation": validation_best,
        "top_20_by_development": rows[:20],
        "csv": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
