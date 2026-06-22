#!/usr/bin/env python3
"""Download a small AVA-Speech subset using VAD_Benchmark's label file."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_DATASET = ROOT / "external" / "VAD_Benchmark" / "dataset"
LABEL_PATH = BENCHMARK_DATASET / "ava_speech_labels_v1.csv"
OUT_DIR = ROOT / "ava_subset"


def unique_ids(label_path: Path) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    with label_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            video_id = row[0]
            if video_id not in seen:
                seen.add(video_id)
                ids.append(video_id)
    return ids


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def download_clip(video_id: str, start_s: int, duration_s: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}_{start_s}_{duration_s}s_16k.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url_lines = run([
        str(ROOT.parent.parent / ".venv" / "bin" / "yt-dlp"),
        "--no-warnings",
        "-f",
        "139/bestaudio/best",
        "--get-url",
        f"https://www.youtube.com/watch?v={video_id}",
    ]).splitlines()
    media_urls = [line for line in url_lines if line.startswith(("http://", "https://"))]
    if not media_urls:
        raise RuntimeError("yt-dlp did not return a media URL")
    subprocess.check_call([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-i",
        media_urls[0],
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--start-s", type=int, default=900)
    parser.add_argument("--duration-s", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=40)
    args = parser.parse_args()

    downloaded: list[Path] = []
    failures: list[tuple[str, str]] = []
    for video_id in unique_ids(LABEL_PATH)[: args.max_attempts]:
        try:
            path = download_clip(video_id, args.start_s, args.duration_s, OUT_DIR)
            print(f"downloaded {video_id}: {path}")
            downloaded.append(path)
            if len(downloaded) >= args.clips:
                break
        except Exception as exc:  # noqa: BLE001 - keep going through unavailable videos.
            failures.append((video_id, str(exc).splitlines()[-1] if str(exc) else repr(exc)))
            print(f"failed {video_id}: {failures[-1][1]}")

    manifest_path = OUT_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(str(path) for path in downloaded) + "\n")
    if failures:
        (OUT_DIR / "download_failures.txt").write_text(
            "\n".join(f"{video_id}: {reason}" for video_id, reason in failures) + "\n"
        )

    if len(downloaded) < args.clips:
        raise SystemExit(f"downloaded {len(downloaded)} clips, wanted {args.clips}")


if __name__ == "__main__":
    main()
