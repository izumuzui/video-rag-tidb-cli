#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"{name} is required but was not found in PATH.")


def read_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_vid_from_path(video_path: str) -> tuple[str, float, float]:
    stem = Path(video_path).stem
    youtube_id, start_raw, end_raw = stem.rsplit("_", 2)
    return youtube_id, float(start_raw), float(end_raw)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def download_clip(item: dict, output_dir: Path) -> dict:
    video_path = item["video_path"]
    youtube_id, start_sec, end_sec = parse_vid_from_path(video_path)
    output_path = output_dir / Path(video_path).name
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"video_path": str(output_path), "status": "skipped"}

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    tmp_template = str(output_dir / f"{Path(video_path).stem}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--download-sections",
        f"*{start_sec}-{end_sec}",
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "-o",
        tmp_template,
        url,
    ]
    run(cmd)
    if not output_path.exists():
        candidates = list(output_dir.glob(f"{Path(video_path).stem}.*"))
        if not candidates:
            raise SystemExit(f"Downloaded file not found for {youtube_id}")
        candidate = candidates[0]
        candidate.rename(output_path)
    return {"video_path": str(output_path), "status": "downloaded"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download QVHighlights video clips from a manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_tool("yt-dlp")
    ensure_tool("ffmpeg")
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = read_manifest(manifest_path)
    if args.limit is not None:
        items = items[: args.limit]

    results = []
    for index, item in enumerate(items, start=1):
        try:
            result = download_clip(item, output_dir)
        except Exception as exc:
            result = {
                "video_path": item["video_path"],
                "status": "failed",
                "error": str(exc),
            }
        result["index"] = index
        result["video_key"] = item.get("video_key")
        results.append(result)

    print(json.dumps({"manifest": str(manifest_path), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
