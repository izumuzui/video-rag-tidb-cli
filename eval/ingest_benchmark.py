#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_rag_cli import (
    build_segments,
    cmd_load_tidb,
    make_provider,
    write_jsonl,
)


def read_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Index benchmark videos and optionally load them into TiDB")
    parser.add_argument("--manifest", required=True, help="JSON/JSONL with video_path entries")
    parser.add_argument("--workdir", required=True, help="Directory for generated artifacts")
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--methods", default="transcript,single_frame,multi_frame,video_clip")
    parser.add_argument("--provider", choices=["mock", "gemini"], default="gemini")
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--multi-frame-count", type=int, default=3)
    parser.add_argument("--output-summary", default=None, help="Optional path to write ingest summary JSON")
    parser.add_argument("--skip-load-tidb", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    provider = make_provider(args.provider, args.gemini_model, args.embedding_model)
    manifest_rows = read_manifest(manifest_path)

    summary = []
    for item in manifest_rows:
        video_path = Path(item["video_path"]).expanduser().resolve()
        video_key = item.get("video_key") or video_path.stem
        video_workdir = workdir / video_key
        segments = build_segments(
            video_path,
            video_workdir,
            args.segment_seconds,
            methods,
            provider,
            args.multi_frame_count,
        )
        index_path = video_workdir / "segments.jsonl"
        write_jsonl(index_path, (asdict(segment) for segment in segments))
        summary.append(
            {
                "video_key": video_key,
                "video_path": str(video_path),
                "index_path": str(index_path),
                "segments": len(segments),
            }
        )

        if not args.skip_load_tidb:
            load_args = argparse.Namespace(index=str(index_path))
            cmd_load_tidb(load_args)

    payload = {"manifest": str(manifest_path), "videos": summary}
    if args.output_summary:
        output_summary = Path(args.output_summary).expanduser().resolve()
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
