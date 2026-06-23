#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_rows(rows: list[dict], *, limit_queries: int | None, max_videos: int | None) -> list[dict]:
    selected = []
    seen_videos: set[str] = set()
    for row in rows:
        video_id = row["vid"]
        if max_videos is not None and video_id not in seen_videos and len(seen_videos) >= max_videos:
            continue
        selected.append(row)
        seen_videos.add(video_id)
        if limit_queries is not None and len(selected) >= limit_queries:
            break
    return selected


def build_benchmark_queries(rows: list[dict], *, split_name: str) -> list[dict]:
    queries = []
    for row in rows:
        if not row.get("relevant_windows"):
            continue
        queries.append(
            {
                "query_id": f"{split_name}-{row['qid']}",
                "query": row["query"],
                "video_ref": f"{row['vid']}.mp4",
                "ground_truth_windows": [
                    {"start_sec": float(start), "end_sec": float(end)}
                    for start, end in row["relevant_windows"]
                ],
                "source": {
                    "dataset": "QVHighlights",
                    "split": split_name,
                    "qid": row["qid"],
                    "vid": row["vid"],
                    "duration": row.get("duration"),
                    "relevant_clip_ids": row.get("relevant_clip_ids", []),
                },
            }
        )
    return queries


def build_manifest(rows: list[dict], *, videos_dir: Path, extension: str) -> list[dict]:
    manifest = []
    seen = set()
    for row in rows:
        vid = row["vid"]
        if vid in seen:
            continue
        seen.add(vid)
        manifest.append(
            {
                "video_key": vid,
                "video_path": str((videos_dir / f"{vid}.{extension}").resolve()),
            }
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare QVHighlights annotations for local retrieval evaluation")
    parser.add_argument("--annotations", required=True, help="Path to QVHighlights JSONL annotation file")
    parser.add_argument("--output-queries", required=True, help="Path to benchmark query JSON")
    parser.add_argument("--output-manifest", default=None, help="Optional path to benchmark video manifest JSON")
    parser.add_argument("--videos-dir", default=None, help="Directory containing downloaded QVHighlights videos")
    parser.add_argument("--video-extension", default="mp4", help="File extension used for local videos")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    annotations_path = Path(args.annotations).expanduser().resolve()
    output_queries = Path(args.output_queries).expanduser().resolve()
    rows = read_jsonl(annotations_path)
    rows = select_rows(rows, limit_queries=args.limit_queries, max_videos=args.max_videos)

    queries = build_benchmark_queries(rows, split_name=args.split_name)
    write_json(output_queries, queries)

    outputs = {
        "annotations": str(annotations_path),
        "queries": str(output_queries),
        "query_count": len(queries),
        "video_count": len({row['vid'] for row in rows}),
    }

    if args.output_manifest:
        if not args.videos_dir:
            raise SystemExit("--videos-dir is required when --output-manifest is used")
        output_manifest = Path(args.output_manifest).expanduser().resolve()
        manifest = build_manifest(
            rows,
            videos_dir=Path(args.videos_dir).expanduser().resolve(),
            extension=args.video_extension.lstrip("."),
        )
        write_json(output_manifest, manifest)
        outputs["manifest"] = str(output_manifest)

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
