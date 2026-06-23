#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> list[dict] | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter available and loaded assets for retrieval evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-manifest-available", required=True)
    parser.add_argument("--output-queries-available", required=True)
    parser.add_argument("--ingest-summary", default=None)
    parser.add_argument("--output-queries-loaded", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    queries_path = Path(args.queries).expanduser().resolve()
    manifest = read_json(manifest_path)
    queries = read_json(queries_path)

    available_manifest = [item for item in manifest if Path(item["video_path"]).exists()]
    available_names = {Path(item["video_path"]).name for item in available_manifest}
    available_queries = [item for item in queries if item["video_ref"] in available_names]

    output_manifest_available = Path(args.output_manifest_available).expanduser().resolve()
    output_queries_available = Path(args.output_queries_available).expanduser().resolve()
    write_json(output_manifest_available, available_manifest)
    write_json(output_queries_available, available_queries)

    payload = {
        "available_videos": len(available_manifest),
        "available_queries": len(available_queries),
        "output_manifest_available": str(output_manifest_available),
        "output_queries_available": str(output_queries_available),
    }

    if args.ingest_summary and args.output_queries_loaded:
        ingest_summary_path = Path(args.ingest_summary).expanduser().resolve()
        ingest_summary = read_json(ingest_summary_path)
        loaded_video_names = {Path(item["video_path"]).name for item in ingest_summary.get("videos", [])}
        loaded_queries = [item for item in available_queries if item["video_ref"] in loaded_video_names]
        output_queries_loaded = Path(args.output_queries_loaded).expanduser().resolve()
        write_json(output_queries_loaded, loaded_queries)
        payload["loaded_videos"] = len(loaded_video_names)
        payload["loaded_queries"] = len(loaded_queries)
        payload["output_queries_loaded"] = str(output_queries_loaded)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
