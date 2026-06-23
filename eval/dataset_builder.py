#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_rag_cli import parse_tidb_config, tidb_connect


@dataclass
class BenchmarkRow:
    query_id: str
    query: str
    video_ref: str
    windows: list[tuple[float, float]]


def read_benchmark_rows(path: Path) -> list[BenchmarkRow]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        raw_rows = json.loads(text)
    else:
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    rows = []
    for index, row in enumerate(raw_rows):
        if "ground_truth_windows" in row:
            windows = [
                (float(window["start_sec"]), float(window["end_sec"]))
                for window in row["ground_truth_windows"]
            ]
        elif "relevant_windows" in row:
            windows = [(float(start), float(end)) for start, end in row["relevant_windows"]]
        else:
            windows = [(float(row["start_sec"]), float(row["end_sec"]))]
        rows.append(
            BenchmarkRow(
                query_id=str(row.get("query_id") or index),
                query=row["query"],
                video_ref=row["video_ref"],
                windows=windows,
            )
        )
    return rows


def fetch_segments(connection, *, video_ref_column: str, method: str | None) -> dict[str, list[dict]]:
    filters = []
    params: list[object] = []
    if method:
        filters.append("vs.indexing_method = %s")
        params.append(method)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    if video_ref_column not in {"file_name", "file_path"}:
        raise SystemExit("--video-ref-column must be file_name or file_path")

    sql = f"""
        SELECT
          vs.id AS segment_id,
          v.{video_ref_column} AS video_ref,
          vs.start_sec,
          vs.end_sec,
          vs.indexing_method,
          vs.modality
        FROM video_segments vs
        JOIN videos v ON v.id = vs.video_id
        {where_sql}
        ORDER BY v.id ASC, vs.start_sec ASC, vs.id ASC
    """
    grouped: dict[str, list[dict]] = {}
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        for row in cursor.fetchall():
            grouped.setdefault(row["video_ref"], []).append(row)
    return grouped


def overlap_ratio(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
    *,
    mode: str,
) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if intersection <= 0:
        return 0.0
    segment_width = max(a_end - a_start, 1e-9)
    query_width = max(b_end - b_start, 1e-9)
    union = max(a_end, b_end) - min(a_start, b_start)

    if mode == "segment":
        return intersection / segment_width
    if mode == "query":
        return intersection / query_width
    if mode == "iou":
        return intersection / max(union, 1e-9)
    raise ValueError(f"Unknown overlap mode: {mode}")


def build_dataset(
    benchmark_rows: list[BenchmarkRow],
    segments_by_video: dict[str, list[dict]],
    *,
    overlap_threshold: float,
    overlap_mode: str,
) -> tuple[list[dict], dict]:
    dataset = []
    stats = {
        "total_queries": len(benchmark_rows),
        "kept_queries": 0,
        "dropped_queries": 0,
        "missing_videos": 0,
        "empty_relevant_ids": 0,
    }

    for row in benchmark_rows:
        segments = segments_by_video.get(row.video_ref)
        if not segments:
            stats["missing_videos"] += 1
            stats["dropped_queries"] += 1
            continue

        relevant_ids = set()
        for window_start, window_end in row.windows:
            for segment in segments:
                if (
                    overlap_ratio(
                        float(segment["start_sec"]),
                        float(segment["end_sec"]),
                        window_start,
                        window_end,
                        mode=overlap_mode,
                    )
                    >= overlap_threshold
                ):
                    relevant_ids.add(int(segment["segment_id"]))
        if not relevant_ids:
            stats["empty_relevant_ids"] += 1
            stats["dropped_queries"] += 1
            continue

        dataset.append(
            {
                "query_id": row.query_id,
                "query": row.query,
                "video_ref": row.video_ref,
                "ground_truth_windows": [
                    {"start_sec": window_start, "end_sec": window_end}
                    for window_start, window_end in row.windows
                ],
                "relevant_ids": sorted(relevant_ids),
            }
        )
        stats["kept_queries"] += 1

    return dataset, stats


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Build retrieval evaluation dataset from benchmark annotations")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON or JSONL")
    parser.add_argument("--output", required=True, help="Path to output dataset JSON")
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--overlap-mode", choices=["segment", "query", "iou"], default="segment")
    parser.add_argument("--method", choices=["transcript", "single_frame", "multi_frame", "video_clip"])
    parser.add_argument("--video-ref-column", choices=["file_name", "file_path"], default="file_name")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    benchmark_rows = read_benchmark_rows(benchmark_path)

    config = parse_tidb_config()
    connection = tidb_connect(config)
    try:
        segments_by_video = fetch_segments(
            connection,
            video_ref_column=args.video_ref_column,
            method=args.method,
        )
    finally:
        connection.close()

    dataset, stats = build_dataset(
        benchmark_rows,
        segments_by_video,
        overlap_threshold=args.overlap_threshold,
        overlap_mode=args.overlap_mode,
    )

    payload = {
        "metadata": {
            "benchmark_path": str(benchmark_path),
            "video_ref_column": args.video_ref_column,
            "method": args.method,
            "overlap_threshold": args.overlap_threshold,
            "overlap_mode": args.overlap_mode,
            "database": config.database,
        },
        "queries": dataset,
        "stats": stats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
