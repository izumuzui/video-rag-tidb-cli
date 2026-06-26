#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_rag_cli import (
    keyword_search_tidb,
    make_provider,
    parse_tidb_config,
    reciprocal_rank_fusion,
    tidb_connect,
    vector_search_tidb,
)

from eval.metrics import hit_rate, mean, mrr, ndcg_at_k
from eval.reranker import CrossEncoderReranker, DEFAULT_RERANK_MODEL, rerank_hits


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_search(
    connection,
    provider,
    *,
    mode: str,
    query: str,
    top_k: int,
    method: str | None,
    rerank_pool_size: int,
    reranker: CrossEncoderReranker | None,
):
    if mode == "keyword":
        return keyword_search_tidb(connection, query, top_k, method), None
    query_embedding = provider.embed_text(query)
    if mode == "vector":
        return vector_search_tidb(connection, query_embedding, top_k, method), None
    keyword_hits = keyword_search_tidb(connection, query, rerank_pool_size, method)
    vector_hits = vector_search_tidb(connection, query_embedding, rerank_pool_size, method)
    hybrid_hits = reciprocal_rank_fusion(keyword_hits, vector_hits, rerank_pool_size)
    if mode == "hybrid":
        return hybrid_hits[:top_k], None
    if reranker is None:
        raise RuntimeError("reranker is required for hybrid_rerank mode")
    reranked, debug_rows = rerank_hits(query, hybrid_hits, reranker, top_k=top_k)
    return reranked, {
        "query": query,
        "pool_size": rerank_pool_size,
        "hybrid_hits": [
            {
                "segment_id": hit.segment_id,
                "score": hit.score,
                "score_type": hit.score_type,
                "content_preview": hit.content[:160],
            }
            for hit in hybrid_hits
        ],
        "reranked_hits": debug_rows,
    }


def format_table(rows: list[dict], ks: list[int]) -> str:
    headers = ["mode", "queries", "MRR"] + [f"Hit@{k}" for k in ks] + [f"nDCG@{k}" for k in ks]
    widths = {header: len(header) for header in headers}
    for row in rows:
        values = build_row_values(row, ks)
        for header, value in zip(headers, values):
            widths[header] = max(widths[header], len(value))

    lines = []
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        values = build_row_values(row, ks)
        lines.append(" | ".join(value.ljust(widths[header]) for header, value in zip(headers, values)))
    return "\n".join(lines)


def build_row_values(row: dict, ks: list[int]) -> list[str]:
    values = [
        row["mode"],
        str(row["query_count"]),
        f'{row["mrr"]:.4f}',
    ]
    values.extend(f'{row[f"hit@{k}"]:.4f}' for k in ks)
    values.extend(f'{row[f"ndcg@{k}"]:.4f}' for k in ks)
    return values


def write_csv(path: Path, rows: list[dict], ks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["mode", "query_count", "mrr"] + [f"hit@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header) for header in headers})


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run retrieval evaluation against TiDB-backed search")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset JSON")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--method", choices=["transcript", "single_frame", "multi_frame", "video_clip"])
    parser.add_argument("--provider", choices=["mock", "gemini"], default="gemini")
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--modes", default="vector,hybrid,hybrid_rerank")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-device", default=None)
    parser.add_argument("--rerank-pool-size", type=int, default=20)
    parser.add_argument("--dump-rerank-details", default=None)
    parser.add_argument("--dump-rerank-limit", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    dump_rerank_details = Path(args.dump_rerank_details).expanduser().resolve() if args.dump_rerank_details else None
    ks = [int(item) for item in args.ks.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    dataset = load_dataset(dataset_path)
    queries = dataset["queries"]

    config = parse_tidb_config()
    connection = tidb_connect(config)
    provider = make_provider(args.provider, args.gemini_model, args.embedding_model)
    reranker = None
    if "hybrid_rerank" in modes:
        reranker = CrossEncoderReranker(args.rerank_model, device=args.rerank_device)

    try:
        summaries = []
        details = []
        rerank_debug = []
        for mode in modes:
            per_query = []
            for item in queries:
                hits, rerank_detail = run_search(
                    connection,
                    provider,
                    mode=mode,
                    query=item["query"],
                    top_k=args.top_k,
                    method=args.method,
                    rerank_pool_size=args.rerank_pool_size,
                    reranker=reranker,
                )
                retrieved_ids = [hit.segment_id for hit in hits]
                relevant_ids = set(int(value) for value in item["relevant_ids"])
                metrics = {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "mode": mode,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": sorted(relevant_ids),
                    "mrr": mrr(retrieved_ids, relevant_ids),
                }
                for k in ks:
                    metrics[f"hit@{k}"] = hit_rate(retrieved_ids, relevant_ids, k)
                    metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
                per_query.append(metrics)
                details.append(metrics)
                if rerank_detail and len(rerank_debug) < args.dump_rerank_limit:
                    rerank_debug.append(
                        {
                            "query_id": item["query_id"],
                            "mode": mode,
                            "relevant_ids": sorted(relevant_ids),
                            **rerank_detail,
                        }
                    )

            summary = {
                "mode": mode,
                "query_count": len(per_query),
                "mrr": mean(item["mrr"] for item in per_query),
            }
            for k in ks:
                summary[f"hit@{k}"] = mean(item[f"hit@{k}"] for item in per_query)
                summary[f"ndcg@{k}"] = mean(item[f"ndcg@{k}"] for item in per_query)
            summaries.append(summary)
    finally:
        connection.close()

    payload = {
        "metadata": {
            "dataset": str(dataset_path),
            "database": config.database,
            "top_k": args.top_k,
            "ks": ks,
            "modes": modes,
            "method": args.method,
            "provider": args.provider,
            "rerank_model": args.rerank_model if "hybrid_rerank" in modes else None,
            "rerank_device": args.rerank_device,
            "rerank_pool_size": args.rerank_pool_size,
        },
        "summary": summaries,
        "details": details,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_csv, summaries, ks)
    if dump_rerank_details:
        dump_rerank_details.parent.mkdir(parents=True, exist_ok=True)
        dump_rerank_details.write_text(json.dumps({"queries": rerank_debug}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_table(summaries, ks))
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved CSV:  {output_csv}")
    if dump_rerank_details:
        print(f"Saved rerank dump: {dump_rerank_details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
