from __future__ import annotations

from dataclasses import replace

from video_rag_cli import SearchHit


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for reranking. "
                "Install requirements-rerank.txt or use the Docker image."
            ) from exc

        kwargs = {"trust_remote_code": True}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._load_model()
        scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return [float(score) for score in scores]


def rerank_hits(
    query: str,
    hits: list[SearchHit],
    reranker: CrossEncoderReranker,
    *,
    top_k: int,
) -> tuple[list[SearchHit], list[dict]]:
    non_empty: list[tuple[int, SearchHit]] = []
    debug_rows: list[dict] = []

    for rank, hit in enumerate(hits, start=1):
        content = (hit.content or "").strip()
        if not content:
            debug_rows.append(
                {
                    "segment_id": hit.segment_id,
                    "before_rank": rank,
                    "before_score": hit.score,
                    "rerank_score": None,
                    "after_rank": None,
                    "content_preview": "",
                    "empty_content": True,
                }
            )
            continue
        non_empty.append((rank, hit))

    pairs = [(query, hit.content) for _, hit in non_empty]
    scores = reranker.score_pairs(pairs)

    rescored: list[SearchHit] = []
    for (before_rank, hit), score in zip(non_empty, scores):
        rescored.append(replace(hit, score=score, score_type="hybrid_rerank"))
        debug_rows.append(
            {
                "segment_id": hit.segment_id,
                "before_rank": before_rank,
                "before_score": hit.score,
                "rerank_score": score,
                "after_rank": None,
                "content_preview": hit.content[:160],
                "empty_content": False,
            }
        )

    rescored.sort(key=lambda hit: hit.score, reverse=True)
    trimmed = rescored[:top_k]
    after_ranks = {hit.segment_id: rank for rank, hit in enumerate(trimmed, start=1)}

    for row in debug_rows:
        row["after_rank"] = after_ranks.get(row["segment_id"])

    debug_rows.sort(
        key=lambda row: (
            row["after_rank"] is None,
            row["after_rank"] if row["after_rank"] is not None else 10**9,
            row["before_rank"],
        )
    )
    return trimmed, debug_rows
