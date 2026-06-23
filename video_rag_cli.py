#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pymysql


EMBEDDING_DIM = 128
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass
class Segment:
    segment_key: str
    video_path: str
    start_sec: float
    end_sec: float
    modality: str
    indexing_method: str
    content: str
    artifacts: dict
    embedding: list[float]
    embedding_model: str


@dataclass
class TiDBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    ssl_ca: str | None


@dataclass
class SearchHit:
    segment_id: int
    video_id: int
    file_name: str
    start_sec: float
    end_sec: float
    modality: str
    indexing_method: str
    content: str
    score: float
    score_type: str


def run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise SystemExit(f"Command failed: {' '.join(cmd)}\n{message}") from exc
    return proc.stdout


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found in PATH.")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required but was not found in PATH.")


def video_duration(video: Path) -> float:
    output = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    ).strip()
    return float(output)


def make_ranges(duration: float, segment_seconds: float) -> list[tuple[float, float]]:
    ranges = []
    start = 0.0
    while start < duration:
        end = min(start + segment_seconds, duration)
        if end - start > 0.05:
            ranges.append((round(start, 3), round(end, 3)))
        start = end
    return ranges


def sample_timestamps(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count must be greater than 0")
    duration = end - start
    if duration <= 0:
        return [round(start, 3)] * count
    if count == 1:
        return [round(start + duration / 2, 3)]

    step = duration / count
    return [round(start + step * (i + 0.5), 3) for i in range(count)]


def extract_frame(video: Path, timestamp: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ]
    )


def extract_clip(video: Path, start: float, duration: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(output),
        ]
    )


def extract_audio(video: Path, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output),
            ]
        )
    except SystemExit:
        return False
    return output.exists() and output.stat().st_size > 0


def extract_audio_segment(video: Path, start: float, duration: float, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video),
                "-t",
                f"{duration:.3f}",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output),
            ]
        )
    except SystemExit:
        return False
    return output.exists() and output.stat().st_size > 0


def has_audio_stream(video: Path) -> bool:
    output = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video),
        ]
    ).strip()
    return bool(output)


def file_digest(path: Path, bytes_to_read: int = 256_000) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        digest.update(f.read(bytes_to_read))
    return digest.hexdigest()[:16]


def mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    values = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            idx = (byte + i * 17) % dim
            values[idx] += 1.0 if byte % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimensions do not match")
    return sum(x * y for x, y in zip(a, b))


def mock_describe_single_frame(frame: Path, start: float, end: float) -> str:
    digest = file_digest(frame)
    return (
        f"Mock visual description for {start:.1f}-{end:.1f}s. "
        f"single frame artifact digest={digest}. Replace this with a Vision model caption."
    )


def mock_describe_multi_frame(frames: list[Path], start: float, end: float) -> str:
    digests = ", ".join(file_digest(frame) for frame in frames)
    return (
        f"Mock visual description for {start:.1f}-{end:.1f}s. "
        f"multi frame artifacts digests=[{digests}]. Replace this with a multi-image caption."
    )


def mock_describe_video_clip(clip: Path, start: float, end: float) -> str:
    digest = file_digest(clip)
    return (
        f"Mock visual description for {start:.1f}-{end:.1f}s. "
        f"video clip artifact digest={digest}. Replace this with a video-capable model caption."
    )


class CaptionEmbeddingProvider:
    embedding_model: str

    def transcribe_audio(self, audio: Path, start: float, end: float) -> str:
        raise NotImplementedError

    def describe_single_frame(self, frame: Path, start: float, end: float) -> str:
        raise NotImplementedError

    def describe_multi_frame(self, frames: list[Path], start: float, end: float) -> str:
        raise NotImplementedError

    def describe_video_clip(self, clip: Path, start: float, end: float) -> str:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError


class MockProvider(CaptionEmbeddingProvider):
    embedding_model = f"mock-hash-{EMBEDDING_DIM}"

    def transcribe_audio(self, audio: Path, start: float, end: float) -> str:
        return (
            f"Mock transcript for {start:.1f}-{end:.1f}s. "
            "Replace this with a speech-to-text or multimodal transcription model."
        )

    def describe_single_frame(self, frame: Path, start: float, end: float) -> str:
        return mock_describe_single_frame(frame, start, end)

    def describe_multi_frame(self, frames: list[Path], start: float, end: float) -> str:
        return mock_describe_multi_frame(frames, start, end)

    def describe_video_clip(self, clip: Path, start: float, end: float) -> str:
        return mock_describe_video_clip(clip, start, end)

    def embed_text(self, text: str) -> list[float]:
        return mock_embedding(text)


class GeminiProvider(CaptionEmbeddingProvider):
    def __init__(self, model: str, embedding_model: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise SystemExit(
                "google-genai is required for --provider gemini. "
                "Run `pip install -r requirements.txt` first."
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required for --provider gemini.")

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model
        self.embedding_model = embedding_model

    def _part_from_file(self, path: Path):
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)

    def _generate(self, parts: list) -> str:
        response = self.client.models.generate_content(model=self.model, contents=parts)
        text = (response.text or "").strip()
        if not text:
            raise SystemExit("Gemini returned an empty caption.")
        return text

    def transcribe_audio(self, audio: Path, start: float, end: float) -> str:
        prompt = (
            f"これは動画の {start:.1f} 秒から {end:.1f} 秒までの音声です。"
            "日本語でできるだけ忠実に文字起こししてください。"
            "要約ではなく書き起こしを返し、話者名の推測はしないでください。"
            "音声がほとんど無い場合は、その旨を短く返してください。"
        )
        return self._generate([self._part_from_file(audio), prompt])

    def describe_single_frame(self, frame: Path, start: float, end: float) -> str:
        prompt = (
            f"これは動画の {start:.1f} 秒から {end:.1f} 秒の区間を代表する1枚のフレームです。"
            "自然言語検索のインデックスに使うため、見えている物体、人物、場所、画面内テキストを"
            "日本語で1〜3文に要約してください。推測しすぎず、観測できる内容を優先してください。"
        )
        return self._generate([self._part_from_file(frame), prompt])

    def describe_multi_frame(self, frames: list[Path], start: float, end: float) -> str:
        prompt = (
            f"これらの画像は同じ動画の {start:.1f} 秒から {end:.1f} 秒の区間から順番に抽出したフレームです。"
            "自然言語検索のインデックスに使うため、この区間で何が起きているかを日本語で1〜3文に要約してください。"
            "重要な動作や変化があれば明示してください。"
        )
        parts = [self._part_from_file(frame) for frame in frames]
        parts.append(prompt)
        return self._generate(parts)

    def describe_video_clip(self, clip: Path, start: float, end: float) -> str:
        prompt = (
            f"これは動画の {start:.1f} 秒から {end:.1f} 秒の短いクリップです。"
            "自然言語検索のインデックスに使うため、この区間の出来事、動作、映っている物を"
            "日本語で1〜3文に要約してください。検索に効く具体的な名詞や動詞を優先してください。"
        )
        return self._generate([self._part_from_file(clip), prompt])

    def embed_text(self, text: str) -> list[float]:
        result = self.client.models.embed_content(model=self.embedding_model, contents=text)
        if not result.embeddings:
            raise SystemExit("Gemini returned no embeddings.")
        return [float(value) for value in result.embeddings[0].values]


def make_provider(name: str, gemini_model: str | None, embedding_model: str | None) -> CaptionEmbeddingProvider:
    if name == "mock":
        return MockProvider()
    if name == "gemini":
        return GeminiProvider(
            model=gemini_model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
            embedding_model=embedding_model
            or os.environ.get("GEMINI_EMBEDDING_MODEL")
            or DEFAULT_GEMINI_EMBEDDING_MODEL,
        )
    raise SystemExit(f"Unknown provider: {name}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_tidb_config() -> TiDBConfig:
    host = os.environ.get("TIDB_HOST")
    user = os.environ.get("TIDB_USER")
    password = os.environ.get("TIDB_PASSWORD")
    database = os.environ.get("TIDB_DATABASE")
    port_raw = os.environ.get("TIDB_PORT", "4000")
    ssl_ca = os.environ.get("TIDB_SSL_CA") or None

    missing = [
        name
        for name, value in [
            ("TIDB_HOST", host),
            ("TIDB_USER", user),
            ("TIDB_PASSWORD", password),
            ("TIDB_DATABASE", database),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing TiDB settings in environment: {', '.join(missing)}")

    return TiDBConfig(
        host=host,
        port=int(port_raw),
        user=user,
        password=password,
        database=database,
        ssl_ca=ssl_ca,
    )


def tidb_connect(config: TiDBConfig, *, use_database: bool = True):
    connect_kwargs = dict(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        autocommit=False,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    connect_kwargs["ssl"] = {"ca": config.ssl_ca} if config.ssl_ca else {}
    if use_database:
        connect_kwargs["database"] = config.database
    return pymysql.connect(
        **connect_kwargs,
    )


def vector_sql_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.12f}".rstrip("0").rstrip(".") for value in values) + "]"


def quote_identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


def row_to_hit(row: dict, *, score_type: str) -> SearchHit:
    return SearchHit(
        segment_id=int(row["segment_id"]),
        video_id=int(row["video_id"]),
        file_name=row["file_name"],
        start_sec=float(row["start_sec"]),
        end_sec=float(row["end_sec"]),
        modality=row["modality"],
        indexing_method=row["indexing_method"],
        content=row["content"],
        score=float(row["score"]),
        score_type=score_type,
    )


def build_method_filter(method: str | None) -> tuple[str, list]:
    if not method:
        return "", []
    return " AND vs.indexing_method = %s", [method]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "up",
    "with",
}


def tokenize_keyword_query(query: str) -> list[str]:
    tokens = re.findall(r"\w+", query.lower())
    deduped = []
    seen = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) == 1 and token.isascii():
            continue
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    if deduped:
        return deduped

    normalized = query.strip().lower()
    if normalized:
        return [normalized]
    return []


def keyword_search_tidb(connection, query: str, top_k: int, method: str | None) -> list[SearchHit]:
    tokens = tokenize_keyword_query(query)
    if not tokens:
        return []

    method_sql, method_params = build_method_filter(method)
    score_terms = " + ".join(["CASE WHEN LOWER(vs.content) LIKE %s THEN 1 ELSE 0 END" for _ in tokens])
    where_terms = " OR ".join(["LOWER(vs.content) LIKE %s" for _ in tokens])
    sql = f"""
        SELECT
          vs.id AS segment_id,
          v.id AS video_id,
          v.file_name,
          vs.start_sec,
          vs.end_sec,
          vs.modality,
          vs.indexing_method,
          vs.content,
          ({score_terms}) AS score
        FROM video_segments vs
        JOIN videos v ON v.id = vs.video_id
        WHERE ({where_terms})
        {method_sql}
        ORDER BY score DESC, vs.id ASC
        LIMIT %s
    """
    like_params = [f"%{token}%" for token in tokens]
    params = like_params + like_params + method_params + [top_k]
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return [row_to_hit(row, score_type="keyword") for row in cursor.fetchall()]


def vector_search_tidb(
    connection,
    query_embedding: list[float],
    top_k: int,
    method: str | None,
) -> list[SearchHit]:
    method_sql, method_params = build_method_filter(method)
    sql = f"""
        SELECT
          vs.id AS segment_id,
          v.id AS video_id,
          v.file_name,
          vs.start_sec,
          vs.end_sec,
          vs.modality,
          vs.indexing_method,
          vs.content,
          VEC_COSINE_DISTANCE(se.embedding, %s) AS distance
        FROM segment_embeddings se
        JOIN video_segments vs ON vs.id = se.segment_id
        JOIN videos v ON v.id = vs.video_id
        WHERE se.embedding IS NOT NULL
        {method_sql}
        ORDER BY distance ASC, vs.id ASC
        LIMIT %s
    """
    params = [vector_sql_literal(query_embedding)] + method_params + [top_k]
    hits = []
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        for row in cursor.fetchall():
            row["score"] = 1.0 - float(row["distance"])
            hits.append(row_to_hit(row, score_type="vector"))
    return hits


def reciprocal_rank_fusion(
    keyword_hits: list[SearchHit],
    vector_hits: list[SearchHit],
    top_k: int,
) -> list[SearchHit]:
    by_id: dict[int, SearchHit] = {}
    scores: dict[int, float] = {}
    k = 60.0

    for rank, hit in enumerate(keyword_hits, start=1):
        by_id[hit.segment_id] = hit
        scores[hit.segment_id] = scores.get(hit.segment_id, 0.0) + 1.0 / (k + rank)

    for rank, hit in enumerate(vector_hits, start=1):
        by_id[hit.segment_id] = hit
        scores[hit.segment_id] = scores.get(hit.segment_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results = []
    for segment_id, score in ranked:
        base = by_id[segment_id]
        results.append(
            SearchHit(
                segment_id=base.segment_id,
                video_id=base.video_id,
                file_name=base.file_name,
                start_sec=base.start_sec,
                end_sec=base.end_sec,
                modality=base.modality,
                indexing_method=base.indexing_method,
                content=base.content,
                score=score,
                score_type="hybrid",
            )
        )
    return results


def segment_key(video: Path, method: str, start: float, end: float) -> str:
    raw = f"{video.resolve()}:{method}:{start:.3f}:{end:.3f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def build_segments(
    video: Path,
    workdir: Path,
    segment_seconds: float,
    methods: set[str],
    provider: CaptionEmbeddingProvider,
    multi_frame_count: int,
) -> list[Segment]:
    ensure_ffmpeg()
    duration = video_duration(video)
    ranges = make_ranges(duration, segment_seconds)
    artifacts_dir = workdir / "artifacts"
    segments: list[Segment] = []
    has_audio = has_audio_stream(video)

    for start, end in ranges:
        if "transcript" in methods and has_audio:
            audio_path = artifacts_dir / "audio" / f"{video.stem}_{start:.3f}_{end:.3f}.wav"
            if extract_audio_segment(video, start, end - start, audio_path):
                content = provider.transcribe_audio(audio_path, start, end)
                segments.append(
                    make_segment(
                        video=video,
                        start=start,
                        end=end,
                        modality="audio",
                        indexing_method="transcript",
                        content=content,
                        artifacts={"audio_path": str(audio_path)},
                        provider=provider,
                    )
                )

        if "single_frame" in methods:
            timestamp = start + (end - start) / 2
            frame = artifacts_dir / "single_frame" / f"{video.stem}_{start:.3f}_{end:.3f}.png"
            extract_frame(video, timestamp, frame)
            content = provider.describe_single_frame(frame, start, end)
            segments.append(
                make_segment(
                    video=video,
                    start=start,
                    end=end,
                    modality="visual",
                    indexing_method="single_frame",
                    content=content,
                    artifacts={"frame_path": str(frame), "frame_timestamp": timestamp},
                    provider=provider,
                )
            )

        if "multi_frame" in methods:
            timestamps = sample_timestamps(start, end, multi_frame_count)
            frames = []
            for i, timestamp in enumerate(timestamps):
                frame = artifacts_dir / "multi_frame" / f"{video.stem}_{start:.3f}_{end:.3f}_{i}.png"
                extract_frame(video, timestamp, frame)
                frames.append(frame)
            content = provider.describe_multi_frame(frames, start, end)
            segments.append(
                make_segment(
                    video=video,
                    start=start,
                    end=end,
                    modality="visual",
                    indexing_method="multi_frame",
                    content=content,
                    artifacts={
                        "frame_paths": [str(frame) for frame in frames],
                        "frame_timestamps": timestamps,
                    },
                    provider=provider,
                )
            )

        if "video_clip" in methods:
            clip = artifacts_dir / "video_clip" / f"{video.stem}_{start:.3f}_{end:.3f}.mp4"
            extract_clip(video, start, end - start, clip)
            content = provider.describe_video_clip(clip, start, end)
            segments.append(
                make_segment(
                    video=video,
                    start=start,
                    end=end,
                    modality="visual",
                    indexing_method="video_clip",
                    content=content,
                    artifacts={"clip_path": str(clip)},
                    provider=provider,
                )
            )

    return segments


def make_segment(
    video: Path,
    start: float,
    end: float,
    modality: str,
    indexing_method: str,
    content: str,
    artifacts: dict,
    provider: CaptionEmbeddingProvider,
) -> Segment:
    return Segment(
        segment_key=segment_key(video, indexing_method, start, end),
        video_path=str(video),
        start_sec=round(start, 3),
        end_sec=round(end, 3),
        modality=modality,
        indexing_method=indexing_method,
        content=content,
        artifacts=artifacts,
        embedding=provider.embed_text(content),
        embedding_model=provider.embedding_model,
    )


def cmd_index(args: argparse.Namespace) -> int:
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    workdir = Path(args.workdir).expanduser().resolve()
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    allowed = {"transcript", "single_frame", "multi_frame", "video_clip"}
    unknown = methods - allowed
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(sorted(unknown))}")

    provider = make_provider(args.provider, args.gemini_model, args.embedding_model)
    segments = build_segments(
        video,
        workdir,
        args.segment_seconds,
        methods,
        provider,
        args.multi_frame_count,
    )
    index_path = workdir / "segments.jsonl"
    write_jsonl(index_path, (asdict(segment) for segment in segments))

    summary = {
        "video": str(video),
        "workdir": str(workdir),
        "segments": len(segments),
        "index_path": str(index_path),
        "methods": sorted(methods),
        "segment_seconds": args.segment_seconds,
        "provider": args.provider,
        "embedding_model": provider.embedding_model,
        "multi_frame_count": args.multi_frame_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index_path = Path(args.index).expanduser().resolve()
    rows = read_jsonl(index_path)
    provider = make_provider(args.provider, args.gemini_model, args.embedding_model)
    query_embedding = provider.embed_text(args.query)
    scored = []
    for row in rows:
        if args.method and row["indexing_method"] != args.method:
            continue
        score = cosine(query_embedding, row["embedding"])
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, row in scored[: args.top_k]:
        results.append(
            {
                "score": round(score, 6),
                "video_path": row["video_path"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "modality": row["modality"],
                "indexing_method": row["indexing_method"],
                "content": row["content"],
                "artifacts": row["artifacts"],
            }
        )
    print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
    return 0


def cmd_search_tidb(args: argparse.Namespace) -> int:
    config = parse_tidb_config()
    connection = tidb_connect(config, use_database=True)
    provider = None
    query_embedding = None

    try:
        if args.mode in {"vector", "hybrid"}:
            provider = make_provider(args.provider, args.gemini_model, args.embedding_model)
            query_embedding = provider.embed_text(args.query)

        if args.mode == "keyword":
            hits = keyword_search_tidb(connection, args.query, args.top_k, args.method)
        elif args.mode == "vector":
            assert query_embedding is not None
            hits = vector_search_tidb(connection, query_embedding, args.top_k, args.method)
        else:
            keyword_hits = keyword_search_tidb(connection, args.query, args.top_k * 2, args.method)
            assert query_embedding is not None
            vector_hits = vector_search_tidb(connection, query_embedding, args.top_k * 2, args.method)
            hits = reciprocal_rank_fusion(keyword_hits, vector_hits, args.top_k)
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "query": args.query,
                "mode": args.mode,
                "results": [asdict(hit) for hit in hits],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_load_tidb(args: argparse.Namespace) -> int:
    index_path = Path(args.index).expanduser().resolve()
    rows = read_jsonl(index_path)
    if not rows:
        raise SystemExit(f"No rows found in index: {index_path}")

    config = parse_tidb_config()
    connection = tidb_connect(config)
    inserted_segments = 0

    try:
        with connection.cursor() as cursor:
            first = rows[0]
            video_path = Path(first["video_path"])
            duration_sec = max(float(row["end_sec"]) for row in rows)
            metadata = {
                "source_index": str(index_path),
                "providers": sorted({row["embedding_model"] for row in rows}),
            }

            cursor.execute(
                """
                INSERT INTO videos (file_name, file_path, duration_sec, metadata_json)
                VALUES (%s, %s, %s, %s)
                """,
                (video_path.name, str(video_path), duration_sec, json.dumps(metadata, ensure_ascii=False)),
            )
            video_id = cursor.lastrowid

            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO video_segments
                      (video_id, start_sec, end_sec, modality, indexing_method, content, artifact_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        video_id,
                        row["start_sec"],
                        row["end_sec"],
                        row["modality"],
                        row["indexing_method"],
                        row["content"],
                        json.dumps(row["artifacts"], ensure_ascii=False),
                    ),
                )
                segment_id = cursor.lastrowid
                cursor.execute(
                    """
                    INSERT INTO segment_embeddings
                      (segment_id, embedding, embedding_model)
                    VALUES (%s, %s, %s)
                    """,
                    (segment_id, vector_sql_literal(row["embedding"]), row["embedding_model"]),
                )
                inserted_segments += 1

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "index_path": str(index_path),
                "inserted_segments": inserted_segments,
                "database": config.database,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_init_tidb(args: argparse.Namespace) -> int:
    config = parse_tidb_config()
    schema_path = Path(args.schema).expanduser().resolve()
    if not schema_path.exists():
        raise SystemExit(f"Schema file not found: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    bootstrap = tidb_connect(config, use_database=False)
    try:
        with bootstrap.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(config.database)}")
        bootstrap.commit()
    finally:
        bootstrap.close()

    connection = tidb_connect(config, use_database=True)
    try:
        with connection.cursor() as cursor:
            for statement in schema_sql.split(";"):
                sql = statement.strip()
                if sql:
                    cursor.execute(sql)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "database": config.database,
                "schema": str(schema_path),
                "status": "initialized",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video RAG indexing experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Extract video artifacts and build JSONL segments")
    index.add_argument("--video", required=True, help="Path to a video file")
    index.add_argument("--workdir", required=True, help="Directory for artifacts and JSONL output")
    index.add_argument("--segment-seconds", type=float, default=5.0)
    index.add_argument(
        "--methods",
        default="single_frame,multi_frame",
        help="Comma-separated methods: transcript,single_frame,multi_frame,video_clip",
    )
    index.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    index.add_argument("--gemini-model", default=None)
    index.add_argument("--embedding-model", default=None)
    index.add_argument("--multi-frame-count", type=int, default=3)
    index.set_defaults(func=cmd_index)

    search = subparsers.add_parser("search", help="Search a local JSONL index with mock embeddings")
    search.add_argument("--index", required=True, help="Path to segments.jsonl")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--method", choices=["transcript", "single_frame", "multi_frame", "video_clip"])
    search.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    search.add_argument("--gemini-model", default=None)
    search.add_argument("--embedding-model", default=None)
    search.set_defaults(func=cmd_search)

    load_tidb = subparsers.add_parser("load-tidb", help="Load a JSONL index into TiDB Cloud")
    load_tidb.add_argument("--index", required=True, help="Path to segments.jsonl")
    load_tidb.set_defaults(func=cmd_load_tidb)

    init_tidb = subparsers.add_parser("init-tidb", help="Create database and apply schema in TiDB Cloud")
    init_tidb.add_argument("--schema", default="sql/schema.sql", help="Path to schema.sql")
    init_tidb.set_defaults(func=cmd_init_tidb)

    search_tidb = subparsers.add_parser("search-tidb", help="Search segments stored in TiDB Cloud")
    search_tidb.add_argument("--query", required=True)
    search_tidb.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    search_tidb.add_argument("--top-k", type=int, default=5)
    search_tidb.add_argument("--method", choices=["transcript", "single_frame", "multi_frame", "video_clip"])
    search_tidb.add_argument("--provider", choices=["mock", "gemini"], default="gemini")
    search_tidb.add_argument("--gemini-model", default=None)
    search_tidb.add_argument("--embedding-model", default=None)
    search_tidb.set_defaults(func=cmd_search_tidb)

    return parser


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
