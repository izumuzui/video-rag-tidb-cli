# video-rag-tidb-cli

動画RAGの記事用に作った、短い動画を Python CLI でインデックス化する実験コード。

目的は「動画をどんな検索単位に分解すると扱いやすいか」の確認。`ffmpeg` で音声・フレーム・クリップを切り出し、Gemini で説明文と embedding を作り、JSONL や TiDB Cloud に保存する構成。

## できること

- 固定長セグメントへの分割
- 単一フレーム / 複数フレーム / 動画クリップの切り出し
- 音声セグメントの切り出しと transcript 生成
- `mock` / `gemini` provider による説明文・embedding 生成
- JSONL インデックスの作成
- ローカル JSONL に対する簡易ベクトル検索
- TiDB Cloud の初期化
- TiDB Cloud へのセグメント投入
- TiDB Cloud 上での keyword / vector / hybrid 検索
- Docker 上での `hybrid_rerank` 評価実行

## 動作環境

- Python 3.11+
- `ffmpeg`
- `ffprobe`

Reranker を使う評価は `torch` や `sentence-transformers` が必要になるため、Mac の素の環境を汚したくない場合は Docker 実行を前提にしています。

## セットアップ

```bash
cd video-rag-tidb-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 環境変数

Gemini や TiDB Cloud を使う場合は、`.env.example` をコピーして `.env` を作成。

```bash
cp .env.example .env
```

主に使う値は次の通り。

- `GEMINI_API_KEY`
- `TIDB_HOST`
- `TIDB_PORT`
- `TIDB_USER`
- `TIDB_PASSWORD`
- `TIDB_DATABASE`
- `TIDB_SSL_CA`

`mock` provider のみを使う場合は API キー不要。

## 最小の動作確認

まずは `mock` provider で、インデックス作成からローカル検索までの動作確認。

```bash
.venv/bin/python video_rag_cli.py index \
  --video /path/to/sample.mp4 \
  --workdir .work/sample \
  --segment-seconds 5 \
  --methods single_frame,multi_frame,video_clip \
  --provider mock
```

```bash
.venv/bin/python video_rag_cli.py search \
  --index .work/sample/segments.jsonl \
  --query "人がドアを開ける" \
  --top-k 5 \
  --method multi_frame \
  --provider mock
```

## Gemini を使う場合

説明文と embedding を実モデルで作るときは `--provider gemini` を使用。

```bash
.venv/bin/python video_rag_cli.py index \
  --video /path/to/sample.mp4 \
  --workdir .work/sample \
  --segment-seconds 5 \
  --methods single_frame,multi_frame,video_clip \
  --provider gemini
```

Gemini で生成した index を検索するときは、クエリ側も `--provider gemini` に統一。

```bash
.venv/bin/python video_rag_cli.py search \
  --index .work/sample/segments.jsonl \
  --query "人がドアを開ける" \
  --top-k 5 \
  --method multi_frame \
  --provider gemini
```

## TiDB Cloud を使う場合

### 1. スキーマを作る

```bash
.venv/bin/python video_rag_cli.py init-tidb
```

### 2. JSONL を投入する

```bash
.venv/bin/python video_rag_cli.py load-tidb \
  --index .work/sample/segments.jsonl
```

### 3. TiDB 上で検索する

```bash
.venv/bin/python video_rag_cli.py search-tidb \
  --query "人がドアを開ける" \
  --mode vector \
  --top-k 5 \
  --method multi_frame \
  --provider gemini
```

`--mode` で使える値は次の通り。

- `keyword`
- `vector`
- `hybrid`
- `hybrid_rerank`

## Reranker 評価を Docker で回す

Cross-Encoder の reranker は CPU でも動きますが、追加依存が重めです。このリポジトリでは Docker で隔離して実行できます。

ビルド:

```bash
docker build -t video-rag-rerank .
```

評価実行例:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD":/app \
  -v "$HOME/.cache/huggingface":/cache/huggingface \
  video-rag-rerank \
  python eval/run_eval.py \
    --dataset eval/qvhighlights_loaded_multi_frame_dataset.json \
    --output-json eval/qvhighlights_loaded_multi_frame_results_rerank.json \
    --output-csv eval/qvhighlights_loaded_multi_frame_results_rerank.csv \
    --dump-rerank-details eval/qvhighlights_loaded_multi_frame_rerank_debug.json \
    --method multi_frame \
    --provider gemini \
    --modes vector,hybrid,hybrid_rerank \
    --rerank-pool-size 20
```

## 実装メモ

- `transcript` は音声がある場合、セグメントごとに切り出して生成
- ベクトル検索は TiDB の `VEC_COSINE_DISTANCE()` を使用
- `hybrid` は RRF で統合

## TiDB Cloud について

`sql/schema.sql` に記事用の DDL を配置。実クラスタで使う前に、TiDB Cloud の利用プラン、VECTOR 次元数、検索機能の仕様に合わせた調整が必要。

このリポジトリでの `keyword` は、MySQL の `MATCH ... AGAINST` による FULLTEXT 検索ではなく、単語分割したクエリを `LIKE` で見る簡易キーワード検索。現在の環境では `MATCH ... AGAINST` を試すと `UnknownType: *ast.MatchAgainst` で失敗。

そのため、この CLI での扱いは次の通り。

- `keyword`: 単語分割した `LIKE` ベースの簡易キーワード検索
- `vector`: TiDB の `VEC_COSINE_DISTANCE()` を使ったベクトル検索
- `hybrid`: 両者の順位を RRF で統合した検索
- `hybrid_rerank`: `hybrid` の上位候補を Cross-Encoder で並べ替える評価用検索

## 注意

このリポジトリは記事用の実験コード。実運用向けにエラーハンドリング、テスト、入出力形式、認証まわりを整えたものではない。
