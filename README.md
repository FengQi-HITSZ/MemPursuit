# MemPursuit

Code for **Recall What Is Missing: Composing Evidence from Long-Term Agent Memory**.

MemPursuit selects complementary memories through simplex-constrained, fully corrective sparse OMP. It reuses query and memory embeddings, selects memories in pursuit order, and produces nested contexts from a single candidate pool.

The repository provides a standalone NumPy/Numba implementation, a Mem0-based memory pipeline, and end-to-end evaluation on **LoCoMo, LongMemEval-S, and BEAM-500K**.

[中文指南](docs/README.zh-CN.md) · [Evaluation protocol](docs/experiments.md) · [Licenses](NOTICE)

## Installation

Use Python 3.10+, Git and Docker with the Compose v2 plugin. The default setup runs locally on CPU and calls external APIs for memory extraction, embeddings, answers and judging. It requires internet access and API credentials with sufficient quota.

Use a fresh Python environment. This repository provides the `mem0` Python namespace; install it in an environment without `mem0ai` to avoid package conflicts.

```bash
git clone https://github.com/FengQi-HITSZ/MemPursuit.git
cd MemPursuit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pipeline,fast]'
python -m spacy download en_core_web_sm
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Memory extraction defaults to `gpt-4o-mini`; embeddings use `text-embedding-3-small` with 1,536 dimensions. Set `OPENAI_BASE_URL` to use another OpenAI-compatible endpoint. Answer and judge models, endpoints and credentials can be configured separately with the `ANSWER_*` and `JUDGE_*` variables in `.env.example`.

## Datasets

Run benchmark commands from the repository root. Download the datasets with:

```bash
python -m scripts.download_datasets locomo longmemeval beam
```

| Benchmark | Split | Evaluation size |
|---|---|---|
| LoCoMo | LoCoMo-10, categories 1–4 | 10 conversations, 1,540 questions |
| LongMemEval-S | Cleaned S | 500 questions |
| BEAM | 500K | 35 conversations, 700 questions |

The downloader saves datasets under `datasets/`. Dataset and model files retain their upstream licenses.

The paper's main comparison uses **K=50**. The commands below evaluate K=10/20/50/200; use the K=50 outputs when comparing with the [reported results](docs/experiments.md#datasets-and-results). Rebuilding memories and rerunning model APIs can produce different scores because extraction and generation are not deterministic and hosted model versions can change.

## LoCoMo

Start the memory service and check its health:

```bash
docker compose up -d --build
curl --noproxy localhost,127.0.0.1 http://localhost:8889/health
```

The first build downloads the spaCy and BM25 models. Wait for the service to become healthy, then run:

```bash
python -m benchmarks.locomo.run --project-name locomo-mempursuit --resume
```

Defaults: GPT-4o-mini for answering and judging, a 200-memory candidate pool, context sizes 10/20/50/200, search threshold 0.1, two conversation workers and 900-second HTTP timeouts. LoCoMo processes one message per memory-extraction call.

For a smaller run:

```bash
python -m benchmarks.locomo.run --project-name locomo-smoke \
  --conversations 0 --max-questions 1 --max-workers 1 --resume
```

This ingests conversation 0 and evaluates one question. Per-question outputs are stored in `results/locomo/predicted_<project>/`; aggregate results are written to `results/locomo/locomo_results_*.json`.

## LongMemEval-S and BEAM

Use the four-service deployment to parallelize memory construction and retrieval:

```bash
docker compose -f compose.native.yaml up -d --build
```

Wait for `/health` on ports 8891–8894, then run:

```bash
python -m benchmarks.longmemeval.run --project-name longmemeval-mempursuit \
  --mem0-hosts http://localhost:8891,http://localhost:8892,http://localhost:8893,http://localhost:8894 \
  --resume

python -m benchmarks.beam.run --project-name beam-mempursuit \
  --mem0-hosts http://localhost:8891,http://localhost:8892,http://localhost:8893,http://localhost:8894 \
  --resume
```

Run the benchmarks sequentially when sharing these services. LongMemEval defaults to 12 ingestion workers, 12 retrieval workers and 16 answer/judge workers; BEAM defaults to 4/4/16. Adjust these with `--ingest-workers`, `--retrieval-workers` and `--max-workers` to match your hardware and API limits.

Both benchmarks process consecutive messages in pairs within each session, with an odd final message processed alone. Each question retrieves one top-200 sequence and evaluates its prefixes. See the [evaluation protocol](docs/experiments.md) for ingestion, retrieval and scoring details.

A single service is supported through `--mem0-host http://localhost:8889`. Use `--mode smoke` to select a smaller set of memory scopes, or `--max-questions 1` to evaluate one question. Each selected scope is still fully ingested.

Outputs are stored in `results/<benchmark>/<project>/`, including a run manifest, ingestion checkpoints, saved retrieval, prompt snapshots, per-cutoff records and `metrics.json`.

## Resume and reuse retrieval

Add `--resume` to continue an interrupted run. Keep the corresponding Qdrant data, SQLite history and output directory together when resuming memory construction.

LoCoMo supports separate retrieval and evaluation:

```bash
python -m benchmarks.locomo.run --project-name locomo-frozen --predict-only --resume
python -m benchmarks.locomo.run --project-name locomo-frozen --evaluate-only
```

LongMemEval and BEAM support evaluating saved retrieval under a new project name:

```bash
python -m benchmarks.longmemeval.run --project-name longmemeval-qwen \
  --reuse-retrieval-dir results/longmemeval/longmemeval-mempursuit \
  --answerer-model Qwen3-30B-A3B-Instruct-2507 --resume
```

Configure `ANSWER_BASE_URL` and `ANSWER_API_KEY` for the chosen model and use the model identifier accepted by your provider. Keep the judge configuration fixed when comparing answer models. For LoCoMo, `--evaluate-only --rejudge` regenerates answers from saved retrieval; copy the prediction directory first to retain both sets of outputs.

## Adaptive selection and acceleration

| Configuration | Selection mode | Backend |
|---|---|---|
| `configs/mempursuit.yaml` | Fixed budget | NumPy |
| `configs/adaptive.yaml` | Residual-based early stopping | NumPy |
| `configs/numba.yaml` | Fixed budget | Numba |

To use adaptive selection on LoCoMo, set `MEM0_CONFIG_FILE=./configs/adaptive.yaml` in `.env`, recreate the service and run:

```bash
docker compose up -d --force-recreate memory
python -m benchmarks.locomo.run --project-name locomo-adaptive --adaptive-context --resume
```

To measure selection latency:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m scripts.benchmark_latency --backend numba --budget 20
```

The default input is synthetic. Supply `--input embeddings.npz` with arrays `query[d]` and `candidates[P,d]` to benchmark your own embeddings. Timing covers normalization, Gram-matrix construction, selection and refitting, excluding warmup, API calls and embedding inference.

## Repository layout

| Path | Purpose |
|---|---|
| `mempursuit/` | Standalone selection algorithm |
| `mem0/reranker/` | Sparse OMP integration and Numba backend |
| `mem0/memory/`, `mem0/vector_stores/` | Memory extraction, persistence and hybrid retrieval |
| `server/` | Self-hosted HTTP API |
| `benchmarks/locomo/` | LoCoMo evaluation |
| `benchmarks/pipeline/` | LongMemEval/BEAM ingestion, retrieval and evaluation |
| `configs/` | Selection and memory-service configurations |

## License

The project is licensed under [Apache 2.0](LICENSE). Third-party components and their licenses are listed in [NOTICE](NOTICE).
