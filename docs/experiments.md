# Evaluation protocol

The evaluation pipeline constructs memories, retrieves candidates, selects evidence with MemPursuit, generates answers and scores them with a judge model.

## Datasets and results

| Benchmark | Split | Questions | GPT-4o-mini result at K=50 |
|---|---|---:|---:|
| LoCoMo | LoCoMo-10, categories 1–4 | 1,540 | 87.08% accuracy |
| LongMemEval-S | Cleaned S | 500 | 71.20% accuracy |
| BEAM | 500K, 35 conversations | 700 | 43.82 / 100 |

These results are reported in **Recall What Is Missing: Composing Evidence from Long-Term Agent Memory**. BEAM aggregates fractional scores; its result is not a thresholded question-pass rate.

## Memory construction

The memory pipeline retrieves ten existing memories and reads the ten most recent stored messages for extraction context. GPT-4o-mini at temperature 0.1 extracts additive facts. Exact text hashes suppress duplicates within the extraction batch and among retrieved memories. New facts are appended, embedded, stored in Qdrant and linked to a separate entity index.

| Setting | LoCoMo | LongMemEval-S | BEAM-500K |
|---|---|---|---|
| Messages per extraction call | 1 | 2 within one session | 2 within one session |
| Memory scope | One conversation | One question's haystack | One conversation |
| Session order | Chronological | Chronological | Chronological |
| Turn order | Dataset order | Dataset order | Dataset order |
| Timestamp source | Session time, converted to epoch | Session timestamp | Session time anchor |

A two-message batch can produce multiple extracted memories. An odd trailing message is processed alone. Message pairs do not cross session boundaries. Evaluation questions and ground-truth annotations are excluded from memory construction.

## Retrieval and selection

1. Encode the query with `text-embedding-3-small` (1,536 dimensions).
2. Retrieve dense and lemma-based BM25 candidates, over-fetching `max(4P, 60)` from each channel.
3. Normalize BM25 with a query-length-dependent sigmoid, combine dense similarity, BM25 and entity scores additively, and retain at most P=200 candidates. Entity scores adjust the candidate union without adding new entries.
4. Normalize query and candidate embeddings. Run simplex-constrained fully corrective sparse OMP with ridge `1e-10` and negative tolerance `1e-10`.
5. For fixed-budget evaluation, request 200 iterations with early stopping disabled. If fewer candidates are available, use the available pool.
6. Evaluate K=10/20/50/200 as nested prefixes of the resulting selection sequence.

LoCoMo uses search threshold **0.1**. LongMemEval-S and BEAM pass **None**, disabling this threshold.

Selection order follows pursuit entry order, including entries whose refitted coefficients become zero. The diagnostic `score` field contains the hybrid retrieval score; `rerank_score` reflects selection order. Sorting by coefficients or hybrid scores changes the selected prefixes.

`configs/mempursuit.yaml` uses NumPy with maximum reranker concurrency 2. `configs/numba.yaml` enables the Numba implementation.

## Answering and judging

### LoCoMo

The runner evaluates categories 1–4 with the `locomo_mem0_accuracy_v1` judge. Selected memories are presented chronologically. Each cutoff has separate saved answers and judgments, and the aggregate metric is answer accuracy.

### LongMemEval-S

The runner uses the prompts in `benchmarks/longmemeval/prompts.py`, identified as `current` in result records. Selected memories are presented chronologically. Answer generation uses temperature 0 and a maximum of 4,096 output tokens; the judge uses temperature 0 and a maximum of 1,024 tokens. The evaluator strips reasoning tags and parses the final judgment to compute accuracy.

### BEAM

The evaluator uses the official-v1 prompts and metrics implemented in `benchmarks/pipeline/beam_official.py`, based on [BEAM commit b2da22e](https://github.com/mohammadtavakoli78/BEAM/tree/b2da22eac88bb0874c64665f13457eb99835774a).

Rubric scores are averaged. Event-ordering questions use normalized Kendall tau as their primary score. The judge client retries malformed responses and recovers unambiguous score fields when possible.

The reader concatenates memory texts in retrieval order without an added separator, with a context gate of `characters // 3.5 < 29000`. Answer generation uses temperature 0 and a maximum of 4,096 output tokens. Scores in `metrics.json` are on a 0–1 scale; multiply by 100 for the reported BEAM score.

## Execution defaults

| Setting | LoCoMo | LongMemEval-S | BEAM |
|---|---:|---:|---:|
| Conversation / ingestion workers | 2 | 12 | 4 |
| Retrieval workers | Within conversation workers | 12 | 4 |
| Answer/judge workers | Within conversation workers | 16 | 16 |
| Memory request timeout | 900 s | 900 s | 900 s |
| Answer/judge request timeout | 900 s | 300 s | 300 s |

`compose.yaml` runs one memory service with one BLAS thread. `compose.native.yaml` runs four memory services with separate collections/history stores and four BLAS threads per service. With multiple endpoints, scopes are assigned round robin in dataset order and remain on the same endpoint throughout the run.

Both deployments use Qdrant `v1.15.4`. Adjust concurrency to fit your CPU capacity and API rate limits.

## Checkpoints and reproducibility

Keep the Qdrant data, SQLite history and output directory together to resume memory construction. Ingestion checkpoints identify completed work; the memory contents remain in the database.

LongMemEval-S and BEAM checkpoint successful message batches, completed scopes, top-200 retrieval and per-cutoff evaluations. Their run manifests record dataset hashes, model settings, prompt hashes and endpoint routing. Use a new project name when changing these settings.

Use saved retrieval to evaluate another answer model on the same evidence. Remote model versions and sampling can affect memory extraction, generated answers and judgments, so record model identifiers and configuration alongside results.

For existing Qdrant collections, `scripts/backfill_bm25.py` can audit BM25 sparse-vector coverage. It runs in dry-run mode unless `--write` is supplied.

## Adaptive selection

Use `configs/adaptive.yaml` with LoCoMo `--adaptive-context` to stop according to residual improvement:

| Parameter | Value |
|---|---:|
| Relative squared-residual improvement tolerance | `1e-5` |
| Numerical floor | `1e-12` |
| Patience | `2` |
| Absolute residual tolerance | `None` |
| Frank–Wolfe-gap tolerance | `0` (disabled) |

The solver rolls back the stalled window and returns the resulting context. The default configuration uses fixed-budget selection.
