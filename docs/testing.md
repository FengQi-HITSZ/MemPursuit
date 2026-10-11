# Testing

Install the development dependencies from the repository root:

```bash
python -m pip install -e '.[pipeline,fast,test]'
```

Run the test suite and static checks:

```bash
python -m pytest -q
python -m ruff check .
```

## Test coverage

- Simplex-constrained sparse OMP, fixed-budget prefixes and adaptive stopping.
- Numerical agreement between the standalone implementation and framework integration, with NumPy and Numba backends.
- Candidate-pool validation, query compression, memory extraction, deduplication, timestamps and SQLite history.
- Dataset parsing, per-benchmark message grouping, context formatting and scoring.
- API handlers, pipeline stages, checkpoint resume and configuration validation.

The unit and pipeline tests use synthetic inputs and test doubles for external services. They run without API credentials or downloaded benchmark datasets.

## Service smoke test

Follow the [installation and dataset setup](../README.md#installation), start the memory service, and run a small LoCoMo evaluation:

```bash
docker compose up -d --build
curl --noproxy localhost,127.0.0.1 http://localhost:8889/health
python -m benchmarks.locomo.run --project-name locomo-smoke \
  --conversations 0 --max-questions 1 --max-workers 1 --resume
```

This exercises the service, Qdrant, embedding model, extraction model, answerer and judge. It ingests the complete first conversation before evaluating one question and requires API credentials.

LongMemEval-S and BEAM support `--mode smoke` to select a smaller set of scopes. See the [benchmark commands](../README.md#longmemeval-s-and-beam) for service setup and endpoint options.
