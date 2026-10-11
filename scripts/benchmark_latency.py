"""Measure selection latency on supplied frozen embeddings or labeled synthetic input."""

from __future__ import annotations
import argparse
import json
import time
import numpy as np
from mempursuit import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="NPZ with query [d] and candidates [P,d]")
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numba")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.input:
        with np.load(args.input, allow_pickle=False) as data:
            query, candidates = data["query"], data["candidates"]
    else:
        rng = np.random.default_rng(42)
        query, candidates = rng.normal(size=1536), rng.normal(size=(200, 1536))
    for _ in range(3):
        select(query, candidates, budget=args.budget, backend=args.backend)
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        select(query, candidates, budget=args.budget, backend=args.backend)
        times.append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "input": args.input or "synthetic, seed=42",
                "backend": args.backend,
                "P": len(candidates),
                "K": args.budget,
                "repeats": len(times),
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "scope": "normalization + Gram + selection/refit; warmup excluded",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
