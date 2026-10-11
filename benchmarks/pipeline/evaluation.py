from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_accuracy(
    records: list[dict[str, Any]],
    *,
    cluster_key: str,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[str(record[cluster_key])].append(float(record["correct"]))
    keys = sorted(groups)
    if not keys:
        return {"low": math.nan, "high": math.nan}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [rng.choice(keys) for _ in keys]
        values = [value for key in selected for value in groups[key]]
        samples.append(mean(values))
    return {"low": _percentile(samples, 0.025), "high": _percentile(samples, 0.975)}


def summarize(records: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["prompt_track"]), int(record["cutoff"]))].append(record)
    tables = []
    for (track, cutoff), rows in sorted(groups.items()):
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[str(row["category"])].append(row)
        category_metrics = {}
        for category, category_rows in sorted(by_category.items()):
            category_metrics[category] = {
                "count": len(category_rows),
                "accuracy": mean(float(row["correct"]) for row in category_rows),
                "exact_match": mean(float(row["exact_match"]) for row in category_rows),
                "token_f1": mean(float(row["token_f1"]) for row in category_rows),
            }
        cluster_key = "scope_id" if benchmark in {"locomo", "beam"} else "question_id"
        tables.append(
            {
                "prompt_track": track,
                "cutoff": cutoff,
                "count": len(rows),
                "overall": {
                    "accuracy": mean(float(row["correct"]) for row in rows),
                    "exact_match": mean(float(row["exact_match"]) for row in rows),
                    "token_f1": mean(float(row["token_f1"]) for row in rows),
                    "mean_context_tokens": mean(float(row["retrieval"]["context_tokens"]) for row in rows),
                    "mean_context_items": mean(float(row["retrieval"]["actual_count"]) for row in rows),
                    "mean_retrieval_seconds": mean(float(row["timing"]["retrieval_seconds"]) for row in rows),
                    "mean_answer_seconds": mean(float(row["timing"]["answer_seconds"]) for row in rows),
                    "mean_judge_seconds": mean(float(row["timing"]["judge_seconds"]) for row in rows),
                    "accuracy_95_ci": bootstrap_accuracy(rows, cluster_key=cluster_key),
                    "uncertainty_note": (
                        "conversation-cluster bootstrap over this single memory construction"
                        if benchmark in {"locomo", "beam"}
                        else "question bootstrap over this single memory construction"
                    ),
                },
                "by_category": category_metrics,
            }
        )
    return {"benchmark": benchmark, "tables": tables}
