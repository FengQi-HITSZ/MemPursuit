"""BEAM official-v1 answer generation and evaluation.

Prompt templates and scoring follow upstream commit
b2da22eac88bb0874c64665f13457eb99835774a. Answer and judge models are configurable.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from .models import BenchmarkQuestion, RetrievalResult
from .openai_gateway import OpenAIGateway


BEAM_OFFICIAL_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
BEAM_OFFICIAL_PROMPTS_SOURCE_SHA256 = (
    "81fe51ab2bcd7e8ad218b3a9a969a340c38d31f90ac2d93fa78a3bb890ba157b"
)
BEAM_OFFICIAL_COMPUTE_METRICS_SHA256 = (
    "e72bf25d36d521d166862063ec94f8e7875328193142d1edddb9b4677d47780d"
)
BEAM_READER_MAX_CONTEXT_TOKENS = 29_000


# Verbatim ``answer_generation_for_rag`` from official src/prompts.py.
ANSWER_GENERATION_FOR_RAG = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below. 

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation 

RESPONSE:
"""


# Verbatim ``unified_llm_judge_base_prompt`` from official src/prompts.py.
UNIFIED_LLM_JUDGE_BASE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""


# Verbatim binary-classifier text and user-message layout from
# src/evaluation/compute_metrics.py::llm_equivalence.
EVENT_EQUIVALENCE_SYSTEM_PROMPT = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """


def _token_count(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def answer_prompt(
    question: BenchmarkQuestion,
    result: RetrievalResult,
) -> tuple[str, dict[str, Any]]:
    """Apply the official RAG reader's 29K context budget and prompt."""
    context = ""
    included = 0
    hits = result.all_context()
    # The official implementation uses len(text)//3.5 as its reader-token
    # approximation and appends retrieved page_content without a separator.
    for hit in hits:
        if len(context + hit.text) // 3.5 < BEAM_READER_MAX_CONTEXT_TOKENS:
            context += hit.text
            included += 1
    prompt = ANSWER_GENERATION_FOR_RAG.replace("<context>", context).replace(
        "<question>", question.question
    )
    return prompt, {
        "reader_context_limit_tokens": BEAM_READER_MAX_CONTEXT_TOKENS,
        "reader_context_limit_estimator": "len(characters)//3.5 (official)",
        "retrieved_items": len(hits),
        "included_items": included,
        "omitted_items": len(hits) - included,
        "included_characters": len(context),
        "included_tokens_o200k": _token_count(context),
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.DOTALL)
        if match:
            value = match.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            raise ValueError(f"BEAM judge did not return a JSON object: {raw!r}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("BEAM judge JSON must be an object")
    return parsed


def _rubric_judgments(
    gateway: OpenAIGateway,
    rubric: tuple[str, ...],
    response: str,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    judgments = []
    usages = []
    total = 0.0
    for item in rubric:
        prompt = UNIFIED_LLM_JUDGE_BASE_PROMPT.replace("<rubric_item>", item).replace(
            "<llm_response>", response
        )
        raw, api_response = gateway.text(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        parsed = _parse_json_object(raw)
        score = float(parsed["score"])
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError(f"BEAM judge returned an unsupported score {score}")
        judgments.append({"rubric_item": item, "raw": raw, **parsed})
        usages.append(api_response.get("usage", {}))
        total += score
    return total / len(rubric), judgments, usages


def _event_alignment(
    gateway: OpenAIGateway,
    reference: tuple[str, ...],
    response: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from scipy.stats import kendalltau

    system = response.split("\n")
    used: set[int] = set()
    system_canon: list[str] = []
    comparisons = []
    usages = []
    for system_item in system:
        matched_index = None
        for reference_index, reference_item in enumerate(reference):
            if reference_index in used:
                continue
            user_prompt = f"""First snippet: {reference_item} \n
                       Second snippet: {system_item}
                    """
            raw, api_response = gateway.text(
                [
                    {"role": "system", "content": EVENT_EQUIVALENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            equivalent = "yes" in raw.lower()
            comparisons.append(
                {
                    "reference_index": reference_index,
                    "system_item": system_item,
                    "raw": raw,
                    "equivalent": equivalent,
                }
            )
            usages.append(api_response.get("usage", {}))
            if equivalent:
                matched_index = reference_index
                break
        if matched_index is not None:
            system_canon.append(reference[matched_index])
            used.add(matched_index)
        else:
            system_canon.append(system_item)

    reference_canon = list(reference)
    true_positive = len(set(reference_canon) & set(system_canon))
    false_positive = len([item for item in system_canon if item not in reference_canon])
    false_negative = len([item for item in reference_canon if item not in system_canon])
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(sequence: list[str]) -> list[int]:
        ranks = {item: index + 1 for index, item in enumerate(sequence)}
        return [ranks.get(item, tie_rank) for item in union]

    tau_b, p_value = kendalltau(
        to_rank(reference_canon), to_rank(system_canon), variant="b", method="auto"
    )
    tau_norm = (float(tau_b) + 1) / 2 if tau_b is not None and not math.isnan(float(tau_b)) else 0.0
    return (
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tau_norm": tau_norm,
            "p_value": float(p_value) if p_value is not None else None,
            "final_score": tau_norm * f1,
            "reference_canonical": reference_canon,
            "system_canonical": system_canon,
        },
        comparisons,
        usages,
    )


def evaluate_answer(
    gateway: OpenAIGateway,
    question: BenchmarkQuestion,
    response: str,
) -> dict[str, Any]:
    """Run the official rubric order, including event-order LLM alignment."""
    if not question.rubric:
        raise ValueError("BEAM official evaluation requires a non-empty rubric")
    started = time.monotonic()
    event_metrics = None
    event_comparisons: list[dict[str, Any]] = []
    alignment_usage: list[dict[str, Any]] = []
    if question.category == "event_ordering":
        event_metrics, event_comparisons, alignment_usage = _event_alignment(
            gateway, question.rubric, response
        )
    rubric_score, rubric_judgments, rubric_usage = _rubric_judgments(
        gateway, question.rubric, response
    )
    # Official report_results.py uses tau_norm for event ordering and the mean
    # rubric judge score for all other abilities.
    primary_score = (
        float(event_metrics["tau_norm"])
        if event_metrics is not None
        else rubric_score
    )
    return {
        "primary_score": primary_score,
        "llm_judge_score": rubric_score,
        "rubric_judgments": rubric_judgments,
        "event_ordering": event_metrics,
        "event_alignment_comparisons": event_comparisons,
        "usage": {
            "rubric": rubric_usage,
            "event_alignment": alignment_usage,
        },
        "elapsed_seconds": time.monotonic() - started,
        "official_report_rule": (
            "tau_norm" if question.category == "event_ordering" else "llm_judge_score"
        ),
    }
