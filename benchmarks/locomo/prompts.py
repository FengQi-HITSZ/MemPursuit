"""
LOCOMO Benchmark Prompts
========================

Answer generation and category-specific judge prompts for the LOCOMO
benchmark (Snap Research, ACL 2024). Uses the industry-standard J-score
methodology: binary LLM judge (CORRECT/WRONG) on categories 1-4.

Category mapping:
    1 = multi-hop (282 questions)
    2 = temporal reasoning (321 questions)
    3 = open-domain (96 questions)
    4 = single-hop (841 questions)
    5 = adversarial (446 questions) — excluded from scoring
"""

from datetime import datetime as _datetime
from typing import List


# ===============================================================================
# CATEGORY NAME MAPPING
# ===============================================================================

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]
JUDGE_PROMPT_VERSION = "locomo_mem0_accuracy_v1"


# ===============================================================================
# ANSWER GENERATION PROMPT
# ===============================================================================

ANSWER_GENERATION_PROMPT = """You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."

# Instructions

## Misc

1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people
2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").
3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.

{memories}

Question: {question}

Work through Steps 1-7, then give your final answer after "ANSWER:".
"""


ANSWERER_MEMORY_LIMIT = 200


def _to_human_date(iso_str: str) -> str:
    """Convert ISO 8601 timestamp to human-readable date (e.g., 'May 7, 2023')."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _datetime.strptime(iso_str[:26].rstrip("Z"), fmt.replace("%z", "")).strftime("%A, %B %d, %Y")
        except ValueError:
            continue
    # Last resort: just show the date portion
    return iso_str[:10]


def _format_user_profile(user_profile: dict) -> str:
    """Format a user profile dict as readable key-value pairs for the prompt.

    Omits keys with null/empty values. Formats lists as comma-separated strings.
    """
    lines = ["## User Profile"]
    for key, value in user_profile.items():
        if value is None:
            continue
        # Format key: snake_case -> Title Case
        display_key = key.replace("_", " ").title()
        if isinstance(value, list):
            if not value:
                continue
            display_value = ", ".join(str(v) for v in value)
        elif isinstance(value, str):
            if not value.strip():
                continue
            display_value = value
        else:
            display_value = str(value)
        lines.append(f"{display_key}: {display_value}")
    # Only return if we have at least one field beyond the header
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def get_answer_generation_prompt(
    question: str,
    search_results: list,
    reference_date: str = None,
    user_profile: dict = None,
) -> str:
    """Build the answer generation prompt from search results.

    Shows the top ANSWERER_MEMORY_LIMIT memories sorted chronologically
    (oldest first) with human-readable dates. No rank numbers or scores
    are shown to avoid anchoring bias.

    Args:
        question: The question to answer.
        search_results: List of memory dicts from search.
        reference_date: Human-readable date string (e.g., "January 04, 2024")
            representing when the conversations took place. Used for temporal
            reasoning. Defaults to "2023" if not provided.
        user_profile: Optional dict of user profile data. When provided, a
            User Profile section is added to the prompt before the memories.
    """
    if reference_date is None:
        reference_date = "2023"

    if not search_results:
        memories_text = "(No relevant memories found)"
    else:
        top_results = search_results[:ANSWERER_MEMORY_LIMIT]
        # Sort chronologically (oldest first) to present as a narrative
        sorted_results = sorted(top_results, key=lambda x: x.get("created_at", ""))
        lines = ["The following memories are presented in chronological order (oldest to newest).", ""]
        for result in sorted_results:
            memory = result.get("memory", "")
            created_at = result.get("created_at", "")
            if created_at:
                date_str = _to_human_date(created_at)
                lines.append(f"({date_str}) {memory}")
            else:
                lines.append(f"(unknown date) {memory}")
        memories_text = "\n".join(lines)

    # Optionally prepend user profile section before memories
    profile_section = ""
    if user_profile:
        profile_section = _format_user_profile(user_profile)
        if profile_section:
            profile_section = profile_section + "\n\n"

    return ANSWER_GENERATION_PROMPT.format(
        memories=profile_section + memories_text,
        question=question,
        reference_date=reference_date,
    )


# ===============================================================================
# JUDGE PROMPT (unified — evidence is an optional add-on)
# ===============================================================================

# The reference Mem0 evaluator sends only ACCURACY_PROMPT as a user message.
# Keep the system prompt empty so judge-only reruns match that setup.
JUDGE_SYSTEM_PROMPT = ""

# ── Evidence chunk (injected when evidence is available) ──
_EVIDENCE_CHUNK = """
## Evidence (actual conversation messages containing the answer)
{evidence_context}
"""

# This is the Mem0 LoCoMo LLM-judge prompt from
# evaluation/metrics/llm_judge.py. Keep the wording intact so results remain
# comparable with memory papers that use the Mem0 evaluation protocol.
ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def _build_judge_prompt(evidence_context: str = None) -> str:
    """Build the Mem0 accuracy prompt, optionally including source evidence."""
    if not evidence_context:
        return ACCURACY_PROMPT
    evidence_section = _EVIDENCE_CHUNK.format(evidence_context=evidence_context)
    return ACCURACY_PROMPT.replace(
        "Now it's time for the real question:",
        f"{evidence_section}\nNow it's time for the real question:",
    )


# Keep JUDGE_PROMPT as a module-level constant for backward compat (no evidence)
JUDGE_PROMPT = _build_judge_prompt(evidence_context=None)


# ===============================================================================
# DISPATCH
# ===============================================================================

def get_judge_prompt(
    category: int,
    question: str,
    answer: str,
    response: str,
) -> str:
    """Return the formatted unified judge prompt (no evidence).

    Accepts category for backwards compatibility but uses the same
    unified prompt for all categories.
    """
    return JUDGE_PROMPT.format(
        question=question,
        gold_answer=answer,
        generated_answer=response,
    )


def get_judge_prompt_with_evidence(
    category: int,
    question: str,
    answer: str,
    response: str,
    evidence_context: str,
) -> str:
    """Return the formatted judge prompt with evidence context.

    Same prompt as get_judge_prompt but with the evidence section
    and evidence-specific rule injected.
    """
    prompt = _build_judge_prompt(evidence_context=evidence_context)
    return prompt.format(
        question=question,
        gold_answer=answer,
        generated_answer=response,
    )


def preprocess_answer(category: int, answer: str) -> str:
    """Preprocess ground truth answer based on category.

    Category 3 (open-domain): use only the first part before semicolon.
    """
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return answer
