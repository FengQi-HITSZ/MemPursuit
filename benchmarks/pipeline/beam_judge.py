from __future__ import annotations
import json
import re
from typing import Any
from .models import BenchmarkQuestion
from .openai_gateway import OpenAIGateway
from .beam_official import evaluate_answer as evaluate_beam_official_answer


def _repair_invalid_json_escapes(raw: str) -> str:
    """Make only syntactically invalid JSON backslash runs parseable.

    A regex that examines each backslash independently is insufficient here:
    in a run of two backslashes the first one legally escapes the second.  If
    the second happens to precede ``(``, rewriting that second slash produces
    a run of three and leaves the JSON invalid.  Treat each consecutive run as
    a unit.  Even runs are already self-escaped; for odd runs the final slash
    must introduce a valid JSON escape, including a complete four-hex-digit
    unicode escape.  Otherwise append one slash to make the run even and
    preserve the following free text literally.
    """

    repaired: list[str] = []
    index = 0
    simple_escapes = frozenset('"\\/bfnrt')
    hex_digits = frozenset("0123456789abcdefABCDEF")
    while index < len(raw):
        if raw[index] != "\\":
            repaired.append(raw[index])
            index += 1
            continue
        end = index
        while end < len(raw) and raw[end] == "\\":
            end += 1
        count = end - index
        valid_final_escape = count % 2 == 0
        if not valid_final_escape and end < len(raw):
            escape = raw[end]
            valid_final_escape = escape in simple_escapes
            if escape == "u":
                digits = raw[end + 1 : end + 5]
                valid_final_escape = len(digits) == 4 and all(char in hex_digits for char in digits)
        if not valid_final_escape:
            count += 1
        repaired.append("\\" * count)
        index = end
    return "".join(repaired)


def _recover_unambiguous_beam_score_json(raw: str) -> tuple[str, str | None]:
    """Recover only an explicit, unique official BEAM score literal.

    The provider can return an object whose free-text ``reason`` is malformed
    JSON even though its score field is clear.  The official metric consumes
    only that score.  Never infer it: recovery is allowed only for exactly one
    quoted ``score`` key with one of the protocol's supported numeric values.
    """

    value = raw.strip()
    try:
        json.loads(value)
        return raw, None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if match:
            candidate = match.group(0)
        elif value.startswith("{"):
            # Providers occasionally truncate only the free-text reason (and
            # therefore the closing brace) after emitting a complete score.
            # The BEAM metric consumes only the score, so the same strict,
            # unique-literal recovery remains safe for this partial object.
            candidate = value
        else:
            return raw, None
    try:
        json.loads(candidate)
        return raw, None
    except json.JSONDecodeError as parse_error:
        score_literals = re.findall(
            r'"score"\s*:\s*(0(?:\.0+)?|0\.5(?:0*)?|1(?:\.0+)?)(?![\d.])',
            candidate,
        )
        if len(score_literals) != 1:
            return raw, None
        score = float(score_literals[0])
        if score not in {0.0, 0.5, 1.0}:
            return raw, None
        error = f"{parse_error.msg}: line {parse_error.lineno} column {parse_error.colno}"
        recovered = json.dumps(
            {
                "score": score,
                "reason": "",
                "operational_score_literal_recovery": True,
                "operational_json_parse_error": error,
            },
            ensure_ascii=False,
        )
        return recovered, error


class _BeamJudgeTransportGateway:
    """Apply conservative JSON recovery while retaining provider raw text."""

    def __init__(self, gateway: OpenAIGateway):
        self.gateway = gateway
        self.repairs: dict[str, list[tuple[str, bool, bool]]] = {}
        self.repair_count = 0
        self.score_recovery_count = 0

    def text(self, messages: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        raw, response = self.gateway.text(messages, **kwargs)
        # JSON permits only \" \\ \/ \b \f \n \r \t and \u escapes. The
        # judge sometimes emits LaTeX delimiters such as \( and \), which are
        # semantically ordinary reason text but make the whole object invalid.
        repaired = _repair_invalid_json_escapes(raw)
        escape_repaired = repaired != raw
        if escape_repaired:
            self.repair_count += 1
        recovered, recovery_error = _recover_unambiguous_beam_score_json(repaired)
        score_recovered = recovery_error is not None
        if score_recovered:
            repaired = recovered
            self.score_recovery_count += 1
        if escape_repaired or score_recovered:
            self.repairs.setdefault(repaired, []).append((raw, escape_repaired, score_recovered))
        return repaired, response

    def restore_raw_judgments(self, result: dict[str, Any]) -> None:
        for judgment in result.get("rubric_judgments", []):
            repaired = judgment.get("raw")
            repairs = self.repairs.get(repaired, [])
            if repairs:
                raw, escape_repaired, score_recovered = repairs.pop(0)
                judgment["raw"] = raw
                if escape_repaired:
                    judgment["operational_json_escape_repair"] = True
                if score_recovered:
                    judgment["operational_score_literal_recovery"] = True


def _evaluate_beam_with_parse_retries(
    gateway: OpenAIGateway,
    question: BenchmarkQuestion,
    raw_answer: str,
    *,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """Retry malformed judge JSON without changing the official protocol.

    The official prompt can occasionally elicit an otherwise valid object
    whose free-text reason contains an illegal JSON backslash escape. HTTP
    retries cannot observe that parse failure. Reissuing the identical
    temperature-zero judge protocol avoids repairing or reinterpreting scores.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    transport_gateway = _BeamJudgeTransportGateway(gateway)
    for attempt in range(max_attempts):
        try:
            result = evaluate_beam_official_answer(transport_gateway, question, raw_answer)
            transport_gateway.restore_raw_judgments(result)
            result["operational_parse_retries"] = attempt
            result["operational_json_escape_repairs"] = transport_gateway.repair_count
            result["operational_score_literal_recoveries"] = transport_gateway.score_recovery_count
            return result
        except (json.JSONDecodeError, ValueError):
            if attempt + 1 >= max_attempts:
                raise
    raise AssertionError("unreachable")
