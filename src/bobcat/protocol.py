"""System One shapes with the explicitly versioned public adapter confidence.

The formulas follow TypeSafe's MIT-licensed system-one-adapter-python,
revision fb52b1030b7fc1f4f1cf39910afa5da54f9835e3. See THIRD_PARTY.md.
Shape/formula compatibility is not numerical equivalence with the Jev model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

CONFIDENCE_PROFILE = "typesafe_adapter_2026_09_22"
MAX_QUESTIONS = 128
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10


class RequestLimitError(ValueError):
    """The validated request exceeds this checkpoint's supported token lengths."""


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def content(value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, (str, dict, list)):
        raise ValueError("Content must be a string, object, or array.")
    try:
        render(value).encode("utf-8")  # reject NaN, non-JSON objects, invalid Unicode
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("Content must be valid finite JSON and UTF-8.") from error


def decode_request(raw: bytes) -> dict:
    """Reject ambiguous duplicate keys before the JSON object becomes a dict."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=reject_constant,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid JSON encoding or nesting.") from error
    if not isinstance(payload, dict):
        raise ValueError("Request must be an object.")
    return payload


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    instructions: Any
    labels: tuple[str, ...]
    descriptions: tuple[str, ...]
    criteria: Any


def parse_request(payload: dict) -> tuple[Any, list[Question]]:
    if not isinstance(payload, dict):
        raise ValueError("Request must be an object.")
    if set(payload) - {"model", "state", "questions"}:
        raise ValueError("Request contains unsupported fields.")
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        raise ValueError("A nonempty model name is required.")
    content(payload.get("state"))
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError("This server accepts 1–128 named questions per request.")
    result = []
    for question_id, spec in questions.items():
        if not isinstance(question_id, str) or not isinstance(spec, dict):
            raise ValueError("Every question needs a string ID and an object.")
        content(question_id)
        if set(spec) - {"type", "instructions", "criteria"}:
            raise ValueError("Question contains unsupported fields.")
        kind, instructions = spec.get("type"), spec.get("instructions")
        content(instructions, nullable=True)
        criteria = spec.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_CHOICES:
                raise ValueError("Choice requires 1–255 named criteria.")
            labels, descriptions = [], []
            for label, description in criteria.items():
                if not isinstance(label, str) or not label:
                    raise ValueError("Choice names must be nonempty strings.")
                content(label)
                content(description, nullable=True)
                labels.append(label)
                # Public Jev contract: both the semantic name and description are input.
                descriptions.append(render(
                    {"name": label, "description": description}
                    if description is not None else label
                ))
        elif kind == "score":
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_SCORE_LEVELS:
                raise ValueError("Score requires an ordered array of 1–10 levels.")
            for level in criteria:
                content(level)
            labels = [str(i) for i in range(len(criteria))]
            descriptions = [render(level) for level in criteria]
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
            ):
                raise ValueError("Noul criteria may describe only true and false.")
            labels = ["no", "yes"]
            descriptions = []
            for value in ("false", "true"):
                description = (criteria or {}).get(value)
                content(description, nullable=True)
                descriptions.append(render({"value": value, "description": description}))
        else:
            raise ValueError("Question type must be choice, noul, or score.")
        result.append(Question(
            question_id, kind, instructions, tuple(labels), tuple(descriptions), criteria,
        ))
    return payload["state"], result


def probabilities(logits: list[float], temperature: float = 1.0) -> list[float]:
    if (not isinstance(logits, (list, tuple)) or not logits
            or type(temperature) not in (int, float)
            or not 0 < temperature or not math.isfinite(temperature)):
        raise ValueError("Need a score and a finite positive temperature.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in logits):
        raise ValueError("A decision score is non-finite.")
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    denominator = sum(weights)
    return [weight / denominator for weight in weights]


def _normalized(values: list[float]) -> list[float]:
    if not values or any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError("Confidence needs a nonempty finite nonnegative vector.")
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("Probability total must be finite.")
    return [p / total for p in values] if total else [1.0 / len(values)] * len(values)


def choice_confidence(values: list[float]) -> float:
    """Public adapter concentration statistic; not P(correct)."""
    normalized = _normalized(values)
    if len(normalized) == 1:
        return 1.0
    uniform = 1.0 / len(normalized)
    return max(0.0, min(1.0, (max(normalized) - uniform) / (1.0 - uniform)))


def score_confidence(values: list[float]) -> float:
    """Public adapter concentration around the first modal ordinal level."""
    normalized = _normalized(values)
    count = len(normalized)
    if count == 1:
        return 1.0
    mode = max(range(count), key=normalized.__getitem__)
    distance = sum(p * abs(i - mode) for i, p in enumerate(normalized))
    uniform_distance = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return max(0.0, 1.0 - distance / uniform_distance)


def answer(question: Question, values: list[float]) -> dict:
    if len(values) != len(question.labels):
        raise ValueError("Probability count does not match the requested criteria.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid probability.")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one.")
    if question.kind == "noul":
        return {"type": "noul", "noul": values[1]}
    result = {
        "type": question.kind,
        "confidence": (
            choice_confidence(values) if question.kind == "choice" else score_confidence(values)
        ),
        "probabilities": dict(zip(question.labels, values, strict=True)),
    }
    if question.kind == "choice":
        result["choice"] = question.labels[max(range(len(values)), key=values.__getitem__)]
    else:
        result["score"] = sum(i * p for i, p in enumerate(values))
        result["legend"] = {str(i): level for i, level in enumerate(question.criteria)}
    return result


def response(model: str, questions: list[Question], scores: list[list[float]],
             temperatures: dict[str, float], input_tokens: int = 0) -> dict:
    if not isinstance(scores, (list, tuple)) or len(questions) != len(scores):
        raise ValueError("Every question must have exactly one score vector.")
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("Input token usage must be a nonnegative integer.")
    return {
        "model": model,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "answers": {
            question.id: answer(
                question, probabilities(logits, temperatures.get(question.kind, 1.0)),
            )
            for question, logits in zip(questions, scores, strict=True)
        },
    }


def validate_response(result: Any, questions: list[Question], *, model: str) -> None:
    """Closed wire contract, independent of the neural backend or its raw reply.

    Only requested IDs/choice labels and the caller's ordinal legend can be
    strings supplied by a caller. Model explanations, messages and raw text have
    no field in a successful response. Invalid replies fail before serialization.
    """
    def number(value, lower, upper):
        return (type(value) in (int, float) and math.isfinite(value)
                and lower <= value <= upper)

    if (not isinstance(result, dict) or set(result) != {"model", "usage", "answers"}
            or result["model"] != model):
        raise ValueError("Backend response violates the closed decision contract.")
    usage = result["usage"]
    if (not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens"}
            or any(type(value) is not int or value < 0 for value in usage.values())
            or usage["output_tokens"] != 0):
        raise ValueError("Invalid decision usage.")
    answers = result["answers"]
    if (not isinstance(answers, dict) or len(questions) != len(answers)
            or set(answers) != {q.id for q in questions}):
        raise ValueError("Unexpected decision IDs.")
    for question in questions:
        item = answers[question.id]
        expected = ({"type", "noul"} if question.kind == "noul" else
                    {"type", "choice", "probabilities", "confidence"}
                    if question.kind == "choice" else
                    {"type", "score", "probabilities", "confidence", "legend"})
        if (not isinstance(item, dict) or set(item) != expected
                or item["type"] != question.kind):
            raise ValueError("Unexpected decision fields or type.")
        if question.kind == "noul":
            if not number(item["noul"], 0, 1):
                raise ValueError("Noul must be a finite probability.")
            continue
        probs = item["probabilities"]
        if (not isinstance(probs, dict) or set(probs) != set(question.labels)
                or any(not number(p, 0, 1) for p in probs.values())
                or not math.isclose(sum(probs.values()), 1., abs_tol=1e-6)
                or not number(item["confidence"], 0, 1)):
            raise ValueError("Invalid decision distribution.")
        if question.kind == "choice":
            if (not isinstance(item["choice"], str) or item["choice"] not in question.labels
                    or probs[item["choice"]] != max(probs.values())):
                raise ValueError("Choice must name a requested maximum-probability option.")
        elif (not number(item["score"], 0, len(question.labels) - 1)
              or not math.isclose(item["score"], sum(
                  i * probs[label] for i, label in enumerate(question.labels)
              ), abs_tol=1e-6)
              or item["legend"] != {
                  str(i): level for i, level in enumerate(question.criteria)
              }):
            raise ValueError("Score must follow the requested ordinal rubric.")
