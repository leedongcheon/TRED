"""Answer normalization and set-level scoring.

Values are normalized by the kind of the gold answer:
- entity: case/underscore/whitespace-insensitive name,
- time:   zero-padded date (``A to A`` is the date ``A``),
- span:   ``A..B`` (gold ``A to B`` is converted), each side as a time,
- number: integer (gold ``1095 days`` -> ``1095``).
Only the surface format is normalized; times are compared at the granularity
written in the gold answer, the same way for every dataset.
"""

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_DATE_RE = re.compile(r"^(\d{1,4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$")
_PRECISION_PARTS = {"year": 1, "month": 2, "day": 3}
_SPAN_RE = re.compile(r"^(.+?)\s*(?:\.\.|\bto\b)\s*(.+)$")
_NUMBER_RE = re.compile(r"^([+-]?\d+)(?:\s*days?)?$", re.IGNORECASE)

KINDS = ("entity", "time", "span", "number")

# Dataset answer_type label -> answer kind (label conversion only).
TCR_ANSWER_KIND = {
    "subject": "entity",
    "object": "entity",
    "timestamp": "time",
    "timestamp_start": "time",
    "timestamp_end": "time",
    "timestamp_range": "span",
    "relation_union_or_intersection": "span",
    "duration": "number",
    "relation_duration": "number",
    "relation_ranking": "number",
}


def canonical_date(text: str, precision: Optional[str] = None) -> Optional[str]:
    match = _DATE_RE.match(str(text).strip())
    if match is None:
        return None
    parts = [f"{int(match.group(1)):04d}"] + [f"{int(g):02d}" for g in match.groups()[1:] if g is not None]
    if precision is not None:
        parts = parts[: _PRECISION_PARTS[precision]]
    return "-".join(parts)


def normalize_value(value: Any, kind: str, precision: Optional[str] = None) -> str:
    text = " ".join(str(value).split())
    if kind == "number":
        match = _NUMBER_RE.match(text)
        return str(int(match.group(1))) if match else text.casefold()
    if kind == "time":
        match = _SPAN_RE.match(text)
        if match:
            left = canonical_date(match.group(1), precision)
            if left is not None and left == canonical_date(match.group(2), precision):
                text = match.group(1)  # "A to A" is the time point A
        return canonical_date(text, precision) or text.casefold()
    if kind == "span":
        match = _SPAN_RE.match(text)
        if match:
            left, right = canonical_date(match.group(1), precision), canonical_date(match.group(2), precision)
            if left and right:
                return f"{left}..{right}"
        single = canonical_date(text, precision)
        return f"{single}..{single}" if single else text.casefold()
    if kind == "entity":
        return " ".join(text.replace("_", " ").split()).casefold()
    raise ValueError(f"unknown answer kind {kind!r}")


def normalize_set(values: Iterable[Any], kind: str, precision: Optional[str] = None) -> Set[str]:
    return {normalize_value(v, kind, precision) for v in values}


def top1(values: Set[str]) -> Optional[str]:
    """The element reported for Hits@1: first by a hash of the normalized value.

    The order must not follow time or the executor's output order (dates sort
    chronologically as strings, so plain lexicographic order would favour the
    earliest date), otherwise an unreduced set could hit "first" questions by luck.
    """
    return min(values, key=lambda v: hashlib.md5(v.encode("utf-8")).hexdigest()) if values else None


def f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def score_answer(
    pred: Iterable[Any],
    gold: Iterable[Any],
    kind: str,
    precision: Optional[str] = None,
    beta: float = 1.0,
) -> Dict[str, float]:
    p, g = normalize_set(pred, kind, precision), normalize_set(gold, kind, precision)
    inter = len(p & g)
    precision_v = inter / len(p) if p else 0.0
    recall_v = inter / len(g) if g else 0.0
    return {
        "em": float(bool(g) and p == g),
        "f1": f_beta(precision_v, recall_v, 1.0),
        "fbeta": f_beta(precision_v, recall_v, beta),
        "precision": precision_v,
        "recall": recall_v,
        "hit1": float(top1(p) in g) if p else 0.0,
        "overlap": float(inter > 0),  # diagnostic only; not a success criterion
    }


def infer_gran(values: Iterable[Any]) -> Optional[str]:
    """Finest date granularity written in ``values`` (None if no value is date-like)."""
    found = set()
    for value in values:
        text = " ".join(str(value).split())
        match = _SPAN_RE.match(text)
        for part in (match.group(1), match.group(2)) if match else (text,):
            date = _DATE_RE.match(part.strip())
            if date:
                found.add("day" if date.group(3) else "month" if date.group(2) else "year")
    return next((g for g in ("day", "month", "year") if g in found), None)


def score_episode(
    answer: Optional[Iterable[Any]],
    gold: Iterable[Any],
    kind: str,
    beta: float = 1.0,
) -> Dict[str, float]:
    """The single scoring function used for reward, filters, analyzer labels and evaluation.

    reward  = F_beta of the answer set (beta=1: F1; 0 without an answer, e.g. at the step limit);
    success = 1 only when F1 = 1 (partial answers are failures) -- episode_success, Stage 1 filter;
    hit1    = the hash-ordered top-1 answer is gold -- model selection and the headline metric.
    """
    if answer is None:
        scores = {"em": 0.0, "f1": 0.0, "fbeta": 0.0, "precision": 0.0, "recall": 0.0, "hit1": 0.0, "overlap": 0.0}
    else:
        scores = score_answer(answer, gold, kind, beta=beta)
    scores["answered"] = float(answer is not None)
    scores["reward"] = scores["fbeta"]
    scores["success"] = float(scores["f1"] == 1.0)
    return scores


def answer_spec(dataset: str, question: Dict[str, Any]) -> Tuple[List[Any], str, Optional[str]]:
    """Return ``(gold, kind, gran)`` for a raw dataset question record.

    ``gran`` (read from the gold strings) is the granularity used to read times
    off facts for coverage.
    """
    if dataset == "MultiTQ":
        gold, kind = list(question["answers"]), ("entity" if question["answer_type"] == "entity" else "time")
    elif dataset == "TimelineCronQR":
        gold, kind = list(question["answer"]), TCR_ANSWER_KIND[question["answer_type"]]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return gold, kind, (infer_gran(gold) if kind in ("time", "span") else None)
