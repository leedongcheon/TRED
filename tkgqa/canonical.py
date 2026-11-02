"""Canonical-form check for Stage 1 trajectories (data filter only, never a reward).

A trajectory is canonical when:
- it ends with ANSWER (no step limit), every step parses (no invalid action),
  and each response is exactly one <action> block;
- every variable feeds the answer, except a failed FIND retried under the same
  name and a FIND re-queried later with the same entities but another relation;
- a question with first/last/earliest/latest selects explicitly: the answer's
  lineage has a PICK or a TIME(first|last), even when one fact happens to remain;
- variables of each type are numbered 1, 2, ... in creation order;
- date literals only restate dates written in the question (``May 2015`` may
  be written ``2015-05``; a mentioned date may be coarsened to its month/year);
  derived times must come from TIME;
- entity names in FIND are taken from the question or from earlier
  observations, so the writer cannot inject entities it knows on its own.
"""

import re
from typing import Iterable, List, Set

from tkgqa.answers import canonical_date
from tkgqa.operations import parse
from tkgqa.executor import Session

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_MONTHS = {m: i + 1 for i, m in enumerate(_MONTH_NAMES)}
_MONTHS.update({m[:3]: i + 1 for i, m in enumerate(_MONTH_NAMES)})
_MONTHS["sept"] = 9
# Full names and abbreviations ("Sep", "Sept.", "Apr").
_MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
_YEAR = r"(\d{3,4})"
# A year must not be part of a longer number ("6,574", "1.2", "2009-12"); a comma or
# hyphen only counts when a digit follows ("2007, with ..." is a year).
_NO_NUM_BEFORE = r"(?<![\d.\-])(?<!\d,)"
_NO_NUM_AFTER = r"(?!\d)(?!,\d)(?!-\d)(?!\s*(?:days?|weeks?|months?|years?)\b)"
_DATE_PATTERNS = [
    # 2009-12-3, 1983-01-01, 2015-05
    (re.compile(_NO_NUM_BEFORE + r"(\d{3,4})-(\d{1,2})(?:-(\d{1,2}))?(?![\d])"), ("y", "m", "d")),
    # 27 February 2007
    (re.compile(r"\b(\d{1,2})\s+" + _MONTH + r",?\s+" + _YEAR + r"\b", re.I), ("d", "M", "y")),
    # September 17, 2009 / September 17 2009
    (re.compile(_MONTH + r"\s+(\d{1,2}),?\s+" + _YEAR + r"\b", re.I), ("M", "d", "y")),
    # September 2009 / May, 2015 / February of 2015
    (re.compile(_MONTH + r",?\s+(?:of\s+)?" + _YEAR + r"\b", re.I), ("M", "y")),
    # 2005
    (re.compile(_NO_NUM_BEFORE + r"(\d{3,4})" + _NO_NUM_AFTER), ("y",)),
]
_SUPERLATIVE_RE = re.compile(r"\b(first|last|earliest|latest)\b", re.I)
_STOPWORDS = {"the", "of", "a", "an", "and", "or", "in", "on", "for", "to", "s", "at", "by"}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def question_dates(question: str) -> Set[str]:
    """Canonical dates written in ``question`` plus their month/year coarsenings."""
    found: Set[str] = set()
    for pattern, fields in _DATE_PATTERNS:
        for match in pattern.finditer(question):
            parts = dict(zip(fields, match.groups()))
            year = parts.get("y")
            month = _MONTHS[parts["M"].lower().rstrip(".")] if "M" in parts else (int(parts["m"]) if parts.get("m") else None)
            day = int(parts["d"]) if parts.get("d") else None
            if month is not None and not 1 <= month <= 12:
                continue
            text = year + (f"-{month}" if month else "") + (f"-{day}" if month and day else "")
            date = canonical_date(text)
            if date is None:
                continue
            pieces = date.split("-")
            for n in range(1, len(pieces) + 1):
                found.add("-".join(pieces[:n]))
    return found


def _tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(str(text).replace("_", " ").casefold()) if t not in _STOPWORDS}


def _same_word(a: str, b: str) -> bool:
    """Equal, or inflections of one word ("Liberia"/"Liberian", "authority"/"authorities")."""
    if a == b:
        return True
    shorter = min(len(a), len(b))
    if shorter < 4:
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= max(4, shorter - 2)


def name_in_context(name: str, contexts: Iterable[str]) -> bool:
    tokens = _tokens(name)
    if not tokens:
        return True
    for context in contexts:
        words = _tokens(context)
        if all(t in words or any(_same_word(t, w) for w in words) for t in tokens):
            return True
    return False


def _literal_endpoints(text: str) -> List[str]:
    return [p.strip() for p in text.split("..")]


def _requeried(session: Session, name: str, lineage: Set[str]) -> bool:
    """A FIND abandoned for a later FIND on the same entities with another relation."""
    var = session.vars[name]
    ids = var.info.get("ids") if var.op == "FIND" else None
    if ids is None:
        return False
    for other in session.vars.values():
        later = other.info.get("ids") if other.op == "FIND" else None
        if later and other.name in lineage and other.step > var.step:
            if later[0] == ids[0] and later[2] == ids[2] and later[1] != ids[1]:
                return True
    return False


def _has_selection(session: Session, lineage: Set[str]) -> bool:
    for name in lineage:
        var = session.vars[name]
        if var.op == "PICK":
            return True
        if var.op == "TIME" and len(var.args) > 1 and getattr(var.args[1], "text", None) in ("first", "last"):
            return True
    return False


def check_canonical(session: Session) -> List[str]:
    """Return the violated rules (empty list = canonical)."""
    violations: List[str] = []
    if not session.done or session.truncated or session.answer_var is None:
        violations.append("no_answer")
    if any(step.status == "invalid" for step in session.log):
        violations.append("invalid_step")
    for step in session.log:
        if re.fullmatch(r"\s*<action>(?:(?!<action>).)+?</action>\s*", step.response, re.S) is None:
            violations.append("format")
            break

    if session.answer_var is not None:
        lineage = set(session.ancestors(session.answer_var))
        if any(name not in lineage and not _requeried(session, name, lineage) for name in session.vars):
            violations.append("unused_variable")
        if _SUPERLATIVE_RE.search(session.question) and not _has_selection(session, lineage):
            violations.append("missing_selection")

    for i, step in enumerate(session.log):
        if step.status != "find_failed":
            continue
        name = (step.action or "").split("=", 1)[0].strip()
        recovered = any(
            later.status == "ok" and later.op == "FIND" and later.var == name for later in session.log[i + 1 :]
        )
        if not recovered:
            violations.append("unrecovered_find_failure")
            break

    for vtype in "fts":
        names = [v.name for v in sorted(session.vars.values(), key=lambda v: v.step) if v.type == vtype]
        if names != [f"{vtype}{k}" for k in range(1, len(names) + 1)]:
            violations.append("numbering")
            break

    allowed_dates = question_dates(session.question)
    contexts = [session.question]
    for step in session.log:
        if step.status != "invalid" and step.action:
            stmt = parse(step.action)
            if stmt.op == "FILTER":
                for arg in stmt.args[2:4]:
                    if arg.kind in ("num", "str"):
                        endpoints = [canonical_date(p) for p in _literal_endpoints(arg.text)]
                        if any(e is None or e not in allowed_dates for e in endpoints):
                            violations.append("date_not_in_question")
            if stmt.op == "FIND":
                for slot in (0, 2):
                    name = stmt.args[slot].text
                    if name.strip() != "?" and not name_in_context(name, contexts):
                        violations.append("entity_not_in_context")
        contexts.append(step.obs)
    return list(dict.fromkeys(violations))
