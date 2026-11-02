"""Filter for analyzer skills before they reach a teacher prompt (TKGQA; applied by the trainer after the
SEED analyzer's own parse). A rejected skill leaves its trajectory without a teacher (OPD mask 0).

Rules: at most 60 words; starts with "When"; names a temporal operation and one of its arguments; valid operations (no
nested operation, no literal assigned to a variable); never an empty answer or giving up; a failure's skill
is an avoidance rule (an avoidance marker: do not, don't, never, avoid, instead, rather than, not from, not on)
and names an operation that fits the analyzer's failure_type (FAILURE_TYPE_OPS; whether it names the operation of
the failure_step itself is a diagnostic only, names_failure_step_op: the fix often uses another operation than the
mistake; the suspect steps of the diagnostic hints are only compared, in the trainer's log); and
nothing specific to the episode: no name it used or saw (FIND arguments, names in observations such as FIND
candidates, other relations, grounded triples and fact tuples, values, the submitted answer), dates, variable
numbers (f1, t2, s1), recall/gold/hints, step numbers or lone capital placeholders (X, Y).
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from tkgqa.operations import OPERATIONS_DOC, SIGS

_OP_RE = re.compile(r"\b(" + "|".join(SIGS) + r")\b")
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)
_ACTION_OP_RE = re.compile(r"\b(" + "|".join(SIGS) + r")\s*\(")
_QUOTED_RE = re.compile(r'"([^"]+)"')
_NESTED_RE = re.compile(r"\b(" + "|".join(SIGS) + r")\s*\([^()]*\b(" + "|".join(SIGS) + r")\s*\(")
_LITERAL_ASSIGN_RE = re.compile(r"\b[fts]\d*\s*=\s*[\"'\d]")
_VARIABLE_RE = re.compile(r"\b[fts]\d+\b")
_DATE_RE = re.compile(r"\b\d{4}(?:-\d{2}){0,2}\b")
_HINDSIGHT_TERM_RE = re.compile(r"\b(?:recall|gold|hints?)\b", re.I)
_STEP_NUMBER_RE = re.compile(r"\bsteps?\s*\d", re.I)
# A lone capital letter standing for an entity / relation ("who did X to Y"); K (PICK's K) and N (shift=N) are operation words.
_PLACEHOLDER_RE = re.compile(r"(?<![\w'(=,/-])(?![AIKN]\b)[A-Z](?![\w(=])")
_SUSPECTS_RE = re.compile(r"^suspects:\s*\[([^\]]*)\]", re.M)
_EMPTY_ANSWER_RE = re.compile(
    r"ANSWER\s*\(\s*\[\s*\]\s*\)|\bANSWER\s+(?:with\s+)?(?:an?\s+)?(?:empty|nothing|none)\b"
    r"|\b(?:give|gives|giving)\s+up\b|\breturn\s+(?:an?\s+)?empty\s+(?:answer|set|result)\b"
    r"|\banswer\s+(?:with\s+)?(?:an?\s+)?empty\b|\bleave\s+the\s+answer\s+empty\b",
    re.IGNORECASE,
)


def _operation_argument_words(doc: str) -> set:
    """Argument keywords read off the operation document (alternations, FILTER relations, keyword names)."""
    words = set()
    for alt in re.findall(r"\b[a-z]+(?:\|[a-z]+)+\b", doc):
        words.update(alt.split("|"))
    for line in doc.splitlines():  # FILTER relation definitions ("    before   = fact ends ...")
        if line.startswith("    ") and "=" in line:
            words.update(re.findall(r"\b([a-z]+)\s+=\s", line))
    words.update(re.findall(r"\b([a-z]+)=", doc))  # keyword arguments
    for signature in re.findall(r"\b[A-Z]+\(([^)]*)\)", doc):  # bare words in operation signatures
        signature = re.sub(r'"[^"]*"', "", signature)
        words.update(re.findall(r"\b[a-z]{3,}\b", signature))
    words.update({"subj", "rel", "obj", "literal"})
    return words


ARGUMENT_WORDS = _operation_argument_words(OPERATIONS_DOC)
# Arguments that are also everyday English words only count in an operation context.
_COMMON_WORDS = {"before", "after", "first", "last", "all", "time", "start", "end", "during",
                 "count", "sum", "order", "by", "unit", "day", "month", "year"}


def mentions_argument(skill: str) -> bool:
    calls = " ".join(re.findall(r"\b(?:" + "|".join(SIGS) + r")\s*\(([^()]*)\)", skill))
    if "," in calls:  # an operation written out with its arguments
        return True
    quoted = " ".join(re.findall(r"['\"]([^'\"]+)['\"]", skill)).lower()
    keyed = " ".join(re.findall(r"\b([a-z]+)\s*=", skill.lower()))
    tokens = re.findall(r"[A-Za-z]+", skill)
    near_op = set()
    for i, tok in enumerate(tokens):
        if tok in SIGS:
            near_op.update(t.lower() for t in tokens[i + 1 : i + 5] if t not in SIGS)
    for word in ARGUMENT_WORDS:
        pattern = r"\b" + re.escape(word) + (r"\w*" if word in ("subj", "obj", "rel") else r"\b")
        if word in _COMMON_WORDS:
            if re.search(pattern, quoted) or re.search(pattern, keyed) or word in near_op or re.search(pattern, calls.lower()):
                return True
        elif re.search(pattern, skill.lower()):
            return True
    return False


def suspect_steps(steps: Sequence[Dict[str, Any]]) -> Optional[List[int]]:
    """The suspects line of the episode's diagnostic hints (on the last step record), or None if absent."""
    hints = str(steps[-1].get("auto_hints", "")) if steps else ""
    match = _SUSPECTS_RE.search(hints)
    if match is None:
        return None
    return [int(x) for x in re.findall(r"-?\d+", match.group(1))]


def _step_op(step: Dict[str, Any]) -> Optional[str]:
    action = _ACTION_RE.search(str(step.get("response", "")))
    match = _ACTION_OP_RE.search(action.group(1)) if action else None
    return match.group(1) if match else None


_LIST_RE = re.compile(r"(?:candidates|other relations|values):\s*([^|\n]+)")
_DATE_ONLY_RE = re.compile(r"^\d{4}(?:-\d{2}){0,2}(?:\.\.\d{4}(?:-\d{2}){0,2})?$")


def _top_level_groups(text: str) -> List[str]:
    """Contents of the top-level (...) groups of ``text`` (names may hold their own parentheses)."""
    groups, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
            if depth == 0:
                groups.append(text[start:i])
    return groups


def _split_top_level(text: str, sep: str = ", ") -> List[str]:
    parts, depth, current = [], 0, ""
    i = 0
    while i < len(text):
        ch = text[i]
        depth += ch == "("
        depth -= ch == ")" and depth > 0
        if depth == 0 and text.startswith(sep, i):
            parts.append(current)
            current, i = "", i + len(sep)
            continue
        current += ch
        i += 1
    return parts + [current]


def _names_in_observation(obs: str) -> set:
    """Names an executor observation shows: quoted names, candidate / relation / value lists, fact tuples."""
    names = set(_QUOTED_RE.findall(obs))
    for group in _top_level_groups(obs):  # grounded: (s, r, o) and facts (s, r, o, date)
        names.update(_split_top_level(group))
    for match in _LIST_RE.finditer(obs):  # candidates: a, b | other relations: a, b | values: a, b
        for part in _split_top_level(match.group(1).strip().rstrip(".")):
            names.add(part if "(" not in part or not part.startswith("(") else part[1:-1])
    return names


def episode_names(steps: Sequence[Dict[str, Any]]) -> List[str]:
    """Names the episode used or saw: quoted FIND arguments, everything the observations show (quoted names,
    FIND candidates, other relations, grounded triples, fact tuples, value lists) and the submitted answer.
    operation keywords, "?", numbers and dates are left out (dates are caught by their own rule)."""
    names = set()
    for step in steps:
        action = _ACTION_RE.search(str(step.get("response", "")))
        if action and "FIND" in action.group(1):
            names.update(_QUOTED_RE.findall(action.group(1)))
        names |= _names_in_observation(str(step.get("observation", "")))
    final = str(steps[-1].get("final_observation", "")) if steps else ""
    if final.startswith("Answer submitted: "):
        names.update(_split_top_level(final[len("Answer submitted: "):].split("\n")[0]))
    names |= _names_in_observation(final)
    keep = set()
    for name in names:
        name = name.strip().strip(".;")
        if len(name) < 3 or name == "?" or _DATE_ONLY_RE.match(name) or re.fullmatch(r"[\d.,+-]+", name):
            continue
        if name.lower() in ARGUMENT_WORDS or name in SIGS:
            continue
        keep.add(name)
    return sorted(keep)


def _contains_name(skill: str, name: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", skill, re.I) is not None


_CAPITAL_WORDS = {"The", "It", "Then", "This", "That", "After", "Before", "When", "Finally", "First", "Next", "However",
                  "Agent", "Answer", "Its", "Instead", "Both", "All"}


_TEMPORAL_MARKER = r"(?:before|after|following|prior to|since|until|preceding)"
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_TEXT_DATE_RE = re.compile(r"\b(?:(\d{1,2})\s+)?(" + "|".join(_MONTHS) + r")\s+(\d{4})\b|\b(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?\b", re.I)


def episode_relations(steps: Sequence[Dict[str, Any]]) -> List[str]:
    """Relation names of the episode: FIND relation arguments, grounded / fact triples, other-relation lists."""
    rels = set()
    for step in steps:
        action = _ACTION_RE.search(str(step.get("response", "")))
        if action and "FIND" in action.group(1):
            quoted = _QUOTED_RE.findall(action.group(1))
            if len(quoted) >= 2:
                rels.add(quoted[1])
        obs = str(step.get("observation", ""))
        for group in _top_level_groups(obs):
            parts = _split_top_level(group)
            if len(parts) >= 3:
                rels.add(parts[1])
        for match in re.finditer(r"other relations:\s*([^|\n]+)", obs):
            rels.update(_split_top_level(match.group(1).strip().rstrip(".")))
    return sorted(r.strip() for r in rels if len(r.strip()) >= 3 and r.strip() != "?")


def _iso_prefix(match) -> str:
    if match.group(2):
        day, month, year = match.group(1), _MONTHS.index(match.group(2).lower()) + 1, match.group(3)
        return f"{year}-{month:02d}" + (f"-{int(day):02d}" if day else "")
    return "-".join(x for x in match.group(4, 5, 6) if x)


def scrub_names(text: str, names: Sequence[str], question: str = "", answers: Sequence[str] = (),
                relations: Sequence[str] = ()) -> str:
    """``text`` with every episode name (longest first) replaced by its role, used on the episode_summary of the
    analyzer SFT targets so the analyzer never learns to write names: "the relation" (a relation name), "the answer
    entity" (a gold or submitted answer), "the reference entity" (named right after before / after / ... in the
    question), "the subject entity" (named elsewhere in the question), "another entity" (the rest); dates become
    "the answer date" (a gold answer) or "the anchor date". Short forms count too: the parts of "Head (Country)" and
    capitalised phrases of the text found inside a name."""
    variants = {}
    for name in names:
        for v in [name, re.sub(r"\s*\([^()]*\)\s*$", "", name), *re.findall(r"\(([^()]*)\)", name)]:
            v = re.sub(r"^(?:the|a|an)\s+", "", v.strip(), flags=re.I)
            if len(v) >= 3 and v.lower() not in ARGUMENT_WORDS and v not in SIGS:
                variants.setdefault(v, name)
    lowered = [n.lower() for n in names]
    for m in re.finditer(r"\b[A-Z][\w/-]*(?:\s+(?:of|the|and|for)?\s*[A-Z][\w/-]*)*", text):
        phrase = m.group(0)
        if (len(phrase) >= 3 and phrase not in SIGS and not set(phrase.split()) <= _CAPITAL_WORDS
                and any(re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", n) for n in lowered)):
            variants.setdefault(phrase, next(n for n in names if phrase.lower() in n.lower()))
    rel_low = {r.lower() for r in relations}
    ans_low = {str(a).lower() for a in answers}

    def role(variant: str, name: str) -> str:
        low = {variant.lower(), name.lower()}
        if low & rel_low:
            return "the relation"
        if low & ans_low or any(variant.lower() in a for a in ans_low):
            return "the answer entity"
        for v in (variant, name):
            if re.search(r"\b" + _TEMPORAL_MARKER + r"\s+(?:[\w'-]+\s+){0,3}?" + re.escape(v), question, re.I):
                return "the reference entity"
        if any(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", question, re.I) for v in (variant, name)):
            return "the subject entity"
        return "another entity"

    marks = {}
    for variant in sorted(variants, key=len, reverse=True):
        pattern = re.compile(r"(?:\b(?:the|a|an)\s+)?(?<!\w)" + re.escape(variant) + r"(?!\w)", re.I)
        if pattern.search(text):
            key = f"\x00{len(marks)}\x00"
            marks[key] = role(variant, variants[variant])
            text = pattern.sub(key, text)
    gold_dates = {str(a) for a in answers if re.fullmatch(r"\d{4}(?:-\d{2}){0,2}", str(a))}
    text = _TEXT_DATE_RE.sub(lambda m: "the answer date" if _iso_prefix(m) in gold_dates else "the anchor date", text)
    for key, value in marks.items():
        text = text.replace(key, value)
    return text


# The operations a skill of each failure type is expected to name (one of them is enough).
FAILURE_TYPE_OPS = {
    "entity_grounding": {"FIND"}, "wrong_relation": {"FIND"}, "find_slot_direction": {"FIND"},
    "anchor_from_wrong_set": {"TIME", "FIND"}, "time_mode": {"TIME", "FIND"}, "time_shift": {"TIME", "FIND"},
    "filter_rel": {"FILTER", "PICK"}, "gran_mismatch": {"FILTER", "PICK"}, "over_filter": {"FILTER", "PICK"},
    "under_filter": {"FILTER", "PICK"}, "pick_wrong": {"PICK"},
    "get_field": {"GET"}, "get_time_mode": {"GET"}, "answer_type": {"GET"},
    "agg_wrong": {"AGG"}, "rank_args": {"RANK"},
    "premature_answer": {"ANSWER", "GET", "FILTER", "PICK"}, "wasted_loop": {"FIND", "PICK", "FILTER"},
}

_AVOIDANCE_RE = re.compile(r"\b(?:do not|don't|never|avoid|instead|rather than|not from|not on)\b", re.I)

# Operation grammar of what a skill recommends (operations outside a "do not ..." / "rather than ..." clause): a written-out
# call passes only arguments its operation takes (names only to FIND), and the prose after FILTER / PICK / AGG asks
# for nothing the temporal operations lack (FILTER narrows by time only, PICK by time order only, AGG counts, sums or averages).
_CALL_NOT_ARGS = {
    "FILTER": {"subj", "obj", "subject", "object", "entity", "relation", "rel", "type", "name", "on", "in", "at", "is",
               "first", "last", "earliest", "latest"},
    "PICK": {"subj", "obj", "subject", "object", "entity", "relation", "rel", "type", "name", "first", "last", "count"},
    "GET": {"rel", "relation", "date", "count", "entity", "earliest", "latest", "first", "last", "answer"},
    "TIME": {"earliest", "latest", "subj", "obj", "date"},
    "AGG": {"max", "min", "maximum", "minimum", "distinct", "unique", "first", "last"},
}
_FIELD = r"(?:entit(?:y|ies)|objects?|subjects?|subj|obj|relations?|names?)"
_PROSE_NOT = {
    "FILTER": re.compile(r"\b(?:entit(?:y|ies)|objects?|subjects?|subj|obj|relations?)\s+(?:types?|fields?|names?|literals?|strings?)\b"
                         r"|\b(?:match|matching|mentioning|containing|exclude|excluding|restrict\w*\s+to\s+those)\b[^;,]*?\b"
                         + _FIELD + r"\b(?!')|\b(?:country|person|people|organi[sz]ation)\s+entities\b"
                         r"|\bby\s+(?:the\s+)?" + _FIELD + r"\b(?!')", re.I),
    "PICK": re.compile(r"\b(?:most|least)\s+(?:frequent|common|often)\b"
                       r"|\bby\s+(?:the\s+)?(?:entity|name|relation|count|frequency|type)\b", re.I),
    "AGG": re.compile(r"\b(?:max(?:imum)?|min(?:imum)?|distinct|unique)\b", re.I),
}
_NEGATION_RE = re.compile(r"\b(?:do not|don't|never|avoid|rather than|not)\b", re.I)
_CLAUSE_END_RE = re.compile(r"[;.,]|\binstead\b|\bthen\b", re.I)


def operation_grammar_violations(skill: str) -> List[str]:
    """Recommended operations that break the operation grammar (see above), as "OP: argument" strings."""
    found = []
    for m in _OP_RE.finditer(skill):
        op, start = m.group(1), m.start()
        clause = skill[max([e.end() for e in _CLAUSE_END_RE.finditer(skill, 0, start)] or [0]):start]
        if _NEGATION_RE.search(clause):
            continue
        rest = skill[m.end():]
        if rest.lstrip().startswith("("):
            args = (_top_level_groups(rest) or [rest])[0]
            if op != "FIND" and _QUOTED_RE.search(args):
                found.append(f"{op}: quoted name")
            bad = {w.lower() for w in re.findall(r"[A-Za-z_]+", _QUOTED_RE.sub("", args))} & _CALL_NOT_ARGS.get(op, set())
            found += [f"{op}: {w}" for w in sorted(bad)]
            continue
        scope = rest[:min([x.start() for x in (_OP_RE.search(rest), re.search(r"[;.]|\binstead\b|\bthen\b", rest, re.I)) if x]
                          or [len(rest)])]
        pattern = _PROSE_NOT.get(op)
        hit = pattern.search(scope) if pattern else None
        if hit:
            found.append(f"{op}: {hit.group(0)}")
    return found


def validate_tkgqa_skill(skill: str, steps: Sequence[Dict[str, Any]], success: Optional[bool],
                         failure_step: Optional[int] = None, failure_type: Optional[str] = None,
                         max_words: int = 60) -> List[str]:
    """Violated rules of an episode-level skill (empty list = accepted)."""
    skill = str(skill or "").strip()
    if not skill:
        return ["missing_skill"]
    violations = []
    if len(skill.split()) > max_words:
        violations.append("too_long")
    if not re.match(r"^[\"'(\[]*when\b", skill, re.I):
        violations.append("not_when")
    if not _OP_RE.search(skill):
        violations.append("no_temporal_operation")
    elif not mentions_argument(skill):
        violations.append("no_operation_argument")
    if _NESTED_RE.search(skill) or _LITERAL_ASSIGN_RE.search(skill):
        violations.append("invalid_operation")
    if operation_grammar_violations(skill):
        violations.append("operation_grammar")
    if _EMPTY_ANSWER_RE.search(skill):
        violations.append("empty_answer")
    if success is False:
        if not _AVOIDANCE_RE.search(skill):
            violations.append("no_avoidance_form")
        expected = FAILURE_TYPE_OPS.get(str(failure_type or ""))
        if expected and _OP_RE.search(skill) and not any(re.search(r"\b" + op + r"\b", skill) for op in expected):
            violations.append("off_failure_type")  # names operations, none of those its failure type is about

    if any(_contains_name(skill, name) for name in episode_names(steps)):
        violations.append("episode_name")
    if _DATE_RE.search(skill):
        violations.append("date")
    if _VARIABLE_RE.search(skill):
        violations.append("variable_number")
    if _HINDSIGHT_TERM_RE.search(skill):
        violations.append("hindsight_term")
    if _PLACEHOLDER_RE.search(skill):
        violations.append("placeholder")
    if _STEP_NUMBER_RE.search(skill):
        violations.append("step_number")
    if success is False:
        # a failure's episode-level skill is an avoidance rule in natural language ("do not ...; instead ..."); the temporal
        # operation goes in the decision-level skill, so the rule may name no operation
        violations = [v for v in violations if v != "no_temporal_operation"]
    return violations


def names_failure_step_op(skill: str, steps: Sequence[Dict[str, Any]], failure_step: Optional[int]) -> Optional[bool]:
    """Diagnostic: does the skill name the operation of the step the analyzer blames (None if not applicable)."""
    if not isinstance(failure_step, int) or not 0 <= failure_step < len(steps):
        return None
    op = _step_op(steps[failure_step])
    return None if op is None else re.search(r"\b" + op + r"\b", str(skill)) is not None


# ---------------------------------------------------------------------------------------------------------------
# Decision-level skills: "At this decision, write <corrected> instead of <agent's>."
_PRESCRIPTION_RE = re.compile(r"^\s*At this decision, write (.+?) instead of (.+?)\.?\s*$", re.S)
_LHS_RE = re.compile(r"^\s*([fts]\d+)\s*=")
_KIND = {"FIND": "f", "FILTER": "f", "PICK": "f", "EXCLUDE": "f", "UNION": "f", "INTERSECT": "f", "TIME": "t",
         "GET": "s", "AGG": "s", "RANK": "s"}


def prescription_parts(text: str):
    """(corrected operation, the agent's operation as written) of a decision-level skill, or None."""
    m = _PRESCRIPTION_RE.match(str(text or "").strip())
    return (m.group(1).strip(), m.group(2).strip().rstrip(".")) if m else None


def fill_result_variable(corrected: str, own: str) -> str:
    """The corrected operation with the agent's result variable at that step when the decision-level skill left it out."""
    if _LHS_RE.match(corrected) or not _LHS_RE.match(own or ""):
        return corrected
    return f"{_LHS_RE.match(own).group(1)} = {corrected}"


# The context rule (seed/analysis.py CONTEXT_RULE): a decision-level skill may use only what the agent could see. Dates are
# compared as written (YYYY-MM-DD / YYYY-MM, a prefix of a seen date counts); the question's English dates are
# normalised first ("30 June 2011" -> 2011-06-30).
_CTX_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{4}-\d{2})\b")
_CTX_MONTHS = {m: i + 1 for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"))}
_CTX_MONTHS.update({m[:3]: i for m, i in list(_CTX_MONTHS.items())}); _CTX_MONTHS["sept"] = 9
_CTX_MON = "|".join(sorted(_CTX_MONTHS, key=len, reverse=True))
_GRAN_RANK = {"day": 0, "month": 1, "year": 2}


def question_times(question: str):
    """(dates the question names as YYYY-MM-DD / YYYY-MM strings, the finest granularity among its named times or None)."""
    s = str(question or "").lower()
    out, grans = set(_CTX_DATE_RE.findall(str(question or ""))), set()
    for d in out:
        grans.add("day" if len(d) == 10 else "month")
    for d, mo, y in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)? (?:of )?(" + _CTX_MON + r")\.?,? (\d{4})\b", s):
        out |= {f"{y}-{_CTX_MONTHS[mo]:02d}-{int(d):02d}", f"{y}-{_CTX_MONTHS[mo]:02d}"}; grans.add("day")
    for mo, d, y in re.findall(r"\b(" + _CTX_MON + r")\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b", s):
        out |= {f"{y}-{_CTX_MONTHS[mo]:02d}-{int(d):02d}", f"{y}-{_CTX_MONTHS[mo]:02d}"}; grans.add("day")
    for mo, y in re.findall(r"\b(" + _CTX_MON + r")\.?,? (?:of )?(\d{4})\b", s):
        out.add(f"{y}-{_CTX_MONTHS[mo]:02d}"); grans.add("month")
    if re.search(r"\b(1\d{3}|20\d{2})\b", s) and not grans:
        grans.add("year")
    return out, (min(grans, key=_GRAN_RANK.get) if grans else None)


def context_violations(corrected: str, question: str, seen_text: str) -> List[str]:
    """date_not_in_context: the corrected operation writes a date the agent has not seen (question / observations up to the
    decision); gran_finer_than_question: a FILTER on a literal time uses a finer gran than the time the question names."""
    out = []
    qdates, qgran = question_times(question)
    seen = str(question or "") + " " + " ".join(sorted(qdates)) + " " + str(seen_text or "")
    if any(d not in seen for d in _CTX_DATE_RE.findall(corrected)):
        out.append("date_not_in_context")
    m = re.match(r"^\s*\w+\s*=\s*FILTER\((.*)\)\s*$", corrected)
    g = re.search(r"gran=(\w+)", corrected)
    if m and g and qgran and g.group(1) in _GRAN_RANK and _GRAN_RANK[g.group(1)] < _GRAN_RANK[qgran]:
        args = [a.strip() for a in m.group(1).split(",")]
        if any(re.fullmatch(r"\d{4}(-\d{2}){0,2}", a) for a in args):
            out.append("gran_finer_than_question")
    return out


def validate_step_prescription(parsed: Dict[str, Any], steps: Sequence[Dict[str, Any]], success, answers,
                               check_relation: bool = True) -> List[str]:
    """Checks of a decision-level skill (failures only; the fixed form; the corrected operation parses and differs from
    the agent's own at failure_step; no gold answer; names only from the question or the trajectory's observations;
    at most 60 words; the context rule). Empty list = accepted. check_relation=False (the RL trainer): a FIND's relation
    slot is not name-checked - the grounder maps a paraphrase to a KG relation with bge-m3 and the executor pre-check
    rejects one that grounds to nothing; the entity slots stay checked (a name the student cannot have seen)."""
    from tkgqa.operations import OperationError, parse
    text, k = (parsed.get("step_skill") or "").strip(), parsed.get("failure_step")
    if success is not None and float(success) >= 1.0:
        return [] if not text else ["step_skill_on_success"]
    if not text:
        return ["no_step_skill"]
    out = []
    if k is None or not 0 <= k < len(steps):
        out.append("failure_step_out_of_range")
    if len(text.split()) > 60:
        out.append("too_long")
    parts = prescription_parts(text)
    if not parts:
        return out + ["form"]
    corrected = parts[0]
    try:
        parse(corrected)
    except OperationError:
        out.append("corrected_not_operation")
    if k is not None and 0 <= k < len(steps):
        own = re.search(r"<action>(.*?)</action>", str(steps[k].get("response", "")), re.S)
        if own and " ".join(own.group(1).split()) == " ".join(corrected.split()):
            out.append("same_as_agent")
    if any(str(a).strip() and str(a).strip().lower() in text.lower() for a in (answers or [])):
        out.append("gold_answer")
    context = " ".join(str(s.get("observation", "")) for s in steps) + " " + str(steps[-1].get("final_observation", "") if steps else "")
    context = " ".join(context.lower().split())
    names = re.findall(r'"([^"]*)"', corrected)
    find = re.match(r'^\s*\w+\s*=\s*FIND\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)\s*$', corrected)
    if find and not check_relation:
        names = [find.group(1), find.group(3)]
    names = [x for x in names if x.strip() and x.strip() != "?"]
    if any(" ".join(x.lower().split()) not in context for x in names):
        out.append("name_not_in_context")
    if steps:
        question = str(steps[0].get("task_description") or steps[0].get("observation") or "")
        upto = k if (k is not None and 0 <= k < len(steps)) else len(steps) - 1
        out += context_violations(corrected, question, " ".join(str(steps[j].get("observation", "")) for j in range(upto + 1)))
    return out


def run_prescription(kg, grounder, question: str, actions: Sequence[str], k: int, corrected: str) -> str:
    """Executor status of the corrected operation at step k after the agent's own steps 0..k-1 (ok, invalid,
    find_failed, ...)."""
    from tkgqa.executor import ExecutorConfig, Session
    s = Session(kg, grounder, question, config=ExecutorConfig(max_steps=k + 2))
    for a in actions[:k]:
        s.step(f"<action> {a or ''} </action>")
    s.step(f"<action> {corrected} </action>")
    return s.log[-1].status
