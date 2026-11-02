"""Symbolic executor of the temporal operations: operations on the TKG, per-question session state (the execution
record: every intermediate variable with its full value and the variables it was computed from), observations.

Conventions (dataset-independent):
- duration of a fact = end - start in days (a single date has duration 0);
- AGG average = floor(total duration / number of facts);
- RANK uses competition ranking: rank = 1 + #facts strictly before (ties share the lowest rank);
- GET(..., merged) merges spans that overlap or touch at ``gran``;
- PICK K counts distinct dates (dense), so ties stay together;
- FIND grounds each entity to its top candidate only (never a lower-ranked
  entity); the relation may fall back to its next candidates when the top one
  has no facts with these entities, and the observation says so. A non-exact
  relation also lists up to two other relations that occur with the entities.
  If nothing matches, the step reports "FIND failed: ..." with candidates for
  the slot that blocks the match and assigns no variable, so the agent can
  retry under the same name.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tkgqa.operations import ANSWER_HINT, RET, TYPE_NAMES, VAR_RE, Arg, OperationError, Statement, admissible_ops, parse
from tkgqa.grounder import Grounder
from tkgqa.kg import TemporalKG
from tkgqa.timeutil import GRANS, format_date, format_span, gran_key, merge_spans, parse_date_literal, shift_ordinal

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)
INITIAL_OBSERVATION = "Nothing has been executed yet."
# Bump when the observation format changes; it is part of prompt_version().
OBSERVATION_VERSION = "3"
FILTER_RELS = ("before", "after", "equal", "overlap", "during", "contains", "between")
_INT_RE = re.compile(r"^[+-]?\d+$")


@dataclass
class Var:
    name: str
    type: str  # 'f' fact set, 't' time, 's' values
    value: Any  # f: sorted fact ids (np.ndarray), t: [(start, end)], s: list
    op: str
    args: tuple
    parents: List[str]
    step: int
    info: Dict[str, Any] = field(default_factory=dict)
    action: str = ""  # canonical text of the defining operation


@dataclass
class Step:
    response: str
    valid: bool  # parsed and type-correct (a failed FIND is still valid)
    obs: str
    status: str = "ok"  # "ok" | "invalid" | "find_failed"
    var: Optional[str] = None
    op: Optional[str] = None
    action: Optional[str] = None  # text inside <action> as written
    error: Optional[str] = None


@dataclass
class ExecutorConfig:
    max_steps: int = 10
    sample_facts: int = 2  # facts shown in a fact-set observation (earliest + latest)
    sample_values: int = 5  # values shown in a value-set observation
    preview_spans: int = 5  # spans shown in a time observation
    find_candidates: int = 5  # candidates listed when FIND fails
    relation_alternatives: int = 2  # other relations shown after a non-exact relation match
    # Optional per-kind minimum similarity for non-exact grounding, e.g.
    # {"relation": 0.6}; below it the slot counts as unmatched.
    min_ground_score: Optional[Dict[str, float]] = None


class FindFailed(Exception):
    """FIND matched no facts; the message is the observation."""


# ---------------------------------------------------------------------- #
# Argument reading
# ---------------------------------------------------------------------- #


class _Args:
    def __init__(self, session: "Session", stmt: Statement):
        self.session = session
        self.stmt = stmt
        self.op = stmt.op
        self._used_kwargs = set()

    def _pos(self, i: int, what: str) -> Arg:
        if i >= len(self.stmt.args):
            raise OperationError(f"{self.op}: missing argument {i + 1} ({what})")
        return self.stmt.args[i]

    def var(self, i: int, vtype: str) -> Var:
        arg = self._pos(i, f"{TYPE_NAMES[vtype]} {vtype}N")
        if arg.kind != "var":
            raise OperationError(f"{self.op}: argument {i + 1} must be a {TYPE_NAMES[vtype]} variable ({vtype}N), got {arg.show()}")
        var = self.session.vars.get(arg.text)
        if var is None:
            raise OperationError(f"{arg.text} is not defined")
        if var.type != vtype:
            raise OperationError(
                f"{self.op}: argument {i + 1} must be a {TYPE_NAMES[vtype]}, but {arg.text} is a {TYPE_NAMES[var.type]}"
            )
        return var

    def word(self, i: int, choices: Tuple[str, ...], optional: bool = False) -> Optional[str]:
        if optional and i >= len(self.stmt.args):
            return None
        arg = self._pos(i, "|".join(choices))
        if arg.kind != "word" or arg.text not in choices:
            raise OperationError(f"{self.op}: argument {i + 1} must be {'|'.join(choices)}, got {arg.show()}")
        return arg.text

    def string(self, i: int, what: str) -> str:
        arg = self._pos(i, what)
        if arg.kind != "str":
            raise OperationError(f'{self.op}: argument {i + 1} ({what}) must be a quoted name, got {arg.show()}')
        return arg.text

    def time(self, i: int) -> List[Tuple[int, int]]:
        arg = self._pos(i, "time tN or date literal")
        if arg.kind == "var":
            return self.var(i, "t").value
        if arg.kind in ("num", "str"):
            try:
                return [parse_date_literal(arg.text)]
            except ValueError as exc:
                raise OperationError(f"{self.op}: {exc}") from None
        raise OperationError(f"{self.op}: argument {i + 1} must be a time variable (tN) or a date literal, got {arg.show()}")

    def kw_word(self, key: str, choices: Tuple[str, ...], required: bool = False, default: Optional[str] = None):
        arg = self.stmt.kwargs.get(key)
        if arg is None:
            if required:
                raise OperationError(f"{self.op}: missing {key}={'|'.join(choices)}")
            return default
        self._used_kwargs.add(key)
        if arg.kind != "word" or arg.text not in choices:
            raise OperationError(f"{self.op}: {key} must be {'|'.join(choices)}, got {arg.show()}")
        return arg.text

    def kw_int(self, key: str) -> Optional[int]:
        arg = self.stmt.kwargs.get(key)
        if arg is None:
            return None
        self._used_kwargs.add(key)
        if arg.kind != "num" or not _INT_RE.match(arg.text):
            raise OperationError(f"{self.op}: {key} must be an integer such as +3 or -1, got {arg.show()}")
        return int(arg.text)

    def finish(self, max_positional: int) -> None:
        if len(self.stmt.args) > max_positional:
            raise OperationError(f"{self.op}: too many arguments ({len(self.stmt.args)} given, at most {max_positional})")
        unknown = sorted(set(self.stmt.kwargs) - self._used_kwargs)
        if unknown:
            raise OperationError(f"{self.op}: unknown keyword {unknown[0]!r}")


# ---------------------------------------------------------------------- #
# Operations: each returns (value, info)
# ---------------------------------------------------------------------- #

_EMPTY = np.empty(0, dtype=np.int64)


_SLOT_KINDS = ("entity", "relation", "entity")


def op_find(session: "Session", a: _Args):
    texts = [a.string(i, slot) for i, slot in enumerate(("subj", "rel", "obj"))]
    a.finish(3)
    wild = [t.strip() == "?" for t in texts]
    if sum(wild) > 1:
        raise OperationError(
            "FIND: at most one slot may be ?; name two of subject, relation and object "
            "(after a failed FIND, take a name from the listed candidates)"
        )
    cfg = session.config
    slot_candidates, unmatched = [], []
    for slot, (text, is_wild, kind) in enumerate(zip(texts, wild, _SLOT_KINDS)):
        if is_wild:
            slot_candidates.append([None])
            continue
        if not text.strip():
            raise OperationError('FIND: empty name; use "?" for the unknown slot')
        candidates = session.grounder.candidates(text, kind)
        min_score = (cfg.min_ground_score or {}).get(kind)
        if not candidates or (
            min_score is not None and not candidates[0].exact and candidates[0].score < min_score
        ):
            unmatched.append(slot)
        slot_candidates.append(candidates)
    if unmatched:
        raise FindFailed(_unmatched_slot(session, texts, wild, slot_candidates, unmatched[0], unmatched))

    # Entities: the top candidate only. Relation: candidates in score order.
    subj, rel_candidates, obj = slot_candidates[0][0], slot_candidates[1], slot_candidates[2][0]
    for relation in rel_candidates:  # [None] when the relation slot is "?"
        ids = tuple(None if c is None else c.id for c in (subj, relation, obj))
        found = session.kg.find(*ids)
        if not len(found):
            continue
        combo = (subj, relation, obj)
        info = {
            "grounding": [(text, c) for text, c in zip(texts, combo) if c is not None],
            "grounded": tuple("?" if c is None else c.text for c in combo),
            "ids": ids,
        }
        if relation is not None and relation is not rel_candidates[0]:
            info["relation_fallback"] = rel_candidates[0].text
        if relation is not None and not relation.exact:
            info["other_relations"] = _other_relations(session, texts[1], ids, relation.id)
        return found, info
    top = [subj, rel_candidates[0], obj]
    raise FindFailed(_diagnose_find(session, texts, wild, [[c] for c in top]))


def _other_relations(session: "Session", text: str, ids, chosen: int) -> List[str]:
    """Relations that occur with the grounded entities, most similar to ``text`` first."""
    k = session.config.relation_alternatives
    if k <= 0:
        return []
    kg = session.kg
    found = kg.find(ids[0], None, ids[2])
    values = [r for r in np.unique(kg.rel[found]).tolist() if r != chosen]
    return [kg.relation_texts[r] for r in session.grounder.rank(text, "relation", values, k=k)]


def _no_match(kind: str, text: str, candidates: List[str]) -> str:
    message = f'FIND failed: no {kind} match for "{text}"'
    return message + ("; candidates: " + ", ".join(candidates) if candidates else "")


def _cooccurring(session: "Session", texts, slot: int, others: List[Optional[int]]) -> Optional[str]:
    """Candidates for ``slot``: values that occur with the other bound slots, best first."""
    if all(o is None for o in others):
        return None
    ids = session.kg.find(*others)
    if not len(ids):
        return None
    kg = session.kg
    values = np.unique((kg.subj, kg.rel, kg.obj)[slot][ids])
    kind = _SLOT_KINDS[slot]
    ranked = session.grounder.rank(texts[slot], kind, values, k=session.config.find_candidates)
    names = kg.relation_texts if kind == "relation" else kg.entity_texts
    return _no_match(kind, texts[slot], [names[i] for i in ranked])


def _unmatched_slot(session: "Session", texts, wild, slot_candidates, slot: int, unmatched: List[int]) -> str:
    others = [
        None if (j == slot or wild[j] or j in unmatched) else slot_candidates[j][0].id
        for j in range(3)
    ]
    message = _cooccurring(session, texts, slot, others)
    if message is not None:
        return message
    fallback = [c.text for c in slot_candidates[slot][: session.config.find_candidates]]
    return _no_match(_SLOT_KINDS[slot], texts[slot], fallback)


def _diagnose_find(session: "Session", texts, wild, slot_candidates) -> str:
    """Blame one slot: relax it and list the values that do co-occur with the others."""
    kg, cfg = session.kg, session.config
    top = [cands[0] for cands in slot_candidates]
    # The least confident grounding is the likelier culprit (exact matches last; ties: relation first).
    order = sorted(
        (i for i in range(3) if not wild[i]),
        key=lambda i: (top[i].exact, top[i].score, {1: 0, 2: 1, 0: 2}[i]),
    )
    for slot in order:
        others = [None if (j == slot or top[j] is None) else top[j].id for j in range(3)]
        message = _cooccurring(session, texts, slot, others)
        if message is not None:
            return message
    shown = ", ".join("?" if c is None else c.text for c in top)
    return f"FIND failed: no facts match ({shown})"


def op_time(session: "Session", a: _Args):
    f = a.var(0, "f")
    mode = a.word(1, ("first", "last", "all"))
    side = a.word(2, ("start", "end"), optional=True)
    shift = a.kw_int("shift")
    unit = a.kw_word("unit", GRANS)
    if (shift is None) != (unit is None):
        raise OperationError("TIME: shift and unit must be given together, e.g. shift=-1, unit=year")
    a.finish(3)
    ids = f.value
    if not len(ids):
        return [], {"empty_source": f.name}
    kg = session.kg
    starts, ends = kg.start[ids], kg.end[ids]
    if mode == "first":
        sel = starts == starts.min()
    elif mode == "last":
        sel = starts == starts.max()
    else:
        sel = np.ones(len(ids), dtype=bool)
    spans = sorted(set(zip(starts[sel].tolist(), ends[sel].tolist())))
    if side == "start":
        spans = sorted({(s, s) for s, _ in spans})
    elif side == "end":
        spans = sorted({(e, e) for _, e in spans})
    if shift:
        try:
            spans = sorted({(shift_ordinal(s, shift, unit), shift_ordinal(e, shift, unit)) for s, e in spans})
        except ValueError as exc:
            raise OperationError(f"TIME: {exc}") from None
    return spans, {}


def op_filter(session: "Session", a: _Args):
    f = a.var(0, "f")
    rel = a.word(1, FILTER_RELS)
    if rel == "between":
        t_from, t_to, n_pos = a.time(2), a.time(3), 4
    else:
        t, n_pos = a.time(2), 3
    gran = a.kw_word("gran", GRANS, required=True)
    a.finish(n_pos)
    ids = f.value
    fs = session.kg.gran_keys("start", gran)[ids]
    fe = session.kg.gran_keys("end", gran)[ids]
    if rel == "between":
        if not t_from or not t_to:
            return _EMPTY, {}
        lo = min(gran_key(s, gran) for s, _ in t_from)
        hi = max(gran_key(e, gran) for _, e in t_to)
        return ids[(fs >= lo) & (fe <= hi)], {}
    mask = np.zeros(len(ids), dtype=bool)
    for start, end in t:
        ts, te = gran_key(start, gran), gran_key(end, gran)
        if rel == "before":
            mask |= fe < ts
        elif rel == "after":
            mask |= fs > te
        elif rel == "equal":
            mask |= (fs == ts) & (fe == te)
        elif rel == "overlap":
            mask |= (fs <= te) & (fe >= ts)
        elif rel == "during":
            mask |= (fs >= ts) & (fe <= te)
        else:  # contains
            mask |= (fs <= ts) & (fe >= te)
    return ids[mask], {}


def op_pick(session: "Session", a: _Args):
    f = a.var(0, "f")
    arg = a._pos(1, "earliest|latest|K")
    if arg.kind == "word" and arg.text in ("earliest", "latest"):
        which = arg.text
    elif arg.kind == "num" and _INT_RE.match(arg.text):
        which = int(arg.text)
        if which == 0:
            raise OperationError("PICK: K must be non-zero (1 = earliest, -1 = latest)")
    else:
        raise OperationError(f"PICK: argument 2 must be earliest|latest|K, got {arg.show()}")
    side = a.word(2, ("start", "end"), optional=True) or "start"
    a.finish(3)
    ids = f.value
    if not len(ids):
        return _EMPTY, {}
    keys = (session.kg.start if side == "start" else session.kg.end)[ids]
    distinct = np.unique(keys)
    if which == "earliest":
        key = distinct[0]
    elif which == "latest":
        key = distinct[-1]
    elif abs(which) > len(distinct):
        return _EMPTY, {}
    else:
        key = distinct[which - 1] if which > 0 else distinct[which]
    return ids[keys == key], {}


def op_exclude(session: "Session", a: _Args):
    f, g = a.var(0, "f"), a.var(1, "f")
    a.finish(2)
    return np.setdiff1d(f.value, g.value, assume_unique=True), {}


def op_intersect(session: "Session", a: _Args):
    f, g = a.var(0, "f"), a.var(1, "f")
    a.finish(2)
    return np.intersect1d(f.value, g.value, assume_unique=True), {}


def op_union(session: "Session", a: _Args):
    n = len(a.stmt.args)
    if n < 2:
        raise OperationError("UNION: needs at least two fact sets")
    values = [a.var(i, "f").value for i in range(n)]
    a.finish(n)
    result = values[0]
    for value in values[1:]:
        result = np.union1d(result, value)
    return result.astype(np.int64), {}


def op_get(session: "Session", a: _Args):
    f = a.var(0, "f")
    field_name = a.word(1, ("subj", "obj", "duration", "time"))
    kg, ids = session.kg, f.value
    if field_name == "time":
        mode = a.word(2, ("start", "end", "span", "merged"), optional=True) or "start"
        gran = a.kw_word("gran", GRANS, required=True)
        a.finish(3)
        starts, ends = kg.start[ids].tolist(), kg.end[ids].tolist()
        if mode in ("start", "end"):
            dates = sorted(set(starts if mode == "start" else ends))
            values = [format_date(d, gran) for d in dates]
        else:
            spans = sorted(set(zip(starts, ends)))
            if mode == "merged":
                spans = merge_spans(spans, gran)
            values = [format_span(span, gran) for span in spans]
        return list(dict.fromkeys(values)), {}
    if "gran" in a.stmt.kwargs:
        raise OperationError("GET: gran only applies to GET(f, time, gran=...)")
    a.finish(2)
    if field_name == "duration":
        return sorted(set((kg.end[ids] - kg.start[ids]).tolist())), {"unit": "days"}
    entity_ids = kg.subj[ids] if field_name == "subj" else kg.obj[ids]
    return sorted({kg.entity_texts[i] for i in entity_ids.tolist()}), {}


def op_agg(session: "Session", a: _Args):
    f = a.var(0, "f")
    kind = a.word(1, ("count", "sum", "average"))
    a.finish(2)
    ids = f.value
    if kind == "count":
        return [int(len(ids))], {}
    durations = session.kg.end[ids] - session.kg.start[ids]
    if kind == "sum":
        return [int(durations.sum())], {"unit": "days"}
    if not len(ids):
        return [], {"unit": "days", "empty_source": f.name}
    return [int(durations.sum() // len(ids))], {"unit": "days"}


def op_rank(session: "Session", a: _Args):
    f, g = a.var(0, "f"), a.var(1, "f")
    by = a.kw_word("by", ("start", "end", "duration"), default="start")
    order = a.kw_word("order", ("asc", "desc"), default="asc")
    a.finish(2)
    kg = session.kg

    def keys(ids):
        if by == "start":
            return kg.start[ids]
        if by == "end":
            return kg.end[ids]
        return kg.end[ids] - kg.start[ids]

    pool = np.sort(keys(f.value))
    targets = keys(g.value)
    if order == "asc":
        ranks = 1 + np.searchsorted(pool, targets, side="left")
    else:
        ranks = 1 + (len(pool) - np.searchsorted(pool, targets, side="right"))
    return sorted(set(ranks.tolist())), {}


def op_answer(session: "Session", a: _Args):
    arg = a._pos(0, "value set sN")
    if arg.kind != "var":
        raise OperationError(ANSWER_HINT)
    s = a.var(0, "s")
    a.finish(1)
    return list(s.value), {"answer_var": s.name}


OPS = {
    "FIND": op_find,
    "TIME": op_time,
    "FILTER": op_filter,
    "PICK": op_pick,
    "EXCLUDE": op_exclude,
    "UNION": op_union,
    "INTERSECT": op_intersect,
    "GET": op_get,
    "AGG": op_agg,
    "RANK": op_rank,
    "ANSWER": op_answer,
}


# ---------------------------------------------------------------------- #
# Observation formatting
# ---------------------------------------------------------------------- #


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _format_list(values: List[str], limit: int) -> str:
    if len(values) <= limit:
        return "values: " + ", ".join(values)
    return "sample: " + ", ".join(values[:limit]) + ", ..."


def format_fact_set(session: "Session", name: str, ids, info: Optional[Dict[str, Any]] = None) -> str:
    """``f1 = 212 facts | grounded: (?, r, o) [| relation fallback | other relations] | span A..B | sample: ...``"""
    info = info or {}
    grounded = info.get("grounded")
    kg, k = session.kg, session.config.sample_facts
    n = len(ids)
    parts = [f"{name} = {_plural(n, 'fact')}"]
    if grounded:
        parts.append("grounded: (" + ", ".join(grounded) + ")")
    if info.get("relation_fallback"):
        parts.append(f'relation fallback: "{info["relation_fallback"]}" has no such facts')
    if info.get("other_relations"):
        parts.append("other relations: " + ", ".join(info["other_relations"]))
    if n:
        span = (int(kg.start[ids].min()), int(kg.end[ids].max()))
        parts.append("span " + format_span(span, kg.date_precision, collapse=True))
        if n <= k:
            parts.append("facts: " + "; ".join(kg.format_fact(i) for i in ids))
        else:
            head, tail = (k + 1) // 2, k // 2
            chosen = list(ids[:head]) + list(ids[n - tail :])
            parts.append("sample: " + "; ".join(kg.format_fact(i) for i in chosen))
    return " | ".join(parts)


def format_obs(session: "Session", stmt: Statement, value: Any, info: Dict[str, Any]) -> str:
    kg, cfg = session.kg, session.config
    unit = " days" if info.get("unit") == "days" else ""
    if stmt.op == "ANSWER":
        return "Answer submitted: " + (", ".join(f"{v}{unit}" for v in value) if value else "(empty)")
    name, vtype = stmt.name, RET[stmt.op]
    if vtype == "f":
        return format_fact_set(session, name, value, info)
    if vtype == "t":
        if not value:
            return f"{name} = none ({info.get('empty_source', 'source')} is empty)"
        spans = [format_span(span, kg.date_precision, collapse=True) for span in value]
        if len(spans) > cfg.preview_spans:
            spans = spans[: cfg.preview_spans] + [f"... ({len(value) - cfg.preview_spans} more)"]
        return f"{name} = " + ", ".join(spans)
    head = f"{name} = {_plural(len(value), 'value')}"
    if not value:
        return head
    return head + " | " + _format_list([f"{v}{unit}" for v in value], cfg.sample_values)


# ---------------------------------------------------------------------- #
# Session
# ---------------------------------------------------------------------- #


class Session:
    """State of one question: variables, step log, and each variable's parents."""

    def __init__(
        self,
        kg: TemporalKG,
        grounder: Grounder,
        question: str,
        max_steps: Optional[int] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        self.kg, self.grounder, self.question = kg, grounder, question
        self.config = config or ExecutorConfig()
        self.max_steps = int(max_steps if max_steps is not None else self.config.max_steps)
        self.vars: Dict[str, Var] = {}
        self.log: List[Step] = []
        self.done = False
        self.truncated = False
        self.answer: Optional[List[Any]] = None
        self.answer_var: Optional[str] = None

    def step(self, response: str) -> str:
        if self.done:
            raise RuntimeError("session is finished")
        match = ACTION_RE.search(response)
        action = match.group(1).strip() if match else None
        try:
            if match is None:
                raise OperationError("missing <action>...</action>")
            stmt = parse(match.group(1))
            self._check(stmt)
            value, info = OPS[stmt.op](self, _Args(self, stmt))
        except OperationError as exc:
            obs = f"Invalid action: {exc}"
            self.log.append(Step(response, valid=False, obs=obs, status="invalid", action=action, error=str(exc)))
            return self._end_step(obs)
        except FindFailed as exc:
            obs = str(exc)
            self.log.append(Step(response, valid=True, obs=obs, status="find_failed", op="FIND", action=action))
            return self._end_step(obs)
        if stmt.op == "ANSWER":
            self.done, self.answer, self.answer_var = True, value, info["answer_var"]
        else:
            self.vars[stmt.name] = Var(
                name=stmt.name,
                type=RET[stmt.op],
                value=value,
                op=stmt.op,
                args=tuple(stmt.args) + tuple(stmt.kwargs.items()),
                parents=list(dict.fromkeys(stmt.var_refs())),
                step=len(self.log),
                info=info,
                action=stmt.show(),
            )
        obs = format_obs(self, stmt, value, info)
        self.log.append(Step(response, valid=True, obs=obs, var=stmt.name, op=stmt.op, action=action))
        return self._end_step(obs)

    def _end_step(self, obs: str) -> str:
        if not self.done and len(self.log) >= self.max_steps:
            self.done = self.truncated = True
            obs += "\nStep limit reached; the episode ends without an answer."
            self.log[-1].obs = obs
        return obs

    def _check(self, stmt: Statement) -> None:
        ret = RET[stmt.op]
        if ret is None:
            if stmt.name:
                raise OperationError("ANSWER does not assign a variable; write ANSWER(sN)")
        else:
            if not stmt.name:
                raise OperationError(f"assign the result to a new variable, e.g. {ret}1 = {stmt.op}(...)")
            match = VAR_RE.match(stmt.name)
            if match is None:
                raise OperationError(f"bad variable name {stmt.name!r}; use {ret}N")
            if match.group(1) != ret:
                raise OperationError(f"{stmt.op} returns a {TYPE_NAMES[ret]}; name it {ret}N, not {stmt.name}")
            if stmt.name in self.vars:
                raise OperationError(f"{stmt.name} is already defined; use a new name")
        for ref in stmt.var_refs():
            if ref not in self.vars:
                raise OperationError(f"{ref} is not defined")

    @property
    def current_observation(self) -> str:
        return self.log[-1].obs if self.log else INITIAL_OBSERVATION

    def admissible(self) -> List[str]:
        return admissible_ops([v.type for v in self.vars.values()])

    def ancestors(self, name: str) -> List[str]:
        """``name`` and all variables it was derived from, in step order."""
        seen, stack = set(), [name]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.vars:
                continue
            seen.add(current)
            stack.extend(self.vars[current].parents)
        return sorted(seen, key=lambda n: self.vars[n].step)
