"""Temporal operations: their documents, tokenizer / parser, and signatures.

The actor solves a question as a sequence of temporal operations, one per step; one step (action) is ``name = OP(args)``
(``ANSWER(sN)`` assigns nothing), and its result is stored in an intermediate variable for later operations. Arguments
are double-quoted strings, variables (fN / tN / sN), bare words, integers, date literals (``2005``, ``2005-03``,
``2005-03-04``, ``2005..2007``) or ``key=value`` keyword arguments; positional and keyword arguments may be mixed and the
positional order is kept. The document strings below are prompt text and call the operations "DSL"; they are kept
verbatim because the released checkpoints were trained with them.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The full document (used by the skill checks for the argument words).
OPERATIONS_DOC = """[DSL]
Variables: fN = fact set, tN = time, sN = value set.
A fact is (subj, rel, obj, start, end); a single date has start = end.
A time is a span [start, end]. Date literals: YYYY, YYYY-MM, YYYY-MM-DD; A..B is a span.
Each step runs one operation. Arguments are variables, quoted names, keywords or literals;
operations cannot be nested: assign an inner operation to a variable first.

FIND("subj", "rel", "obj") -> f
  Matching facts. Write names in natural language; at most one slot may be ?.
  e.g. f1 = FIND("?", "pay a visit to", "Velora")   f2 = FIND("Taro Senn", "pay a visit to", "Kestria")
TIME(f, first|last|all [, start|end] [, shift=±N, unit=day|month|year]) -> t
  Span of the first/last fact (ordered by start date), or the spans of all facts.
  start/end keeps one end. shift moves the result by N units; give shift and unit together.
  e.g. t1 = TIME(f2, first, start)
FILTER(f, REL, t, gran=day|month|year) -> f
  Keep facts whose span satisfies REL against t, compared at gran:
    before   = fact ends before t starts     after    = fact starts after t ends
    equal    = same span                     overlap  = shares any date with t
    during   = fact lies inside t            contains = fact covers all of t
  If t holds several spans, a fact is kept when REL holds for any of them.
  e.g. f3 = FILTER(f1, after, t1, gran=day)
FILTER(f, between, tA, tB, gran=day|month|year) -> f
  Keep facts inside [start of tA, end of tB], both ends included.
PICK(f, earliest|latest|K [, start|end]) -> f
  Earliest, latest, or K-th earliest date (K < 0 counts from the latest), by start unless end
  is given. All facts on the chosen date are kept.
EXCLUDE(f, f2) -> f     UNION(f, f2, ...) -> f     INTERSECT(f, f2) -> f
GET(f, subj|obj|duration) -> s          entities, or durations in days
GET(f, time, gran=day|month|year [, start|end|span|merged]) -> s
                                        start dates by default; span = A..B per fact;
                                        merged joins spans that overlap or touch
AGG(f, count|sum|average) -> s          number of facts, or total/average duration in days
RANK(f, f2 [, by=start|end|duration] [, order=asc|desc]) -> s   position of f2's fact within f
ANSWER(s)                               finish with a set from GET, AGG or RANK

Convention: get a reference time with TIME; narrow candidate facts with FILTER and PICK.
When the question asks for the first/last/earliest/latest one, select it explicitly, even if
only one fact remains: PICK for the facts you answer about, TIME(f, first|last) for a reference event."""

# Short form of OPERATIONS_DOC, the document in the actor prompt ({operations_spec} in agent_system/environments/prompts/tkgqa.py):
# the same operations, arguments and comparison rules, without the examples and the longer explanations.
OPERATIONS_DOC_COMPACT = """[DSL] One operation per step, no nesting. fN = facts (subj, rel, obj, start, end), tN = time spans,
sN = values. Dates: YYYY, YYYY-MM, YYYY-MM-DD; A..B is a span.
FIND("subj", "rel", "obj") -> f        names in natural language; at most one "?"
TIME(f, first|last|all [, start|end] [, shift=±N, unit=day|month|year]) -> t   (first/last by start date)
FILTER(f, REL, t, gran=day|month|year) -> f   REL: before (ends before t starts), after (starts after t ends),
  equal (same span), overlap (shares a date), during (inside t), contains (covers t); any span of t counts
FILTER(f, between, tA, tB, gran=...) -> f     inside [start of tA, end of tB]
PICK(f, earliest|latest|K [, start|end]) -> f  K-th earliest, K < 0 from the latest; keeps facts on that date
EXCLUDE(f, f2) / UNION(f, f2, ...) / INTERSECT(f, f2) -> f
GET(f, subj|obj|duration) -> s         GET(f, time, gran=... [, start|end|span|merged]) -> s
AGG(f, count|sum|average) -> s         RANK(f, f2 [, by=start|end|duration] [, order=asc|desc]) -> s
ANSWER(s)                              s from GET, AGG or RANK
Use TIME for a reference time; narrow facts with FILTER and PICK. For first/last/earliest/latest questions
select explicitly: PICK for the answer facts, TIME(f, first|last) for a reference event."""

# One-line summary of the temporal operations for the analyzer prompt (seed/analysis.py).
OPERATIONS_LINE = ('FIND("subj","rel","obj") one slot may be ? | TIME(f, first|last|all [,start|end] [,shift=N,unit=]) | '
            'FILTER(f, before|after|equal|overlap|during|contains, t, gran=day|month|year) | FILTER(f, between, tA, tB, gran=) | '
            'PICK(f, earliest|latest|K [,start|end]) | EXCLUDE/UNION/INTERSECT | GET(f, subj|obj|duration) | '
            'GET(f, time, gran= [,start|end|span|merged]) | AGG(f, count|sum|average) | RANK(f, f2 [,by=][,order=]) | ANSWER(s)')

# Required variable arguments per op (types only). Used for type checks and
# for the admissible-operation set.
SIGS: Dict[str, List[str]] = {
    "FIND": [],
    "TIME": ["f"],
    "FILTER": ["f"],
    "PICK": ["f"],
    "EXCLUDE": ["f", "f"],
    "UNION": ["f", "f"],
    "INTERSECT": ["f", "f"],
    "GET": ["f"],
    "AGG": ["f"],
    "RANK": ["f", "f"],
    "ANSWER": ["s"],
}
# Type of the variable each op assigns (ANSWER assigns nothing).
RET: Dict[str, Optional[str]] = {
    "FIND": "f",
    "TIME": "t",
    "FILTER": "f",
    "PICK": "f",
    "EXCLUDE": "f",
    "UNION": "f",
    "INTERSECT": "f",
    "GET": "s",
    "AGG": "s",
    "RANK": "s",
    "ANSWER": None,
}
TYPE_NAMES = {"f": "fact set", "t": "time", "s": "value set"}
VAR_RE = re.compile(r"^([fts])(\d+)$")


_NESTED = "Nested operations are not allowed; assign the inner operation to a variable first."
ANSWER_HINT = (
    "ANSWER takes a value-set variable such as s1, made by GET, AGG or RANK. If a result came out "
    "empty, revisit the FIND or FILTER that produced it instead of answering."
)
_LITERAL_ASSIGN = (
    "Literals cannot be assigned to variables; write a date directly as a FILTER argument, "
    "or get a time from facts with TIME."
)


class OperationError(Exception):
    """An invalid action; the message is shown to the agent as the observation."""


@dataclass(frozen=True)
class Arg:
    kind: str  # "str" | "var" | "word" | "num"
    text: str  # raw token text (for "str": the unquoted value)

    def show(self) -> str:
        return f'"{self.text}"' if self.kind == "str" else self.text


@dataclass
class Statement:
    name: Optional[str]
    op: str
    args: List[Arg] = field(default_factory=list)
    kwargs: Dict[str, Arg] = field(default_factory=dict)

    def var_refs(self) -> List[str]:
        refs = [a.text for a in self.args if a.kind == "var"]
        refs += [a.text for a in self.kwargs.values() if a.kind == "var"]
        return refs

    def show(self) -> str:
        parts = [a.show() for a in self.args] + [f"{k}={v.show()}" for k, v in self.kwargs.items()]
        call = f"{self.op}({', '.join(parts)})"
        return f"{self.name} = {call}" if self.name else call


_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<str>"(?:[^"\\]|\\.)*")
      | (?P<num>[+-]?\d+(?:-\d{1,2}){0,2}(?:\.\.\d+(?:-\d{1,2}){0,2})?)
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<punct>[=(),])
    )""",
    re.VERBOSE,
)


def tokenize(src: str) -> List[tuple]:
    tokens, pos, src = [], 0, src.strip()
    while pos < len(src):
        match = _TOKEN_RE.match(src, pos)
        if match is None or match.end() == pos:
            snippet = src[pos : pos + 15]
            hint = " (use double quotes for names)" if src[pos:].lstrip().startswith("'") else ""
            raise OperationError(f"cannot parse near {snippet!r}{hint}")
        kind = match.lastgroup
        text = match.group(kind)
        if kind == "str":
            text = re.sub(r"\\(.)", r"\1", text[1:-1])
        tokens.append((kind, text))
        pos = match.end()
        while pos < len(src) and src[pos].isspace():
            pos += 1
    return tokens


def parse(src: str) -> Statement:
    if re.match(r"\s*ANSWER\s*\(\s*[\[\"']", src):
        raise OperationError(ANSWER_HINT)
    tokens = tokenize(src)
    if not tokens:
        raise OperationError("empty action")
    i = 0

    def peek(offset: int = 0):
        j = i + offset
        return tokens[j] if j < len(tokens) else (None, None)

    name = None
    if peek()[0] == "ident" and peek(1) == ("punct", "="):
        name = peek()[1]
        i += 2
        if peek()[0] in ("num", "str"):
            raise OperationError(_LITERAL_ASSIGN)
    kind, op = peek()
    if kind != "ident":
        raise OperationError("expected an operation such as FIND(...)")
    if op not in SIGS:
        raise OperationError(f"unknown operation {op!r}; use one of {', '.join(SIGS)}")
    i += 1
    if peek() != ("punct", "("):
        raise OperationError(f"expected '(' after {op}")
    i += 1

    stmt = Statement(name=name, op=op)
    if peek() == ("punct", ")"):
        i += 1
    else:
        while True:
            kind, text = peek()
            if kind is None:
                raise OperationError("missing ')'")
            if kind == "ident" and peek(1) == ("punct", "("):
                raise OperationError(_NESTED)
            if kind == "ident" and peek(1) == ("punct", "="):
                key = text
                i += 2
                if peek()[0] == "ident" and peek(1) == ("punct", "("):
                    raise OperationError(_NESTED)
                value = _arg(peek())
                if key in stmt.kwargs:
                    raise OperationError(f"keyword {key!r} given twice")
                stmt.kwargs[key] = value
            else:
                stmt.args.append(_arg(peek()))
            i += 1
            sep = peek()
            if sep == ("punct", ","):
                i += 1
                continue
            if sep == ("punct", ")"):
                i += 1
                break
            raise OperationError("expected ',' or ')' between arguments")
    if i != len(tokens):
        raise OperationError("one operation per action; unexpected text after ')'")
    return stmt


def _arg(token) -> Arg:
    kind, text = token
    if kind is None:
        raise OperationError("missing ')'")
    if kind == "str":
        return Arg("str", text)
    if kind == "num":
        return Arg("num", text)
    if kind == "ident":
        return Arg("var" if VAR_RE.match(text) else "word", text)
    raise OperationError(f"unexpected {text!r}")


def admissible_ops(var_types: List[str]) -> List[str]:
    """Ops whose required variable types are all available."""
    have = Counter(var_types)
    return [op for op, need in SIGS.items() if all(have[t] >= n for t, n in Counter(need).items())]
