"""Execution hindsight (EH) for the analyzer: diagnostics computed from the execution record and the gold answer.

``step_hindsight`` is the one-line annotation of each action's result that the environment attaches to the step's
info (the analyzer's "Hindsight:" line), and ``auto_hints`` the diagnostic hints it attaches when the episode ends;
``compute_hindsight`` is the record-level version computed after the episode ends. EH is analyzer input only and never
reaches the actor prompt.
- role: variables on the answer-producing path ("main"), or on a reference path ("ref") when they reach the answer only
  through a FILTER time argument (the reference time) or the subtracted set of EXCLUDE. Reference variables describe
  the reference event, so answer preservation is not measured there.
- cov: gold-answer recall, the share of gold answers still inside a variable (fact sets are read through the answer
  kind); None when not measurable (times, numbers).
- answer-loss step (lost-at step): the first step on the answer-producing path whose recall falls below its best parent
  on that path.
- suspect steps (the decisions to revise): the answer-loss step plus the reference-path steps feeding it (the cause can
  be an earlier reference choice: a reference-time dependency). A PICK can only lose gold its input still held, so a
  loss at PICK also takes the FILTERs that built that input and their reference paths (a wrong reference time makes
  PICK choose the wrong fact). Without a loss: for "too many answers" the FILTER / PICK steps of the answer-producing
  path and the reference paths; for "never retrieved" the retrieval (FIND) steps.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from tkgqa.analysis import Analyzer
from tkgqa.answers import normalize_set, score_answer, score_episode
from tkgqa.operations import Arg
from tkgqa.executor import Session, Var


def parents_by_role(var: Var) -> Tuple[List[str], List[str]]:
    """(parents on the answer-producing path, reference parents) of a variable."""
    positional = [a for a in var.args if isinstance(a, Arg)]
    refs = [(i, a.text) for i, a in enumerate(positional) if a.kind == "var"]
    if var.op == "FILTER":
        return [t for i, t in refs if i == 0], [t for i, t in refs if i >= 2]
    if var.op == "EXCLUDE":
        return [t for i, t in refs if i == 0], [t for i, t in refs if i == 1]
    return [t for _, t in refs], []


def branch_roles(session: Session) -> Dict[str, str]:
    """"main" or "ref" for every variable on the answer's lineage ("main" wins when both apply)."""
    roles: Dict[str, str] = {}
    if session.answer_var is None:
        return roles
    stack = [(session.answer_var, "main")]
    while stack:
        name, role = stack.pop()
        var = session.vars.get(name)
        if var is None or roles.get(name) in ("main", role):
            continue
        roles[name] = role
        main, ref = parents_by_role(var)
        stack.extend((p, role) for p in main)
        stack.extend((p, "ref") for p in ref)
    return roles


def _size(var: Var) -> str:
    n = len(var.value)
    unit = {"f": "fact", "t": "span", "s": "value"}[var.type]
    return f"{n} {unit}" + ("" if n == 1 else "s")


def compute_hindsight(
    session: Session,
    analyzer: Analyzer,
    gold: List[Any],
    kind: str,
    gran: Optional[str] = None,
) -> Dict[str, Any]:
    gold_norm = normalize_set(gold, kind)
    roles = branch_roles(session)
    cov: Dict[str, Optional[float]] = {}
    for name, var in session.vars.items():
        cov[name] = None if roles.get(name) == "ref" or var.type == "t" else analyzer.coverage(var, gold_norm, kind, gran)
    score = score_episode(session.answer, gold, kind)

    steps = []
    for k, step in enumerate(session.log):
        entry: Dict[str, Any] = {"step": k, "status": step.status, "op": step.op}
        if step.status == "ok" and step.op == "ANSWER":
            entry.update(precision=score["precision"], recall=score["recall"], f1=score["f1"])
        elif step.status == "ok":
            var = session.vars[step.var]
            entry.update(var=var.name, size=_size(var), role=roles.get(var.name, "unused"), cov=cov[var.name])
        steps.append(entry)

    step_of = lambda name: session.vars[name].step
    main_vars = sorted((n for n, r in roles.items() if r == "main"), key=step_of)
    drop = None
    for name in main_vars:
        own = cov[name]
        if own is None:
            continue
        parent_cov = [cov[p] for p in parents_by_role(session.vars[name])[0] if cov.get(p) is not None]
        if parent_cov and own < max(parent_cov):
            drop = name
            break

    answered = session.answer is not None
    reached = [n for n, c in cov.items() if c]
    if not answered:
        failure = "no_answer"
    elif score["success"]:
        failure = "correct"
    elif score["f1"] > 0:
        failure = "partial"
    elif not reached:
        failure = "wrong_value" if kind == "number" else "not_retrieved"
    elif any(roles.get(n) == "main" for n in reached):
        failure = "lost_on_path"
    else:
        failure = "off_path"

    suspects: Set[int] = set()
    ref_steps = {step_of(n) for n, r in roles.items() if r == "ref"}
    if failure != "correct":
        if drop is not None:
            suspects.add(step_of(drop))
            sources = [drop]
            if session.vars[drop].op == "PICK":
                stack, seen = list(parents_by_role(session.vars[drop])[0]), set()
                while stack:
                    name = stack.pop()
                    if name in seen or name not in session.vars:
                        continue
                    seen.add(name)
                    if session.vars[name].op == "FILTER":
                        suspects.add(step_of(name))
                        sources.append(name)
                    stack.extend(parents_by_role(session.vars[name])[0])
            for source in sources:
                for ref in parents_by_role(session.vars[source])[1]:
                    suspects |= {step_of(a) for a in session.ancestors(ref)}
        elif failure == "partial" and score["recall"] == 1.0:
            suspects |= {step_of(n) for n in main_vars if session.vars[n].op in ("FILTER", "PICK")} | ref_steps
        elif failure == "not_retrieved":
            suspects |= {k for k, s in enumerate(session.log) if s.op == "FIND"}

    return {
        "steps": steps,
        "failure_type": failure,
        "drop_step": step_of(drop) if drop is not None else None,
        "suspect_steps": sorted(suspects),
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "success": score["success"],
    }


def _gold_sides(analyzer: Analyzer, var: Var, gold_norm: Set[str]) -> str:
    """Which side of the facts holds gold entities: "subj", "obj", "subj and obj" or ""."""
    kg, ids = analyzer.kg, list(var.value)
    sides = []
    for side, column in (("subj", kg.subj), ("obj", kg.obj)):
        names = normalize_set([kg.entity_texts[i] for i in column[ids].tolist()], "entity")
        if gold_norm & names:
            sides.append(side)
    return " and ".join(sides)


def _recall(var: Var, analyzer: Analyzer, gold: List[Any], gold_norm: Set[str], kind: str, gran: Optional[str]) -> Optional[float]:
    if var.type == "s":
        return score_answer(var.value, gold, kind)["recall"]
    if var.type == "f":
        return analyzer.coverage(var, gold_norm, kind, gran)
    return None


def step_hindsight(session: Session, k: int, analyzer: Analyzer, gold: List[Any], kind: str,
                   gran: Optional[str] = None) -> str:
    """The analyzer's Hindsight line for step ``k`` (analyzer input only, never in the actor prompt).

    f: "f3 = 2 facts, recall 1.00, gold on subj" (number answers: the size only; "size vs gold" for count
    questions needs the answer's operation, known only at the end, and no current dataset has count questions);
    t: "t1 = 1 span, aux from f1"; s: "s1 = 1 value, recall 1.00, precision 1.00";
    invalid: "n/a (invalid action)"; failed FIND: "no variable (FIND failed)";
    ANSWER: "answer recall 1.00, precision 1.00". recall = share of gold answers still in the value.
    """
    step = session.log[k]
    if step.status == "invalid":
        return "n/a (invalid action)"
    if step.status == "find_failed":
        return "no variable (FIND failed)"
    if step.op == "ANSWER":
        score = score_episode(session.answer, gold, kind)
        return f"answer recall {score['recall']:.2f}, precision {score['precision']:.2f}"
    var = session.vars[step.var]
    head = f"{var.name} = {_size(var)}"
    if var.type == "t":
        return f"{head}, aux from {', '.join(var.parents)}" if var.parents else f"{head}, aux"
    if var.type == "s":
        score = score_answer(var.value, gold, kind)
        return f"{head}, recall {score['recall']:.2f}, precision {score['precision']:.2f}"
    gold_norm = normalize_set(gold, kind)
    recall = analyzer.coverage(var, gold_norm, kind, gran)
    if recall is None:
        return head
    line = f"{head}, recall {recall:.2f}"
    if kind == "entity" and recall > 0:
        sides = _gold_sides(analyzer, var, gold_norm)
        if sides:
            line += f", gold on {sides}"
    return line


def auto_hints(session: Session, analyzer: Analyzer, gold: List[Any], kind: str, gran: Optional[str],
               hindsight: Dict[str, Any]) -> str:
    """Rule-based hints for the analyzer, computed when the episode ends (they may be wrong):

    gold first present: the first step whose result holds a gold answer (and on which side);
    lost at: the step on the answer-producing path where recall first falls (compute_hindsight's answer-loss step);
    wasted: invalid actions and failed FINDs;
    unused: steps whose variable does not feed the answer (answered episodes only; "n/a (no answer)" otherwise);
    suspects: compute_hindsight's suspect steps.
    """
    gold_norm = normalize_set(gold, kind)
    first = "never"
    for k, step in enumerate(session.log):
        if step.status != "ok" or step.op == "ANSWER":
            continue
        var = session.vars[step.var]
        recall = _recall(var, analyzer, gold, gold_norm, kind, gran)
        if recall:
            sides = _gold_sides(analyzer, var, gold_norm) if var.type == "f" and kind == "entity" else ""
            first = f"step {k} ({var.name}, as {sides})" if sides else f"step {k} ({var.name})"
            break
    wasted = [k for k, step in enumerate(session.log) if step.status in ("invalid", "find_failed")]
    if session.answer_var:
        lineage = set(session.ancestors(session.answer_var))
        unused = str([k for k, step in enumerate(session.log)
                      if step.status == "ok" and step.op != "ANSWER" and step.var not in lineage])
    else:
        unused = "n/a (no answer)"
    drop = hindsight.get("drop_step")
    return "\n".join([
        f"gold first present: {first}",
        f"lost at: {'none' if drop is None else f'step {drop}'}",
        f"wasted: {wasted}",
        f"unused: {unused}",
        f"suspects: {list(hindsight.get('suspect_steps') or [])}",
    ])


# Discarded gold-bearing facts: the analyzer sees only two facts of each fact set, so for the answer-loss step the facts it removed
# although they hold a gold answer (at most DROPPED_GOLD_LIMIT) are added to that step's Hindsight line. The environment
# appends them to auto_hints as one line "<DROPPED_GOLD_MARK><step>: <suffix>"; seed/analysis.py moves that line to the
# step's Hindsight line when it builds the analyzer prompt, and DROPPED_GOLD_INTRO explains it there.
DROPPED_GOLD_MARK = "dropped gold at step "
DROPPED_GOLD_LIMIT = 10
DROPPED_GOLD_INTRO = ('\n"dropped facts holding a gold answer" = facts that step removed (for a value result: the input facts) although\n'
                      'their subj, obj or date is a gold answer.')


# Extra answers: an answer that holds every gold value plus values the question rules out (e.g. its own reference entity)
# scores F1 < 1 without any lost gold; the environment appends one auto_hints line naming those values.
EXTRA_ANSWERS_MARK = "extra answers: "


def extra_answers_line(answer: List[Any], gold: List[Any], kind: str, question: str) -> str:
    """"extra answers: the answer holds every gold answer and n other values: a, b[; named in the question: a]" when the
    submitted answer covers the gold set and holds values outside it; "" otherwise."""
    values = [str(v) for v in dict.fromkeys(answer or [])]
    gold_norm = normalize_set(gold, kind)
    if not values or not gold_norm:
        return ""
    norm = {v: normalize_set([v], kind) for v in values}
    if not gold_norm <= set().union(*norm.values()):
        return ""
    extra = [v for v in values if not norm[v] & gold_norm]
    if not extra:
        return ""
    shown = ", ".join(extra[:5]) + (f" (+{len(extra) - 5} more)" if len(extra) > 5 else "")
    named = [v for v in extra if v.lower() in (question or "").lower()]
    line = f"{EXTRA_ANSWERS_MARK}the answer holds every gold answer and {len(extra)} other value{'s' if len(extra) > 1 else ''}: {shown}"
    return line + (f"; named in the question: {', '.join(named[:3])}" if named else "")


def dropped_gold_suffix(session: Session, analyzer: Analyzer, gold: List[Any], kind: str, gran: Optional[str],
                        k: int) -> str:
    """" | dropped facts holding a gold answer (n of m; d dropped): (s, r, o, t); ..." for answer-loss step k, or "".

    Discarded = in the step's input fact sets on the answer-producing path, not in its result (a value result, e.g. GET: the input facts
    holding a gold answer). At most DROPPED_GOLD_LIMIT: one per gold value first, then the facts closest in time to what
    the step kept (time order when it kept none), shown in time order."""
    from tkgqa.timeutil import format_span
    kg = analyzer.kg
    gold_norm = normalize_set(gold, kind)
    if not gold_norm or k is None or not 0 <= k < len(session.log) or session.log[k].var not in session.vars:
        return ""
    var = session.vars[session.log[k].var]
    inputs: Set[int] = set()
    for p in parents_by_role(var)[0]:
        pv = session.vars.get(p)
        if pv is not None and pv.type == "f":
            inputs |= {int(x) for x in pv.value}
    if var.type == "f":
        kept = {int(x) for x in var.value}
        dropped, label, verb = sorted(inputs - kept), "dropped facts holding a gold answer", "dropped"
    else:
        kept, dropped, label, verb = set(), sorted(inputs), "input facts holding a gold answer", "read"
    holding = []
    for i in dropped:
        hit = sorted((analyzer.fields_of([i], kind, gran, None) or set()) & gold_norm)
        if hit:
            holding.append((i, hit[0]))
    if not holding:
        return ""
    kept_t = [int(kg.start[i]) for i in kept]
    dist = (lambda i: min(abs(int(kg.start[i]) - t) for t in kept_t)) if kept_t else (lambda i: int(kg.start[i]))
    order = sorted(holding, key=lambda x: (dist(x[0]), x[0]))
    chosen, seen = [], set()
    for i, g in order:  # one per gold value first
        if g not in seen:
            chosen.append((i, g))
            seen.add(g)
    for item in order:
        if len(chosen) >= DROPPED_GOLD_LIMIT:
            break
        if item not in chosen:
            chosen.append(item)
    chosen = sorted(chosen[:DROPPED_GOLD_LIMIT], key=lambda x: (int(kg.start[x[0]]), x[0]))
    facts = []
    for i, _ in chosen:
        f = kg.fact(int(i))
        s, r, o = f.subj, f.rel, f.obj
        when = format_span((f.start, f.end), kg.date_precision, collapse=True)
        facts.append(f"({s}, {r}, {o}, {when})")
    return f" | {label} ({len(chosen)} of {len(holding)}; {len(dropped)} {verb}): " + "; ".join(facts)
