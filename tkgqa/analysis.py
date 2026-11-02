"""Post-hoc analysis of a finished session against the gold answer.

``coverage`` follows the design note: the share of gold items present in a
variable (fact sets are read through the answer kind; times are n/a).
Failure types and the suspect step are a first draft:
- correct (F1 = 1) / partial (0 < F1 < 1, a failure) / no_answer (no ANSWER before the step limit);
- not_retrieved: no variable ever contained a gold item;
- lost_on_path: the answer's lineage contained gold items but lost them;
- off_path: gold items appear only in variables outside the answer's lineage;
- wrong_value: wrong number answer (fact sets carry no coverage for numbers).
The suspect step is the first step on the answer's lineage whose coverage is
lower than its best parent's.
"""

from typing import Any, Dict, Iterable, List, Optional, Set

from tkgqa.answers import normalize_set, score_episode
from tkgqa.executor import Session, Var
from tkgqa.kg import TemporalKG
from tkgqa.timeutil import format_date, format_span


class Analyzer:
    def __init__(self, kg: TemporalKG):
        self.kg = kg

    def fields_of(self, ids: Iterable[int], kind: str, gran: Optional[str], precision: Optional[str]) -> Optional[Set[str]]:
        kg = self.kg
        ids = list(ids)
        if kind == "entity":
            names = [kg.entity_texts[i] for i in kg.subj[ids].tolist()] + [kg.entity_texts[i] for i in kg.obj[ids].tolist()]
            return normalize_set(names, kind)
        if kind == "time":
            dates = set(kg.start[ids].tolist()) | set(kg.end[ids].tolist())
            return normalize_set([format_date(d, gran or "day") for d in dates], kind, precision)
        if kind == "span":
            spans = set(zip(kg.start[ids].tolist(), kg.end[ids].tolist()))
            return normalize_set([format_span(s, gran or "day") for s in spans], kind, precision)
        return None  # numbers are not read off facts

    def coverage(
        self,
        var: Var,
        gold: Set[str],
        kind: str,
        gran: Optional[str] = None,
        precision: Optional[str] = None,
    ) -> Optional[float]:
        """Share of (normalized) gold items found in ``var``; None when undefined."""
        if not gold:
            return None
        if var.type == "f":
            items = self.fields_of(var.value, kind, gran, precision)
            if items is None:
                return None
        elif var.type == "s":
            items = normalize_set(var.value, kind, precision)
        else:
            return None
        return len(gold & items) / len(gold)

    def analyze(
        self,
        session: Session,
        gold_values: List[Any],
        kind: str,
        gran: Optional[str] = None,
        precision: Optional[str] = None,
    ) -> Dict[str, Any]:
        gold = normalize_set(gold_values, kind, precision)
        coverage = {name: self.coverage(var, gold, kind, gran, precision) for name, var in session.vars.items()}
        answered = session.answer is not None
        score = score_episode(session.answer, gold_values, kind)
        lineage = session.ancestors(session.answer_var) if session.answer_var else []

        suspect_var = None
        for name in lineage:
            parent_cov = [coverage[p] for p in session.vars[name].parents if coverage.get(p) is not None]
            own = coverage[name]
            if parent_cov and own is not None and own < max(parent_cov):
                suspect_var = name
                break

        if not answered:
            failure = "no_answer"
        elif score["success"]:
            failure = "correct"
        elif score["f1"] > 0:
            failure = "partial"
        else:
            reached = [n for n, c in coverage.items() if c]
            if not reached:
                failure = "wrong_value" if kind == "number" else "not_retrieved"
            elif any(n in lineage for n in reached):
                failure = "lost_on_path"
            else:
                failure = "off_path"

        return {
            "score": score,
            "failure_type": failure,
            "suspect_var": suspect_var,
            "suspect_step": session.vars[suspect_var].step if suspect_var else None,
            "coverage": coverage,
            "lineage": lineage,
            "num_steps": len(session.log),
            "num_invalid": sum(not s.valid for s in session.log),
            "truncated": session.truncated,
        }
