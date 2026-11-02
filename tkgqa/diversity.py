"""Decision-slot diversity of rollout groups.

GRPO needs reward differences inside a group, and SEED's OPD signal needs the
policy to spread probability over alternatives at decision tokens; SEED does
not create exploration. So diversity is measured where temporal decisions are
made, not over whole trajectories:
- per slot, the share of question groups whose rollouts chose different values;
- per slot, the probability the policy put on its chosen value at that token,
  and the mass it put on the alternatives (from sampled-token logprobs).
"""

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from tkgqa.operations import parse
from tkgqa.executor import Session

SLOTS = ("pick", "filter_rel", "filter_gran", "time_mode", "get_gran")

_VOCAB = {
    "pick": ("earliest", "latest"),
    "filter_rel": ("before", "after", "equal", "overlap", "during", "contains", "between"),
    "time_mode": ("first", "last", "all"),
    "gran": ("day", "month", "year"),
}
_CONTEXT = {
    "pick": re.compile(r"PICK\(\s*\w+\s*,\s*$"),
    "filter_rel": re.compile(r"FILTER\(\s*\w+\s*,\s*$"),
    "time_mode": re.compile(r"TIME\(\s*\w+\s*,\s*$"),
    "gran": re.compile(r"gran\s*=\s*$"),
}


def slot_values(session: Session) -> Dict[str, Tuple[str, ...]]:
    """Values chosen at each decision slot by the executed (valid) steps, in order."""
    values: Dict[str, List[str]] = defaultdict(list)
    for step in session.log:
        if step.status != "ok" or not step.action:
            continue
        stmt = parse(step.action)
        args = [a.text for a in stmt.args]
        if stmt.op == "PICK" and len(args) > 1:
            values["pick"].append(args[1])
        elif stmt.op == "FILTER":
            if len(args) > 1:
                values["filter_rel"].append(args[1])
            if "gran" in stmt.kwargs:
                values["filter_gran"].append(stmt.kwargs["gran"].text)
        elif stmt.op == "TIME" and len(args) > 1:
            values["time_mode"].append(args[1])
        elif stmt.op == "GET" and len(args) > 1 and args[1] == "time" and "gran" in stmt.kwargs:
            values["get_gran"].append(stmt.kwargs["gran"].text)
    return {k: tuple(v) for k, v in values.items()}


def group_divergence(groups: Dict[str, Sequence[Dict[str, Tuple[str, ...]]]]) -> Dict[str, Dict[str, float]]:
    """Per slot:
    - value_split: among groups where >= 2 rollouts use the slot, the share whose rollouts chose
      different values;
    - presence_split: share of groups where some rollouts use the slot and others do not.
    """
    out = {}
    for slot in SLOTS:
        eligible = split = presence_split = 0
        for rollouts in groups.values():
            used = [r[slot] for r in rollouts if slot in r]
            if 0 < len(used) < len(rollouts):
                presence_split += 1
            if len(used) >= 2:
                eligible += 1
                split += len(set(used)) > 1
        out[slot] = {
            "groups_using": float(eligible),
            "value_split": split / eligible if eligible else float("nan"),
            "presence_split": presence_split / len(groups) if groups else float("nan"),
        }
    return out


def _vocab_word(token: str, vocab: Sequence[str]):
    """The slot word a (possibly partial) token starts, if exactly one."""
    token = token.strip()
    if len(token) < 2:
        return None
    matches = [w for w in vocab if w.startswith(token)]
    return matches[0] if len(matches) == 1 else None


def decision_probs(pieces: Iterable[Tuple[str, float, Dict[str, float]]]) -> List[Tuple[str, str, float, float]]:
    """Decisions in one generation.

    ``pieces``: per sampled token, (decoded text, logprob, {alternative decoded text: logprob}).
    Returns (slot, value, p_chosen, p_alternatives) for each decision token inside <action>,
    where p_alternatives sums the top-k alternatives that start another value of the same slot.
    """
    text, out = "", []
    for piece, logprob, alternatives in pieces:
        if text.rfind("<action>") > text.rfind("</action>"):
            for slot, context in _CONTEXT.items():
                if not context.search(text):
                    continue
                word = _vocab_word(piece, _VOCAB[slot])
                if word is None:
                    continue
                name = slot
                if slot == "gran":
                    name = "filter_gran" if re.search(r"FILTER\([^()]*$", text) else "get_gran"
                alt = sum(
                    math.exp(lp)
                    for t, lp in alternatives.items()
                    if (w := _vocab_word(t, _VOCAB[slot])) is not None and w != word
                )
                out.append((name, word, math.exp(logprob), alt))
        text += piece
    return out


def summarize_decisions(decisions: Iterable[Tuple[str, str, float, float]]) -> Dict[str, Dict[str, float]]:
    by_slot: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for slot, _, p, alt in decisions:
        by_slot[slot].append((p, alt))
    summary = {}
    for slot in SLOTS:
        rows = by_slot.get(slot, [])
        n = len(rows)
        summary[slot] = {
            "decisions": float(n),
            "p_chosen_mean": sum(p for p, _ in rows) / n if n else float("nan"),
            "p_alternatives_mean": sum(a for _, a in rows) / n if n else float("nan"),
            "near_deterministic": sum(p > 0.99 for p, _ in rows) / n if n else float("nan"),
        }
    return summary
