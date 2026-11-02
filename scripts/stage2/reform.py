"""Restore the fixed form of decision-level skills that give only the corrected operation. The labeller often writes a
failure's decision-level skill as a bare temporal operation such as "f4 = FILTER(f1, after, ...)" instead of the form the
analyzer prompt asks for, "At this decision, write <corrected> instead of <own>.". A failure label whose step_skill is
a single parsable temporal operation and whose failure_step is in range gets
    At this decision, write <that operation> instead of <the agent's operation at failure_step>.
with the agent's operation taken from the trajectory (the <action> of analysis_steps[failure_step]); nothing else changes.
The labels are rewritten in place (skills.jsonl), the originals kept in skills.before_reform.jsonl and in
parsed.step_skill_raw. Labels already in the form, successes and anything else are left as they are. No API call.

    python -m scripts.stage2.reform data/tkgqa/stage2/MultiTQ
"""

import glob
import json
import os
import re
import shutil
import sys
from collections import Counter

from tkgqa.operations import OperationError, parse
from tkgqa.skill_filter import prescription_parts

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)


def bare_op(text):
    t = text.strip().strip("`").strip()
    m = ACTION_RE.search(t)
    t = (m.group(1) if m else t).strip().rstrip(".").strip()
    if "\n" in t:
        return None
    try:
        parse(t)
    except (OperationError, Exception):
        return None
    return t


def main() -> int:
    d = sys.argv[1]
    src, keep = os.path.join(d, "skills.jsonl"), os.path.join(d, "skills.before_reform.jsonl")
    if not os.path.exists(keep):
        shutil.copy(src, keep)
    trajs = {(t["uid"], t["rollout"]): t for p in sorted(glob.glob(os.path.join(d, "trajectories.shard*.jsonl")))
             for t in map(json.loads, open(p, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(keep, encoding="utf-8")]
    n = Counter()
    for r in rows:
        p = r.get("parsed")
        if not p or float(r.get("success", 0)) >= 1.0:
            continue
        text = (p.get("step_skill") or "").strip()
        if not text:
            n["no step_skill"] += 1; continue
        if prescription_parts(text):
            n["already in the form"] += 1; continue
        op, k = bare_op(text), p.get("failure_step")
        steps = trajs[(r["uid"], r["rollout"])]["analysis_steps"]
        if op is None:
            n["not a single temporal operation"] += 1; continue
        if not isinstance(k, int) or not 0 <= k < len(steps):
            n["failure_step out of range"] += 1; continue
        m = ACTION_RE.search(str(steps[k].get("response") or ""))
        if not m:
            n["no agent action at failure_step"] += 1; continue
        p["step_skill_raw"] = text
        p["step_skill"] = f"At this decision, write {op} instead of {m.group(1).strip()}."
        n["reformed"] += 1
    with open(src, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(json.dumps(dict(n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
