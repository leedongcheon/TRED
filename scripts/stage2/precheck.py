"""Executor pre-check of the decision-level skills (CPU, no model), before the continuation verification: replay the failed
trajectory's own actions up to failure_step, then run the corrected operation once; an invalid one is dropped before any
rollout. Decision-level skills often omit the result variable ("GET(f2, obj)"); both the decision-level skill as written and with the
agent's own result variable at that step filled in ("s1 = GET(f2, obj)") are run.
    python -m scripts.stage2.precheck data/tkgqa/stage2/MultiTQ [--dataset MultiTQ]
writes <dir>/skills.prechecked.jsonl (step_exec, step_exec_filled; step_runnable = filled operation ran without an
invalid status) from skills.step_checked.jsonl.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

from scripts.common import REPO_ROOT
from tkgqa.executor import ExecutorConfig, Session
from tkgqa.grounder import Grounder
from tkgqa.kg import TemporalKG

STEP_RE = re.compile(r"^\s*At this decision, write (.+?) instead of (.+?)\.?\s*$", re.S)
LHS_RE = re.compile(r"^\s*([fts]\d+)\s*=")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dir")
ap.add_argument("--dataset", default="MultiTQ")
ap.add_argument("--data_root", default=os.path.join(REPO_ROOT, "datasets"))
ap.add_argument("--store_root", default=os.path.join(REPO_ROOT, "stores"))
args = ap.parse_args()
d = args.dir
kg = TemporalKG.load(args.dataset, args.data_root)
grounder = Grounder.from_store_root(kg, args.store_root, device="cpu")
trajs = {(t["uid"], t["rollout"]): t for p in sorted(glob.glob(os.path.join(d, "trajectories.shard*.jsonl"))) for t in map(json.loads, open(p, encoding="utf-8"))}
rows = [json.loads(line) for line in open(os.path.join(d, "skills.step_checked.jsonl"), encoding="utf-8")]


def run(t, k, action):
    s = Session(kg, grounder, t["question"], config=ExecutorConfig(max_steps=k + 2))
    for a in t["actions"][:k]:
        s.step(f"<action> {a or ''} </action>")
    s.step(f"<action> {action} </action>")
    last = s.log[-1]
    return last.status, last.obs.split("\n")[0][:120]


stats = Counter()
for r in rows:
    if r["success"] >= 1 or not r["parsed"] or not (r["parsed"].get("step_skill") or "").strip():
        continue
    m = STEP_RE.match(r["parsed"]["step_skill"].strip())
    k = r["parsed"].get("failure_step")
    t = trajs[(r["uid"], r["rollout"])]
    if not m or k is None or not 0 <= k < len(t["actions"]):
        continue
    corrected = m.group(1).strip()
    own = t["actions"][k] or ""
    filled = corrected if LHS_RE.match(corrected) or not LHS_RE.match(own) else f"{LHS_RE.match(own).group(1)} = {corrected}"
    r["step_exec"] = run(t, k, corrected)
    r["step_exec_filled"] = run(t, k, filled)
    r["step_corrected_filled"] = filled
    r["step_runnable"] = r["step_exec_filled"][0] != "invalid"
    for name, flag in (("checked_ok", r["step_accepted"]), ("all", True)):
        if flag:
            stats[(name, "as written", r["step_exec"][0])] += 1
            stats[(name, "variable filled", r["step_exec_filled"][0])] += 1
with open(os.path.join(d, "skills.prechecked.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for name in ("checked_ok", "all"):
    tot = sum(v for (n, how, _), v in stats.items() if n == name and how == "as written")
    print(f"== {name} prescriptions ({tot}):")
    for how in ("as written", "variable filled"):
        print(f"   {how:16}", {st: v for (n, h, st), v in sorted(stats.items()) if n == name and h == how})
ok = [r for r in rows if r.get("step_accepted") and r.get("step_runnable")]
print(f"accepted by the checks and runnable (variable filled): {len(ok)}")
print("invalid examples:", [(r["step_corrected_filled"], r["step_exec_filled"][1]) for r in rows if r.get("step_accepted") and not r.get("step_runnable")][:6])
