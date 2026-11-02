"""Apply the decision-level skill checks (tkgqa.skill_filter.validate_step_prescription) to the reformed labels; writes
<dir>/skills.step_checked.jsonl with step_violations / step_accepted recomputed. No API call.
    python -m scripts.stage2.refilter data/tkgqa/stage2/MultiTQ
"""
import glob
import json
import os
import sys
from collections import Counter

from tkgqa.skill_filter import validate_step_prescription

d = sys.argv[1]
trajs = {(t["uid"], t["rollout"]): t for p in sorted(glob.glob(os.path.join(d, "trajectories.shard*.jsonl"))) for t in map(json.loads, open(p, encoding="utf-8"))}
rows = [json.loads(line) for line in open(os.path.join(d, "skills.jsonl"), encoding="utf-8")]
before = Counter(r.get("step_accepted", False) for r in rows if r["success"] < 1)
for r in rows:
    t = trajs[(r["uid"], r["rollout"])]
    r["step_violations"] = validate_step_prescription(r["parsed"], t["analysis_steps"], r["success"], t["answers"]) if r["parsed"] else ["json"]
    r["step_accepted"] = bool(r["parsed"]) and r["success"] < 1.0 and not r["step_violations"]
fail = [r for r in rows if r["success"] < 1]
print(f"failures {len(fail)}: step prescriptions accepted {before[True]} before -> {sum(r['step_accepted'] for r in fail)} now")
print("violations:", Counter(v for r in fail for v in r["step_violations"]).most_common())
print("accepted by failure_type:", Counter(r["parsed"]["failure_type"] for r in fail if r["step_accepted"]).most_common())
with open(os.path.join(d, "skills.step_checked.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
