"""The Stage 1 checkpoint choice: the earliest checkpoint from --from_step on whose heldout rollouts meet
    action validity at T 0.4 >= 0.90, mixed groups at T 1.0 >= 0.35, no answer (or the step limit) at T 1.0 <= 0.10,
read from the evaluations scripts.eval.rollouts wrote into each checkpoint (eval_<dataset>_heldout.json at T 0.4,
eval_<dataset>_heldout_T1.0.json at T 1.0, 8 rollouts per question). Prints one row per evaluated checkpoint and, on
the last line, the chosen checkpoint (or nothing when none meets the rule yet).

    python -m scripts.stage1.select outputs/stage1/MultiTQ --dataset MultiTQ
"""
import argparse
import glob
import json
import os
import re
import sys

VALID_T04, MIXED_T10, NO_ANSWER_T10 = 0.90, 0.35, 0.10


def measures(ckpt: str, dataset: str):
    p4, p1 = f"{ckpt}/eval_{dataset}_heldout.json", f"{ckpt}/eval_{dataset}_heldout_T1.0.json"
    if not (os.path.exists(p4) and os.path.exists(p1)):
        return None
    e4, e1 = json.load(open(p4)), json.load(open(p1))
    rows = [json.loads(l) for l in open(p1.replace(".json", ".rollouts.jsonl"), encoding="utf-8")]
    no_answer = sum(r.get("failure_type") == "no_answer" or bool(r.get("truncated")) for r in rows) / len(rows)
    return {"valid_T0.4": e4["action_validity"], "success_T0.4": e4["success"], "valid_T1.0": e1["action_validity"],
            "success_T1.0": e1["success"], "mixed_T1.0": e1["mixed_groups"], "no_answer_T1.0": no_answer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--dataset", default="MultiTQ")
    ap.add_argument("--from_step", type=int, default=50)
    args = ap.parse_args()
    ckpts = sorted(glob.glob(os.path.join(args.run, "global_step_*")), key=lambda c: int(re.findall(r"\d+$", c)[0]))
    chosen = None
    for c in ckpts:
        if int(re.findall(r"\d+$", c)[0]) < args.from_step:
            continue
        m = measures(c, args.dataset)
        if m is None:
            continue
        ok = m["valid_T0.4"] >= VALID_T04 and m["mixed_T1.0"] >= MIXED_T10 and m["no_answer_T1.0"] <= NO_ANSWER_T10
        print(os.path.basename(c), " ".join(f"{k} {v:.3f}" for k, v in m.items()), "MEETS" if ok else "-", file=sys.stderr)
        if ok and chosen is None:
            chosen = c
    print(chosen or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
