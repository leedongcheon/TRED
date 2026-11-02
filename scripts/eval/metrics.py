"""EM, Hits@1 (SCoP's definition) and F1 of saved evaluations (scripts.eval.rollouts), with paired bootstrap CIs for
chosen pairs. No model is run; the saved rollouts are re-scored.
  EM      the normalised answer set equals the gold set (tkgqa.answers.score_answer "em"; = success, F1 = 1)
  Hits@1  SCoP's Hits@1: the FIRST predicted answer is in the gold set. The answer is the executor's value list in its
          own order (GET: entity names sorted, dates in time order); values are normalised as in tkgqa.answers
  Hits@1e the expected Hits@1 of a uniformly random answer of the set (|pred & gold| / |pred|), order-free
  F1      the mean per-question F1 of the answer set (no answer = 0)
    python -m scripts.eval.metrics TRED=<HF dir> GRPO=<HF dir> --pair TRED-GRPO [--dataset MultiTQ] [--split test] [--out table.md]
"""
import argparse
import json
import os
import random

from scripts.common import REPO_ROOT as REPO
from tkgqa.answers import normalize_set, normalize_value, score_answer


def per_question(path, gold):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g, kind = gold[r["uid"]]
        ans = list(r.get("answer") or [])
        s = score_answer(ans, g, kind) if ans else {"em": 0.0, "f1": 0.0, "precision": 0.0}
        gn = normalize_set(g, kind)
        first = normalize_value(ans[0], kind) if ans else None
        out[r["uid"]] = {"EM": s["em"], "Hits@1": float(first is not None and first in gn), "Hits@1e": s["precision"] if ans else 0.0,
                         "F1": s["f1"], "multi": len(gn) > 1}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="NAME=dir holding eval_<dataset>_<split>.rollouts.jsonl")
    ap.add_argument("--dataset", default="MultiTQ")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(REPO, "data/tkgqa/splits", a.dataset, f"{a.split}.jsonl"), encoding="utf-8")]
    gold = {q["uid"]: (list(q["answers"]), q["kind"]) for q in qs}
    keys = ("EM", "Hits@1", "Hits@1e", "F1")
    res = {}
    for spec in a.models:
        name, d = spec.split("=", 1)
        p = os.path.join(d, f"eval_{a.dataset}_{a.split}.rollouts.jsonl")
        if os.path.exists(p):
            res[name] = per_question(p, gold)
    lines = ["| model | EM | Hits@1 (SCoP) | Hits@1 expected | F1 | EM single | EM multi |", "|---|---|---|---|---|---|---|"]
    for n, pq in res.items():
        v = list(pq.values()); m = lambda k, xs: sum(x[k] for x in xs) / len(xs) if xs else float("nan")
        single, multi = [x for x in v if not x["multi"]], [x for x in v if x["multi"]]
        lines.append(f"| {n} | {m('EM', v):.4f} | {m('Hits@1', v):.4f} | {m('Hits@1e', v):.4f} | {m('F1', v):.4f} | {m('EM', single):.4f} | {m('EM', multi):.4f} |")
    if a.pair:
        lines += ["", "| pair (A - B) | EM | Hits@1 (SCoP) | Hits@1 expected | F1 |", "|---|---|---|---|---|"]
        rng = random.Random(2026)
        for pr in a.pair:
            x, y = pr.split("-", 1)
            if x not in res or y not in res:
                continue
            uids = sorted(set(res[x]) & set(res[y])); cells = []
            for k in keys:
                d = [res[x][u][k] - res[y][u][k] for u in uids]
                mean = sum(d) / len(d)
                bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(a.boot))
                lo, hi = bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot) - 1]
                cells.append(f"{100 * mean:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")
            lines.append(f"| {x} - {y} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
