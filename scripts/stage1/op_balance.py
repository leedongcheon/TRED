"""Operation balance of the Stage 1 SFT set: every temporal operation of the successful teacher trajectories appears in
>= --min trajectories. The trajectories of a rarer operation (n > 0) are appended ceil(MIN / n) - 1 more times (a
trajectory with several rare operations: the largest count), in the order of their first row, each copy as the whole
block of its rows; val is unchanged. An operation in fewer than --min_support trajectories is left as it is: a single
trajectory that passed by luck is noise, not a behaviour to copy many times.

    python -m scripts.stage1.op_balance data/tkgqa/sft/stage1_raw/MultiTQ data/tkgqa/sft/stage1/MultiTQ
"""

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter

import pandas as pd

OP_RE = re.compile(r"<action>\s*(?:[a-z]+\d+\s*=\s*)?([A-Z]+)\s*\(")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--min", type=int, default=25)
    ap.add_argument("--min_support", type=int, default=5, help="ops in fewer trajectories are not repeated")
    args = ap.parse_args()
    df = pd.read_parquet(os.path.join(args.src, "train.parquet"))
    order = list(dict.fromkeys(df.uid))
    ops = {u: set() for u in order}
    for u, r in zip(df.uid, df.response):
        m = OP_RE.search(r)
        if m:
            ops[u].add(m.group(1))
    count = Counter(o for s in ops.values() for o in s)
    extra = {o: math.ceil(args.min / n) - 1 for o, n in count.items() if args.min_support <= n < args.min}
    copies = {u: max([extra[o] for o in ops[u] if o in extra], default=0) for u in order}
    blocks = [df]
    for u in order:
        if copies[u]:
            blocks += [df[df.uid == u]] * copies[u]
    out = pd.concat(blocks, ignore_index=True)
    os.makedirs(args.out, exist_ok=True)
    out.to_parquet(os.path.join(args.out, "train.parquet"), index=False)
    shutil.copy(os.path.join(args.src, "val.parquet"), os.path.join(args.out, "val.parquet"))
    rep = [u for u in order if copies[u]]
    manifest = {"source": args.src,
                "rule": f"every temporal operation of the successful teacher trajectories in >= {args.min} trajectories: the trajectories "
                        f"of a rarer op repeated ceil({args.min}/n) - 1 more times",
                "ops_before": dict(count.most_common()),
                "min_support": args.min_support,
                "not_balanced": {o: n for o, n in count.items() if n < args.min_support},
                "repeated": {"trajectories": len(rep), "extra_copies": sorted({copies[u] for u in rep})},
                "rows": {"train_before": len(df), "train_after": len(out), "val": len(pd.read_parquet(os.path.join(args.src, "val.parquet")))}}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: manifest[k] for k in ("ops_before", "repeated", "rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
