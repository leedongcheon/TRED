"""The question splits of a dataset from its raw question files: data/tkgqa/splits/<dataset>/<split>.ids (one question
id per line, in the order the pipeline reads them) -> <split>.jsonl records
    {"uid": "<dataset>:<id>", "dataset", "id", "question", "answers", "answer_type", "qtype", "kind", "source_split"}
MultiTQ (questions/{train,dev,test}.json): kind = answer_type (entity / time).
TimelineCronQR (questions/{train,validation,test}.json): answers = "answer", qtype = the dataset's own qtype, kind from
the answer type (tkgqa.answers.TCR_ANSWER_KIND: entity / time / span / number); validation is written "dev".
Splits:
    stage1     Stage 1 teacher pool (shuffled train questions)
    heldout    100 dev questions: the Stage 1 checkpoint choice and the Stage 3 validation set
    stage2     the Stage 2 collection (MultiTQ 1,602, TimelineCronQR 1,115)
    rl         2,400 Stage 3 training questions
    test       the whole test set (MultiTQ 54,584, TimelineCronQR 2,080)
    test_dedup TimelineCronQR: the 1,881 test questions whose text does not occur in train

    python -m scripts.data.prepare_splits --dataset MultiTQ [--raw datasets/MultiTQ/questions]
"""
import argparse
import glob
import json
import os
import sys

from scripts.common import REPO_ROOT
from tkgqa.answers import TCR_ANSWER_KIND
from tkgqa.kg_vocab import resolve_dataset_name

RAW_FILES = {"MultiTQ": {"train": "train", "dev": "dev", "test": "test"},
             "TimelineCronQR": {"train": "train", "validation": "dev", "test": "test"}}


def record(dataset: str, q: dict, source: str) -> dict:
    if dataset == "MultiTQ":
        answers, qtype, kind = q["answers"], q["qtype"], q["answer_type"]
        qid = q["quid"]
    else:
        answers, qtype, kind = q["answer"], q["qtype"], TCR_ANSWER_KIND[q["answer_type"]]
        qid = q["id"]
    return {"uid": f"{dataset}:{qid}", "dataset": dataset, "id": qid, "question": q["question"], "answers": answers,
            "answer_type": q["answer_type"], "qtype": qtype, "kind": kind, "source_split": source}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", default=None, help="default: datasets/<dataset>/questions")
    ap.add_argument("--split_dir", default=None, help="default: data/tkgqa/splits/<dataset>")
    args = ap.parse_args()
    dataset = resolve_dataset_name(args.dataset)
    raw_dir = args.raw or os.path.join(REPO_ROOT, "datasets", dataset, "questions")
    split_dir = args.split_dir or os.path.join(REPO_ROOT, "data", "tkgqa", "splits", dataset)
    raw = {}
    for name, source in RAW_FILES[dataset].items():
        with open(os.path.join(raw_dir, f"{name}.json"), encoding="utf-8") as f:
            for q in json.load(f):
                raw[int(q["quid"] if dataset == "MultiTQ" else q["id"])] = (source, q)
    for path in sorted(glob.glob(os.path.join(split_dir, "*.ids"))):
        ids = [int(line) for line in open(path, encoding="utf-8") if line.strip()]
        missing = [i for i in ids if i not in raw]
        if missing:
            print(f"{path}: {len(missing)} ids not in {raw_dir} (e.g. {missing[:3]})", file=sys.stderr)
            return 1
        out = path[: -len(".ids")] + ".jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for i in ids:
                source, q = raw[i]
                f.write(json.dumps(record(dataset, q, source), ensure_ascii=False) + "\n")
        print(f"{dataset} {os.path.basename(out)}: {len(ids)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
