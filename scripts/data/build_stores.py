"""Build the bge-m3 entity / relation embedding stores of one dataset (stores/<dataset>/bge-m3/{entity,relation}), the files
the grounder loads (tkgqa.grounder.Grounder.from_store_root). The vocabulary comes from tkgqa.kg_vocab.load_vocab
(datasets/<dataset>/kg/...); one GPU encodes it in a few minutes.

    python -m scripts.data.build_stores --dataset MultiTQ [--overwrite]
"""

import argparse
import os
import time

from tkgqa.embedding_store import DEFAULT_MODEL, KINDS, BGEM3Encoder, store_dir, utc_timestamp, write_store
from scripts.common import REPO_ROOT as REPO
from tkgqa.kg_vocab import load_vocab, resolve_dataset_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default=os.path.join(REPO, "datasets"))
    ap.add_argument("--store_root", default=os.path.join(REPO, "stores"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    dataset = resolve_dataset_name(args.dataset)
    vocab = load_vocab(dataset, args.data_root)
    enc = BGEM3Encoder(args.model, device=args.device)
    kg_dir = os.path.join(args.data_root, dataset, "kg")
    sources = sorted(os.path.join(dataset, "kg", f) for f in os.listdir(kg_dir) if f.endswith(".txt"))
    for kind in KINDS:
        items = vocab.items(kind)
        texts = [it.text for it in items]
        t0 = time.time()
        emb = enc.encode(texts, batch_size=args.batch_size, show_progress=True)
        sec = time.time() - t0
        rows = [{"id": i, "name": it.name, "text": it.text, "freq": it.freq} for i, it in enumerate(items)]
        manifest = {"dataset": dataset, "kind": kind, "model": args.model, "model_revision": enc.revision, "pooling": "cls",
                    "normalized": True, "dim": enc.dim, "count": len(rows), "embedding_dtype": "float16",
                    "max_length": enc.max_length, "num_truncated": enc.count_truncated(texts), "text_field": "text",
                    "source_files": sources, "kg_num_facts": vocab.num_facts, "kg_dropped_facts": vocab.dropped_facts,
                    "encode_seconds": round(sec, 2), "created_at": utc_timestamp()}
        path = store_dir(args.store_root, dataset, kind, args.model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_store(path, rows, emb.cpu().numpy().astype("float16"), manifest, overwrite=args.overwrite)
        print(f"{dataset} {kind}: {len(rows)} items -> {path} ({sec:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
