#!/usr/bin/env bash
# Data of one dataset: the question splits from the raw files (data/tkgqa/splits/<dataset>/*.ids -> *.jsonl), the RL
# parquet (data/tkgqa/rl/<dataset>) and the bge-m3 grounder stores (stores/<dataset>, on the first GPU). Skips what exists.
#   DATASET=MultiTQ bash scripts/data/prepare.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
[[ -d datasets/$DATASET/kg && -d datasets/$DATASET/questions ]] || { log "no raw data at datasets/$DATASET (README: Data)"; exit 1; }
log "data: $DATASET"
[[ -f $SPL/heldout.jsonl ]] || "$PY" -m scripts.data.prepare_splits --dataset $DATASET | sed 's/^/    /' || exit 1
[[ -f data/tkgqa/rl/$DATASET/train.parquet ]] || "$PY" -m scripts.data.make_rl_data --dataset $DATASET | sed 's/^/    /' || exit 1
[[ -f stores/$DATASET/bge-m3/relation/manifest.json ]] || CUDA_VISIBLE_DEVICES=$GA "$PY" -m scripts.data.build_stores --dataset $DATASET 2>&1 \
  | quiet | grep -E "items ->" | sed 's/^/    /'
[[ -f stores/$DATASET/bge-m3/relation/manifest.json ]] || { log "grounder stores FAILED"; exit 1; }
