#!/usr/bin/env bash
# Stage 2, Hindsight-Skill SFT of one dataset, from the Stage 1 checkpoint (outputs/stage1/<dataset>/selected, or
# STAGE1_CKPT); restartable (every step skips what is already there).
#   2a  collection: the Stage 1 checkpoint on the stage2 questions, 8 rollouts at T 0.7, with execution hindsight
#   2b  labelling order: 700 successful + every failed distinct trajectory, type-stratified
#   2c  labels (API, OPENAI_MODEL): the analyzer's 5 fields (summary, selected step, failure type, episode-level and
#       decision-level hindsight skills) for each trajectory
#   2d  decision-level skills: the fixed form, the checks, the executor pre-check
#   2e  continuation verification from the Stage 1 checkpoint (16 continuations per context, two shards), final targets
#   2f  SFT set: analyzer examples + own-trajectory replay (1/4 of them) + every verified corrected trajectory
#   2g  SFT from the Stage 1 checkpoint, 2 epochs; Stage 3 starts from the 1-epoch checkpoint -> outputs/stage2/<dataset>/selected
# API spending is capped (CAP_LABEL, default $7 at GPT-4.1-mini prices).
#   DATASET=MultiTQ bash scripts/stage2/run.sh 2>&1 | tee -a logs/stage2_MultiTQ.log
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PRICE=(--price_in "${PRICE_IN:-0.4}" --price_cached "${PRICE_CACHED:-0.1}" --price_out "${PRICE_OUT:-1.6}")
CAP_LABEL="${CAP_LABEL:-7}"
SFT1_RAW=data/tkgqa/sft/stage1_raw/$DATASET; D2=data/tkgqa/stage2/$DATASET; OWN=data/tkgqa/sft/own_replay/$DATASET
REPLAY=data/tkgqa/sft/replay/$DATASET; SFT2=data/tkgqa/sft/stage2/$DATASET; RUN2=outputs/stage2/$DATASET
CK1="${STAGE1_CKPT:-$(cat outputs/stage1/$DATASET/selected 2>/dev/null)}"
[[ -n $CK1 && -f $CK1/config.json ]] || { log "no Stage 1 checkpoint (run scripts/stage1/run.sh or set STAGE1_CKPT)"; exit 1; }
[[ -f $SFT1_RAW/train.parquet ]] || { log "no Stage 1 SFT set at $SFT1_RAW (the replay counts of 2f; run scripts/stage1/run.sh)"; exit 1; }

log "2a collection with $CK1 ($SPL/stage2.jsonl x 8 at T 0.7)"
if [[ ! -f $D2/trajectories.shard1.jsonl ]]; then
  mkdir -p $D2
  both stage2_collect_$DATASET "\"$PY\" -m scripts.stage2.collect rollout --model_path $CK1 --dataset $DATASET --questions $SPL/stage2.jsonl --n 8 --temperature 0.7 --shard @K@ --num_shards 2 --out $D2/trajectories.shard@K@.jsonl.part && mv $D2/trajectories.shard@K@.jsonl.part $D2/trajectories.shard@K@.jsonl" \
    || { log "collection FAILED (logs/stage2_collect_${DATASET}_*.log)"; exit 1; }
fi
CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage2.collect summary $D2/trajectories.shard*.jsonl 2>&1 | quiet | head -6 | sed 's/^/    /'

log "2b labelling order"
[[ -s $D2/analyze_order.jsonl ]] || "$PY" -m scripts.stage2.label_order $D2 | sed 's/^/    /'

log "2c $OPENAI_MODEL labels (cap \$$CAP_LABEL)"
if ! grep -q -E "^done: " logs/stage2_label_$DATASET.log 2>/dev/null; then
  CUDA_VISIBLE_DEVICES= "$PY" -u -m scripts.stage2.label $D2/analyze_order.jsonl --model $OPENAI_MODEL "${PRICE[@]}" --max_cost $CAP_LABEL \
    --resume --out $D2/skills.jsonl 2>&1 | quiet | tee logs/stage2_label_$DATASET.log | grep -v '^ *[0-9]*/[0-9]* |' | tail -2 | sed 's/^/    /'
fi
[[ -s $D2/skills.jsonl ]] || { log "no labels"; exit 1; }

log "2d decision-level skills: fixed form, checks, executor pre-check"
if [[ ! -f $D2/skills.prechecked.jsonl ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage2.reform $D2 | sed 's/^/    /'
  CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage2.refilter $D2 | head -1 | sed 's/^/    /'
  CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage2.precheck $D2 --dataset $DATASET 2>&1 | quiet | grep "accepted by the checks" | sed 's/^/    /'
fi

log "2e continuation verification with $CK1"
if [[ ! -f $D2/prescription_validation.summary.json ]]; then
  both stage2_verify_$DATASET "\"$PY\" -m scripts.stage2.verify run --model_path $CK1 --dataset $DATASET --dir $D2 --shard @K@ --num_shards 2" \
    || { log "continuation verification FAILED (logs/stage2_verify_${DATASET}_*.log)"; exit 1; }
  "$PY" -m scripts.stage2.verify merge --dir $D2 --num_shards 2 | sed 's/^/    /'
fi
"$PY" -m scripts.stage2.merge_skills $D2 --dataset $DATASET | sed 's/^/    /'

log "2f Hindsight-Skill SFT set $SFT2"
if [[ ! -f $SFT2/.done ]]; then
  "$PY" -m scripts.stage2.build_sft --dataset $DATASET --base $D2 --replay_dir $SFT1_RAW --replay 0 --out_dir $SFT2 \
    > logs/stage2_build_sft_$DATASET.log 2>&1 || { log "SFT set FAILED (logs/stage2_build_sft_$DATASET.log)"; exit 1; }
  NQ=$(( $("$PY" -c "import json; print(json.load(open('$SFT2/manifest.json'))['analyzer_examples']['train'])") / 4 ))
  "$PY" -m scripts.stage2.own_replay --dir $D2 --n $NQ --out $OWN > /dev/null
  "$PY" -m scripts.stage2.corrected --dir $D2 --own $OWN --rows 1000000 --out $REPLAY > /dev/null
  NR=$("$PY" -c "import pandas as pd; print(len(pd.read_parquet('$REPLAY/train.parquet')))")
  "$PY" -m scripts.stage2.build_sft --dataset $DATASET --base $D2 --replay_dir $REPLAY --replay $NR --out_dir $SFT2 \
    > logs/stage2_build_sft_$DATASET.log 2>&1 || { log "SFT set FAILED (logs/stage2_build_sft_$DATASET.log)"; exit 1; }
  touch $SFT2/.done
fi
"$PY" -c "import json; m = json.load(open('$SFT2/manifest.json')); print('   ', m['analyzer_examples'], m['replay_examples'])"

log "2g Hindsight-Skill SFT from $CK1 -> $RUN2"
if [[ ! -f $RUN2/.done ]]; then
  [[ -e $RUN2 ]] && { log "STOP: $RUN2 exists without .done"; exit 1; }
  MODEL_PATH="$REPO/$CK1" DATA_DIR="$SFT2" OUT_DIR="$REPO/$RUN2" EPOCHS=2 MAX_LEN=8192 NAME=stage2 \
    bash scripts/train/sft.sh > logs/stage2_sft_$DATASET.log 2>&1 && touch $RUN2/.done \
    || { log "Stage 2 SFT FAILED (logs/stage2_sft_$DATASET.log)"; exit 1; }
fi
CK2=$(ls -d $RUN2/global_step_* | sort -V | head -1)
echo "$CK2" > $RUN2/selected
log "  Stage 2 checkpoint (the 1-epoch one): $CK2 (-> $RUN2/selected)"
