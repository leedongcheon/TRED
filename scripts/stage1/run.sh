#!/usr/bin/env bash
# Stage 1, Temporal-Operation SFT of one dataset; restartable (every step skips what is already there).
#   1a  teacher runs (API, OPENAI_MODEL): a pilot of 10 questions, then up to 25 accepted trajectories per question type
#       (budget from the pilot; 2 attempts at T 0.0 / 0.7, the retry carries the wrong answer, never the gold answer)
#   1b  SFT set (one trajectory per question) + operation balance (every temporal operation in >= 25 trajectories)
#   1c  SFT of BASE_MODEL: 3-epoch schedule, lr 5e-6, batch 4, a checkpoint every 25 steps, stopped after step 300
#   1d  checkpoint choice: the earliest from step 50 with heldout validity (T 0.4) >= .90, mixed groups (T 1.0) >= .35,
#       no answer (T 1.0) <= 10% (scripts/stage1/select.py) -> outputs/stage1/<dataset>/selected
# API spending is capped (CAP_PILOT + CAP_MAIN, default $0.5 + $3.5 at GPT-4.1-mini prices, PRICE_IN / PRICE_CACHED /
# PRICE_OUT per 1M tokens).
#   DATASET=MultiTQ bash scripts/stage1/run.sh 2>&1 | tee -a logs/stage1_MultiTQ.log
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PRICE=(--price_in "${PRICE_IN:-0.4}" --price_cached "${PRICE_CACHED:-0.1}" --price_out "${PRICE_OUT:-1.6}")
CAP_PILOT="${CAP_PILOT:-0.5}"; CAP_MAIN="${CAP_MAIN:-3.5}"
TEACHER=data/tkgqa/stage1/$DATASET; SFT1_RAW=data/tkgqa/sft/stage1_raw/$DATASET; SFT1=data/tkgqa/sft/stage1/$DATASET; RUN1=outputs/stage1/$DATASET
if [[ ! -f $BASE_MODEL/config.json ]]; then  # a Hugging Face id: download it once (the HF cache) and use the local copy
  log "base model $BASE_MODEL from the Hugging Face Hub"
  BASE_MODEL=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$BASE_MODEL" 2> logs/base_model.log) \
    && [[ -f $BASE_MODEL/config.json ]] || { log "cannot get the base model (logs/base_model.log)"; exit 1; }
  export BASE_MODEL
fi
bash scripts/data/prepare.sh || exit 1

log "1a teacher runs with $OPENAI_MODEL -> $TEACHER"
mkdir -p $TEACHER
GEN=(--dataset $DATASET --attempts 2 --temperatures 0.0,0.7 --max_steps $MAX_STEPS --device cpu --model $OPENAI_MODEL "${PRICE[@]}")
if [[ ! -f $TEACHER/.pilot_done ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage1.teacher "${GEN[@]}" --limit 10 --max_cost $CAP_PILOT --out $TEACHER/pilot.jsonl \
    $([[ -f $TEACHER/pilot.jsonl ]] && echo --resume) 2>&1 | quiet | tail -2 | sed 's/^/    /'
  [[ ${PIPESTATUS[0]} == 0 ]] && touch $TEACHER/.pilot_done || { log "pilot FAILED"; exit 1; }
fi
if [[ ! -f $TEACHER/.main_done ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage1.teacher "${GEN[@]}" --limit 1200 --exclude $TEACHER/pilot.jsonl \
    --budget_from $TEACHER/pilot.jsonl --count_from $TEACHER/pilot.jsonl --per_type_cap 25 --max_cost $CAP_MAIN \
    --out $TEACHER/main.jsonl $([[ -f $TEACHER/main.jsonl ]] && echo --resume) 2>&1 | quiet | tail -3 | sed 's/^/    /'
  [[ ${PIPESTATUS[0]} == 0 ]] && touch $TEACHER/.main_done || { log "main FAILED"; exit 1; }
fi

log "1b SFT set $SFT1_RAW and its operation balance $SFT1"
if [[ ! -f $SFT1_RAW/train.parquet ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -m scripts.stage1.build_sft $TEACHER/pilot.jsonl $TEACHER/main.jsonl --out_dir $SFT1_RAW \
    > logs/stage1_build_sft_$DATASET.log 2>&1 || { log "SFT set FAILED (logs/stage1_build_sft_$DATASET.log)"; exit 1; }
fi
[[ -f $SFT1/train.parquet ]] || "$PY" -m scripts.stage1.op_balance $SFT1_RAW $SFT1 | sed 's/^/    /'

log "1c Temporal-Operation SFT -> $RUN1"
if [[ ! -f $RUN1/global_step_300/config.json ]]; then
  [[ -e $RUN1 ]] && { log "STOP: $RUN1 exists without step 300"; exit 1; }
  MODEL_PATH="$BASE_MODEL" DATA_DIR="$SFT1" OUT_DIR="$REPO/$RUN1" EPOCHS=3 SAVE_FREQ=25 STOP_AFTER=300 MAX_LEN=4096 NAME=stage1 \
    bash scripts/train/sft.sh > logs/stage1_sft_$DATASET.log 2>&1 || { log "Stage 1 SFT FAILED (logs/stage1_sft_$DATASET.log)"; exit 1; }
fi

log "1d Stage 1 checkpoint choice"
measure() {  # measure <checkpoint> <gpu>: heldout at T 1.0 and T 0.4, 8 rollouts each
  CUDA_VISIBLE_DEVICES=$2 "$PY" -m scripts.eval.rollouts --model_path $1 --dataset $DATASET --n 8 --temperature 1.0 \
    --out $1/eval_${DATASET}_heldout_T1.0.json > logs/stage1_eval_${DATASET}_$(basename $1).log 2>&1
  CUDA_VISIBLE_DEVICES=$2 "$PY" -m scripts.eval.rollouts --model_path $1 --dataset $DATASET --n 8 --temperature 0.4 \
    >> logs/stage1_eval_${DATASET}_$(basename $1).log 2>&1
}
CK1=""; ST=($(for s in $(seq 50 25 300); do [[ -f $RUN1/global_step_$s/config.json ]] && echo $s; done))
for ((i = 0; i < ${#ST[@]}; i += 2)); do
  CK1=$("$PY" -m scripts.stage1.select $RUN1 --dataset $DATASET 2> logs/stage1_select_$DATASET.log | tail -1)
  [[ -n $CK1 ]] && break
  if [[ "$GA" == "$GB" ]]; then
    measure $RUN1/global_step_${ST[i]} $GA; (( i + 1 < ${#ST[@]} )) && measure $RUN1/global_step_${ST[i + 1]} $GA
  else
    measure $RUN1/global_step_${ST[i]} $GA & a=$!; b=""
    (( i + 1 < ${#ST[@]} )) && { measure $RUN1/global_step_${ST[i + 1]} $GB & b=$!; }
    wait $a ${b:+$b}
  fi
done
CK1=$("$PY" -m scripts.stage1.select $RUN1 --dataset $DATASET 2> logs/stage1_select_$DATASET.log | tail -1)
sed 's/^/    /' logs/stage1_select_$DATASET.log
[[ -n $CK1 ]] || { log "no Stage 1 checkpoint meets the rule -> stop"; exit 1; }
echo "$CK1" > $RUN1/selected
log "  Stage 1 checkpoint: $CK1 (-> $RUN1/selected)"
