#!/usr/bin/env bash
# Test of one dataset (the whole test split, greedy, two shards): the given Hugging Face checkpoints, or by default the
# Stage 2 checkpoint and both Stage 3 arms at step 150; then EM / Hits@1 / F1 with paired bootstrap CIs
# (outputs/stage3/<dataset>/metrics_test.md).
#   DATASET=MultiTQ bash scripts/eval/run.sh                       # start, TRED 150, GRPO 150
#   DATASET=MultiTQ bash scripts/eval/run.sh NAME=<HF dir> ...      # chosen checkpoints
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
TEST="${TEST:-$SPL/test.jsonl}"; TS=$(basename "$TEST" .jsonl); RUN3=outputs/stage3/$DATASET
if (( $# )); then
  M=("$@"); P=()
else
  M=("start=$(cat outputs/stage2/$DATASET/selected 2>/dev/null)"); P=()
  for a in tred grpo; do H=$RUN3/$a/global_step_150/actor/huggingface; [[ -f $H/config.json ]] && M+=("${a^^}=$H"); done
  [[ ${#M[@]} == 3 ]] && P=(--pair TRED-GRPO)
fi
for spec in "${M[@]}"; do
  name=${spec%%=*}; h=${spec#*=}
  [[ -f $h/config.json ]] || { log "no checkpoint for $name ($h)"; exit 1; }
  [[ -f $h/eval_${DATASET}_${TS}.json ]] && continue
  log "test $TS: $name ($h)"
  both test_${DATASET}_$name "\"$PY\" -m scripts.eval.rollouts --dataset $DATASET --model_path $h --questions $TEST --n 1 --temperature 0.0 --shard @K@ --num_shards 2" \
    && "$PY" -m scripts.eval.rollouts --dataset $DATASET --model_path $h --questions $TEST --merge 2 | sed 's/^/    /' \
    || { log "test FAILED (logs/test_${DATASET}_${name}_*.log)"; exit 1; }
done
mkdir -p $RUN3
"$PY" -m scripts.eval.metrics "${M[@]}" "${P[@]}" --dataset $DATASET --split $TS --out $RUN3/metrics_$TS.md
