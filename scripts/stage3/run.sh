#!/usr/bin/env bash
# Stage 3, Self-Evolving OPD of one dataset, from the Stage 2 checkpoint (outputs/stage2/<dataset>/selected, or START):
# one arm (ARM=tred, or ARM=grpo for the GRPO baseline) for 150 steps (scripts/train/rl.sh), then the checkpoints of steps
# 50 / 100 / 150 merged into Hugging Face weights (outputs/stage3/<dataset>/<arm>/global_step_<n>/actor/huggingface).
#   DATASET=MultiTQ ARM=tred bash scripts/stage3/run.sh 2>&1 | tee -a logs/stage3_MultiTQ_tred.log
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
ARM="${ARM:-tred}"
START="${START:-$(cat outputs/stage2/$DATASET/selected 2>/dev/null)}"
[[ -n $START && -f $START/config.json ]] || { log "no Stage 2 checkpoint (run scripts/stage2/run.sh or set START)"; exit 1; }
RUN3=outputs/stage3/$DATASET/$ARM

log "3 Self-Evolving OPD, $ARM, from $START -> $RUN3"
ARM=$ARM START="$REPO/$START" EXP=$ARM bash scripts/train/rl.sh > logs/stage3_${DATASET}_${ARM}_train.log 2>&1 \
  || { log "  $ARM FAILED (logs/stage3_${DATASET}_${ARM}_train.log)"; exit 1; }
for s in 50 100 150; do  # FSDP shards -> HF weights
  A=$RUN3/global_step_$s/actor
  [[ -f $A/huggingface/config.json ]] || CUDA_VISIBLE_DEVICES= "$PY" -m scripts.train.merge_ckpt merge --backend fsdp --local_dir $A \
    --target_dir $A/huggingface > logs/merge_${DATASET}_${ARM}_$s.log 2>&1
done
log "  $ARM done"
