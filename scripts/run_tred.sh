#!/usr/bin/env bash
# The whole TRED pipeline of one dataset (DATASET=MultiTQ, the default, or TimelineCronQR), in order; every step skips
# what is already there: data, Stage 1 (Temporal-Operation SFT), Stage 2 (Hindsight-Skill SFT), Stage 3 (Self-Evolving
# OPD: TRED, then the GRPO baseline), test.
#   DATASET=MultiTQ bash scripts/run_tred.sh 2>&1 | tee -a logs/run_tred_MultiTQ.log
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
bash scripts/stage1/run.sh || exit 1
bash scripts/stage2/run.sh || exit 1
ARM=tred bash scripts/stage3/run.sh || exit 1
ARM=grpo bash scripts/stage3/run.sh || exit 1
bash scripts/eval/run.sh || exit 1
log "done: outputs/stage3/$DATASET/metrics_test.md"
