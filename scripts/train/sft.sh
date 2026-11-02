#!/usr/bin/env bash
# SFT of the actor (and, in Stage 2, the analyzer) on step-level examples: prompt -> response, loss on the response only
# (verl's FSDP SFT trainer). Checkpoints hold model weights only (no optimizer), in SAVE_DTYPE, at each epoch end and every
# SAVE_FREQ steps; STOP_AFTER ends the run after that step without changing the LR schedule.
# The GPUs come from scripts/env.sh (CUDA_VISIBLE_DEVICES); the global batch does not depend on their number.
#   DATA_DIR=data/tkgqa/sft/stage1/MultiTQ OUT_DIR=outputs/stage1/MultiTQ EPOCHS=3 SAVE_FREQ=25 STOP_AFTER=300 \
#     bash scripts/train/sft.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
GPUS="${GPUS:?set CUDA_VISIBLE_DEVICES in .env}"
NPROC="$(awk -F, '{print NF}' <<< "$GPUS")"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL}"
DATA_DIR="${DATA_DIR:?set DATA_DIR (train.parquet / val.parquet)}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-6}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"  # global batch; must divide by the number of GPUs
MICRO="${MICRO:-1}"
MAX_LEN="${MAX_LEN:-4096}"
SAVE_FREQ="${SAVE_FREQ:-null}"   # null = epoch ends only
STOP_AFTER="${STOP_AFTER:-null}" # null = all epochs
SAVE_DTYPE="${SAVE_DTYPE:-bf16}"
NAME="${NAME:-sft}"              # run name only

export CUDA_VISIBLE_DEVICES="$GPUS"
mkdir -p "$OUT_DIR"
echo "$NAME SFT | gpus=$GPUS model=$MODEL_PATH data=$DATA_DIR out=$OUT_DIR"
echo "        | epochs=$EPOCHS lr=$LR batch=$TRAIN_BATCH micro=$MICRO max_len=$MAX_LEN save_freq=$SAVE_FREQ stop_after=$STOP_AFTER"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    'data.prompt_dict_keys=[]' \
    'data.response_dict_keys=[]' \
    data.train_batch_size="$TRAIN_BATCH" \
    data.micro_batch_size_per_gpu="$MICRO" \
    data.max_length="$MAX_LEN" \
    data.truncation=error \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=True \
    optim.lr="$LR" \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name="tred_$NAME" \
    trainer.experiment_name="${NAME}_$DATASET" \
    'trainer.logger=[console]' \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.stop_after_steps="$STOP_AFTER" \
    trainer.save_dtype="$SAVE_DTYPE" \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    "$@"
