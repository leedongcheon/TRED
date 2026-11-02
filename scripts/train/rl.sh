#!/usr/bin/env bash
# Stage 3, Self-Evolving OPD (verl + SEED's trainer) in the TKGQA environment, from the Stage 2 checkpoint. One arm per run:
#   ARM=tred   TRED: outcome-based RL (group-relative advantage, F1 reward) jointly with on-policy self-distillation. The
#              policy is its own analyzer (the 5-field prompt with the execution hindsight, greedy); its episode-level and
#              decision-level hindsight skills form the skill-augmented context, and the OPD term (weight OPD) is
#              confidence-gated by sigmoid(5 * (teacher - student)) token log-prob differences
#   ARM=grpo   the same estimator and settings without analysis and distillation (the GRPO baseline)
# Fixed: SEED episode_step, step_priority, 150 steps x 16 questions x 8 rollouts at T 1.0, lr 1e-6, KL 0.001 (low_var_kl),
# invalid-action penalty 0.01, gamma 0.95, the full history (MAX_STEPS = 10; TimelineCronQR 15). Validation (100 heldout questions, T 0.4)
# before training and every TEST_FREQ steps; checkpoints (model weights, no optimizer) every SAVE_FREQ steps.
# The GPUs come from scripts/env.sh (CUDA_VISIBLE_DEVICES); micro batches by a token budget, so the updates do not depend
# on the number of GPUs. scripts/stage3/run.sh calls this with the Stage 2 checkpoint.
#   ARM=tred START=outputs/stage2/MultiTQ/global_step_526 bash scripts/train/rl.sh 2>&1 | tee logs/rl_tred.log
#   smoke: ARM=tred START=... STEPS=2 VAL_BEFORE=False SAVE_FREQ=-1 TEST_FREQ=-1 EXP=smoke bash scripts/train/rl.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
ARM="${ARM:?set ARM=tred or grpo}"
START="${START:?set START to the Stage 2 checkpoint (HF dir)}"
GPUS="${GPUS:?set CUDA_VISIBLE_DEVICES in .env}"
NGPU="$(awk -F, '{print NF}' <<< "$GPUS")"
case $ARM in
  tred) ENABLE_ANALYSIS=True OPD_COEF="${OPD:-0.01}";;
  grpo) ENABLE_ANALYSIS=False OPD_COEF=0.0;;
  *) echo "ARM must be tred or grpo" >&2; exit 1;;
esac
STEPS="${STEPS:-150}"             # 2,400 RL questions / 16 per step = one pass
MAX_STEPS="${MAX_STEPS:-$DATASET_MAX_STEPS}"  # episode step limit = history length (offline.max_steps_for)
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"
VAL_BEFORE="${VAL_BEFORE:-True}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"  # questions per step
GROUP="${GROUP:-8}"               # rollouts per question
ROLLOUT_T="${ROLLOUT_T:-1.0}"
LR="${LR:-1e-6}"
KL="${KL:-0.001}"
OPD_GATE_BETA="${OPD_GATE_BETA:-5.0}"
MICRO="${MICRO:-2}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"
PPO_MAX_TOKENS="${PPO_MAX_TOKENS:-10240}"     # per GPU, update micro batches by token budget
LOGPROB_MAX_TOKENS="${LOGPROB_MAX_TOKENS:-32768}"  # per GPU, old / ref / teacher log-probs
GPU_MEM="${GPU_MEM:-0.45}"
EXP="${EXP:-${ARM}_$(basename "$START")}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/stage3/$DATASET/$EXP}"
DATA_DIR="${DATA_DIR:-$REPO/data/tkgqa/rl/$DATASET}"
RAY_TMP="${RAY_TMP:-/tmp/ray_tred_$(id -u)_${GPUS//,/}}"  # one local Ray instance per run

[[ -e "$OUT_DIR/.done" ]] && { log "$OUT_DIR is done already"; exit 0; }
export CUDA_VISIBLE_DEVICES="$GPUS" VLLM_ATTENTION_BACKEND=FLASH_ATTN
ulimit -u 65536 2>/dev/null || true
mkdir -p "$OUT_DIR"
[[ -f "$DATA_DIR/train.parquet" ]] || "$PY" -m scripts.data.make_rl_data --dataset "$DATASET"

echo "Stage 3 $ARM | dataset=$DATASET start=$START gpus=$GPUS steps=$STEPS batch=${TRAIN_BATCH}x$GROUP T=$ROLLOUT_T max_steps=$MAX_STEPS"
echo "             | opd=$OPD_COEF gate_beta=$OPD_GATE_BETA lr=$LR kl=$KL out=$OUT_DIR"

"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=seed \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size=100 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$START" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.opd_loss_coef="$OPD_COEF" \
    actor_rollout_ref.actor.opd_gate_beta="$OPD_GATE_BETA" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    "actor_rollout_ref.actor.checkpoint.contents=['model','extra']" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_T" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=9216 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.seed.step_advantage_w=0.0 \
    algorithm.seed.episode_skill_teacher_advantage_w=0.0 \
    algorithm.seed.step_skill_teacher_advantage_w=0.0 \
    algorithm.seed.skill_mode=episode_step \
    algorithm.seed.skill_teacher_mode=step_priority \
    algorithm.seed.opd_start_after_steps=null \
    algorithm.seed.opd_stop_after_steps=null \
    algorithm.seed.failed_only=False \
    algorithm.seed.failed_only_after_steps=null \
    algorithm.seed.failure_success_threshold=1.0 \
    algorithm.seed.mode=mean_std_norm \
    algorithm.seed.enable_analysis="$ENABLE_ANALYSIS" \
    algorithm.seed.selector=llm \
    algorithm.seed.analysis_backend=policy_vllm \
    algorithm.seed.analysis_num_workers=1 \
    algorithm.seed.analysis_context_length=8192 \
    algorithm.seed.analysis_max_completion_tokens=1024 \
    algorithm.seed.analysis_max_step_skills_per_traj=1 \
    algorithm.seed.analysis_prompt_version=seed \
    algorithm.seed.analysis_render_response=action_only \
    algorithm.seed.normalize_teacher_adv=False \
    env.env_name=tkgqa \
    env.seed=0 \
    env.max_steps="$MAX_STEPS" \
    env.history_length="$MAX_STEPS" \
    env.rollout.n="$GROUP" \
    env.tkgqa.dataset="$DATASET" \
    env.tkgqa.data_root="$REPO/datasets" \
    env.tkgqa.store_root="$REPO/stores" \
    env.tkgqa.device=cpu \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name="tred_stage3_$DATASET" \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node="$NGPU" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train="$VAL_BEFORE" \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.rollout_data_dir="$OUT_DIR" \
    +ray_init.address=local \
    "+ray_init._temp_dir=$RAY_TMP" \
    +ray_init.include_dashboard=False \
    "+ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH" \
    "$@"
touch "$OUT_DIR/.done"
