# Shared settings of the TRED scripts: `source scripts/env.sh` (from anywhere).
# Reads the repo-level .env (values already exported in the shell take precedence); see README: Environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"; value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    [[ -n "${!key+x}" ]] || export "$key=$value"
  done < "$REPO/.env"
fi
if [[ -z "${PY:-}" ]]; then  # the active conda env, else the conda env "tred" (README: Setup), else python on PATH
  for c in "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" "$HOME"/{miniforge3,miniconda3,anaconda3}/envs/tred/bin/python "$(command -v python 2>/dev/null)"; do
    [[ -n "$c" && -x "$c" ]] && { PY="$c"; break; }
  done
fi
export PY="${PY:?set PY to the python of the environment from requirements.txt}"
"$PY" -c "import omegaconf, pandas" 2>/dev/null || { echo "$PY lacks the packages of requirements.txt; set PY to the right python" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$(dirname "$PY"):$PATH"
export DATASET="${DATASET:-MultiTQ}"   # MultiTQ or TimelineCronQR
case "$DATASET" in TimelineCronQR) DATASET_MAX_STEPS=15;; *) DATASET_MAX_STEPS=10;; esac
export MAX_STEPS="${MAX_STEPS:-$DATASET_MAX_STEPS}"  # episode step limit (offline.MAX_STEPS_BY_DATASET), the same in every stage
export "TKGQA_MAX_STEPS_${DATASET}=$MAX_STEPS"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"   # the Stage 1 starting point: a Hugging Face id or a local directory
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4.1-mini}"                   # the Stage 1 teacher and the Stage 2 labeller
# The GPUs: CUDA_VISIBLE_DEVICES (default 0,1). Sharded steps run one shard on each of the first two; with one GPU they
# run one after the other.
export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
GA="${GPUS%%,*}"; GB="$(cut -d, -f2 <<< "$GPUS,")"; GB="${GB:-$GA}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
SPL=data/tkgqa/splits/$DATASET
cd "$REPO"
mkdir -p logs
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
quiet() { grep --line-buffered -v -i -E "warn|^\s*$"; }
both() {  # both <log tag> <command with @K@>: shard 0 and shard 1 (of 2), on GA and GB at once, or one after the other
  local tag=$1 cmd=$2 rc=0
  if [[ "$GA" == "$GB" ]]; then
    for k in 0 1; do CUDA_VISIBLE_DEVICES=$GA bash -c "${cmd//@K@/$k}" > logs/${tag}_$k.log 2>&1 || rc=1; done
    return $rc
  fi
  local ps=() k=0
  for g in $GA $GB; do ( CUDA_VISIBLE_DEVICES=$g bash -c "${cmd//@K@/$k}" > logs/${tag}_$k.log 2>&1 ) & ps+=($!); k=$((k + 1)); done
  for p in "${ps[@]}"; do wait $p || rc=1; done; return $rc
}
