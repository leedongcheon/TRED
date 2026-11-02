# TRED: Execution-Grounded Hindsight Self-Distillation for Temporal Knowledge Graph Question Answering

**TRED** (Temporal Reasoning learned via Execution-grounded hindsight self-Distillation) for temporal knowledge graph
question answering (TKGQA), on MultiTQ and TimelineCronQR.

The actor (Qwen2.5-3B-Instruct) answers a question by choosing one **temporal operation** per step (`FIND`, `TIME`,
`FILTER`, `PICK`, `GET`, `EXCLUDE`, `UNION`, `INTERSECT`, `AGG`, `RANK`, `ANSWER`). A **symbolic executor** runs each
operation on the temporal KG, stores its result in an intermediate variable, and returns a summarized observation;
the full intermediate results and the dependencies between variables are kept in the **execution record**. From the
execution record and the gold answer, TRED computes **execution hindsight (EH)**: the gold-answer recall of every
intermediate candidate set, the answer-loss step, the discarded gold-bearing facts, and diagnostic hints that trace a
loss back along the answer-producing path (e.g. a reference-time dependency). An **analyzer** reads a completed
trajectory with its EH and writes **hindsight skills**: an episode-level hindsight skill (a reusable reasoning rule)
and, for a failure, a decision-level hindsight skill that proposes a correction at a selected step
("At this decision, write `<correction proposal>` instead of `<the agent's operation>`.").

TRED trains one model as both actor and analyzer, in three stages:

| Stage | What happens |
|---|---|
| 1. Temporal-Operation SFT | A teacher LLM (GPT-4.1-mini) solves questions against the symbolic executor; its successful trajectories (one per question, balanced over temporal operations) teach the actor to choose temporal operations. |
| 2. Hindsight-Skill SFT | The Stage 1 actor rolls out on the Stage 2 questions; GPT-4.1-mini writes the hindsight skills of each trajectory from its EH. Decision-level skills are checked (form, context, executor) and kept only if they pass continuation verification (16 continuations from the selected step with the skill vs. without it and vs. a control skill). The model is fine-tuned to play both roles: the analyzer (on the accepted labels) and the actor (on its own correct trajectories and the verified corrected trajectories). |
| 3. Self-Evolving OPD | Outcome-based RL (F1 reward, group-relative advantage) jointly with on-policy self-distillation: the policy analyzes its own trajectories with their EH, its hindsight skills form the skill-augmented context, and the OPD term is confidence-gated by the token log-probability difference between the skill-augmented and the plain context. Trajectories and skills are regenerated as the policy improves. The GRPO baseline is the same run without analysis and distillation. |

The prompts the models see call the temporal operations "DSL"; this text is kept verbatim because the released
checkpoints were trained with it.

## Setup

Python 3.12, CUDA 12 (the runs used torch 2.8.0, vLLM 0.11.0, transformers 4.57.3, flash-attn 2.8.3, Ray 2.50.0):

```bash
conda create -n tred python=3.12 -y && conda activate tred
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

The base model (`Qwen/Qwen2.5-3B-Instruct`) and the grounder's encoder (`BAAI/bge-m3`) are fetched from the Hugging Face
Hub on first use.

## Environment

The scripts read the repo-level `.env` (values already exported in the shell take precedence). Put the GPUs, the base
model and the API settings there before running the scripts:

```bash
# Runtime devices: sharded steps run one shard on each of the first two GPUs (one after the other with one GPU).
CUDA_VISIBLE_DEVICES=0,1

# Base model: a Hugging Face id or a local directory.
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct

# Used by the Stage 1 teacher runs and the Stage 2 hindsight-skill labels.
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
# OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_RETRIES=5
OPENAI_API_RETRY_DELAY=1.0

# On a machine without internet access, once the models are in the Hugging Face cache.
# HF_HUB_OFFLINE=1
```

API spending stops at a cost cap per step: Stage 1 teacher runs $0.5 + $3.5 (`CAP_PILOT`, `CAP_MAIN`), Stage 2 labels
$7 (`CAP_LABEL`), at GPT-4.1-mini prices (`PRICE_IN`, `PRICE_CACHED`, `PRICE_OUT` per 1M tokens).

## Data

Place the raw files of each dataset as

```
datasets/MultiTQ/kg/full.txt, entity2id.json, relation2id.json
datasets/MultiTQ/questions/train.json, dev.json, test.json
datasets/TimelineCronQR/kg/full.txt, full_no_underscore.txt      # facts with KG names / with readable names, same lines
datasets/TimelineCronQR/questions/train.json, validation.json, test.json
```

The question splits are shipped as question ids (`data/tkgqa/splits/<dataset>/*.ids`). Stage 1 builds the split
records, the RL data and the grounder stores of its dataset first (`DATASET=<dataset> bash scripts/data/prepare.sh`):

| split | MultiTQ | TimelineCronQR | use |
|---|---|---|---|
| `stage1` | 384,207 | 14,059 | Stage 1 teacher pool (shuffled train questions) |
| `heldout` | 100 | 100 | Stage 1 checkpoint choice, Stage 3 validation (dev questions) |
| `stage2` | 1,602 | 1,115 | Stage 2 collection |
| `rl` | 2,400 | 2,400 | Stage 3 training questions |
| `test` | 54,584 | 2,080 | test: the whole test set |
| `test_dedup` | - | 1,881 | TimelineCronQR test without the 199 questions whose text also occurs in train |

An episode has at most 10 steps on MultiTQ and 15 on TimelineCronQR (`MAX_STEPS`). `data/tkgqa/unsolvable/MultiTQ.json`
lists the MultiTQ questions left out of the Stage 2 targets (grounding ceiling; gold answers finer than the question
asks).

## Training

Every stage script works on one dataset (`DATASET`), assumes the repo root as the working directory, and skips the steps
whose outputs exist, so an interrupted run continues where it stopped.

### Stage 1: Temporal-Operation SFT

```bash
# MultiTQ
DATASET=MultiTQ bash scripts/stage1/run.sh

# TimelineCronQR
DATASET=TimelineCronQR bash scripts/stage1/run.sh
```

The teacher (GPT-4.1-mini) solves questions against the symbolic executor, up to 25 accepted trajectories per question
type; the SFT set keeps one trajectory per question, balanced so that every temporal operation occurs in at least 25
trajectories. The base model is fine-tuned on a 3-epoch schedule stopped after step 300 (lr 5e-6, batch 4, a checkpoint
every 25 steps), and the earliest checkpoint from step 50 whose heldout rollouts meet validity (T 0.4) >= 0.90, mixed
groups (T 1.0) >= 0.35 and no answer (T 1.0) <= 10% is chosen (`outputs/stage1/<dataset>/selected`).

### Stage 2: Hindsight-Skill SFT

```bash
# MultiTQ
DATASET=MultiTQ bash scripts/stage2/run.sh

# TimelineCronQR
DATASET=TimelineCronQR bash scripts/stage2/run.sh
```

The Stage 1 checkpoint rolls out on the `stage2` questions (8 rollouts at T 0.7) with execution hindsight; GPT-4.1-mini
labels 700 successful and every failed distinct trajectory. Decision-level skills are checked (form, context, executor
pre-check) and verified by continuation (16 continuations from the selected step with the skill, without it and with a
control skill; accepted when the skill wins by at least 2 against both). The SFT set holds the analyzer examples, steps of the
model's own correct trajectories (a quarter as many rows) and every verified corrected trajectory; the Stage 1 checkpoint is
fine-tuned for 2 epochs and Stage 3 starts from the 1-epoch checkpoint (`outputs/stage2/<dataset>/selected`).

### Stage 3: Self-Evolving OPD

```bash
# MultiTQ
DATASET=MultiTQ ARM=tred bash scripts/stage3/run.sh
DATASET=MultiTQ ARM=grpo bash scripts/stage3/run.sh     # the GRPO baseline

# TimelineCronQR
DATASET=TimelineCronQR ARM=tred bash scripts/stage3/run.sh
DATASET=TimelineCronQR ARM=grpo bash scripts/stage3/run.sh
```

150 steps of 16 questions x 8 rollouts at T 1.0 (lr 1e-6, KL 0.001, OPD weight 0.01, gate sigmoid(5 x the
teacher-student log-probability difference)); the checkpoints of steps 50 / 100 / 150 are exported as Hugging Face
weights under `outputs/stage3/<dataset>/<arm>/global_step_<n>/actor/huggingface`.

## Evaluation

```bash
DATASET=MultiTQ bash scripts/eval/run.sh                 # the Stage 2 checkpoint, TRED 150 and GRPO 150
DATASET=MultiTQ bash scripts/eval/run.sh NAME=<HF dir>   # any checkpoints
```

Greedy decoding on the whole test split; EM (the answer set equals the gold set), Hits@1 (the first predicted answer is
gold, SCoP's definition) and F1, with paired bootstrap 95% CIs (`outputs/stage3/<dataset>/metrics_test.md`).
`DATASET=<dataset> bash scripts/run_tred.sh` runs data, Stages 1-3 and the evaluation in one go.

## Repository layout

```
tkgqa/          temporal KG, grounder (bge-m3), symbolic executor and temporal operations, answers and scoring,
                execution hindsight, skill checks, Stage 1 few-shot examples
seed/           SEED's analyzer (the 5-field hindsight prompt) and skill prompting
agent_system/   the TKGQA environment, actor prompt and interaction history (verl-agent layout)
verl/, gigpo/   the RL / SFT trainers (verl, verl-agent, SEED), with the TKGQA decision-level skill hook
scripts/        data, stage1, stage2, stage3, train, eval (run.sh of each stage); run_tred.sh runs them in order
data/           the question-id splits and the unsolvable-question list
```

## Reproducibility notes

- The teacher runs and the labels come from an API model, so a fresh run produces different Stage 1 and Stage 2 data
  than the paper's; every later step is seeded (data order, sampling seeds per request).
- Continuation verification splits its cases into two shards by question; the control skill and the sampling seeds
  depend on the shard, so keep two shards to reproduce a run.

## Acknowledgements and license

TRED builds on [SEED](https://jinyangwu.github.io/seed/) (MIT), [verl](https://github.com/volcengine/verl) (Apache 2.0)
and [verl-agent](https://github.com/langfengQ/verl-agent) (Apache 2.0), and uses MultiTQ and bge-m3. The code is released
under the MIT License (`LICENSE`); `Notice.txt` lists the projects it includes, and files taken from them keep their
copyright headers.
