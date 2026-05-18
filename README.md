# DYNA

DYNA is a local research workspace for the paper algorithm "Dynamic Rescaling
Anomalous Token with 1D Convolution". The repository is now a single monorepo
that vendors the two codebases used in this project:

- `LLaMA-Factory`: training code, DYNA implementation, and local launch
  configs.
- `llama-recipes`: evaluation framework and reusable evaluation launcher
  examples.

## Repository Layout

```text
DYNA/
  README.md
  AGENTS.md
  LLaMA-Factory/
    src/llamafactory/train/rwcls/
    examples/train_full/instruct_tuning/
    examples/dyna/
    exec/ift/
    exec/dyna/
  llama-recipes/
    tools/benchmarks/llm_eval_harness/meta_eval/
```

## Git Policy

Generated training and evaluation artifacts should stay local. Do not commit
directories such as:

- `eval_results*`, `work_dir*`, `kernel_meta/`
- `outputs/`, `logs/`, `runs/`, `wandb/`
- `artifacts/`, `checkpoints/`, `checkpoint*/`, `ckpt/`, `saves/`

In `LLaMA-Factory`, large experiment matrices and one-off launchers are treated
as local artifacts:

- `examples/train_full/instruct_tuning/`
- `examples/train_full/lora/`
- `examples/train_full/sae/`
- `exec/ift/`
- `exec/sae/`
- ad hoc `exec/*.sh`

Reusable local DYNA helpers live under `examples/dyna/` and `exec/dyna/`. These
are intentionally ignored by git in this workspace.

## DYNA Implementation

The current DYNA implementation is in:

- `LLaMA-Factory/src/llamafactory/train/rwcls/modeling_rwcls.py`
- `LLaMA-Factory/src/llamafactory/train/rwcls/workflow.py`
- `LLaMA-Factory/src/llamafactory/train/tuner.py`

The public training stage is `stage: dyna`. It is routed to the historical
`rwcls` implementation directory internally. The implementation computes token-level
cross entropy, smooths token losses with a 1D Gaussian convolution, selects
local top-k anomalous tokens, maintains a global anomalous-token pool, and
rescales losses after warmup.

Important: the `dyna` training stage must be registered in the LLaMA-Factory
hparams/parser code before running `stage: dyna` configs. In this workspace,
that wiring is in the vendored `LLaMA-Factory` tree.

## Current DYNA Config Matrix

The local ignored DYNA matrix in `LLaMA-Factory/examples/dyna/` mirrors the
original historical `rwcls` experiments, but uses DYNA-facing names and
`stage: dyna`:

| Config stem | Model | Dataset/domain |
| --- | --- | --- |
| `dyna-3.2-1b-base-code` | Llama-3.2-1B | Magicoder code |
| `dyna-3.2-1b-base-math` | Llama-3.2-1B | MathInstruct |
| `dyna-3.2-1b-base-sq-math` | Llama-3.2-1B | ScaleQuest Math |
| `dyna-3.2-1b-instruct-code` | Llama-3.2-1B-Instruct | Magicoder code |
| `dyna-3.2-1b-instruct-math` | Llama-3.2-1B-Instruct | ScaleQuest Math |
| `dyna-3.2-3b-base-code` | Llama-3.2-3B | Magicoder code |
| `dyna-3.2-3b-base-math` | Llama-3.2-3B | MathInstruct |
| `dyna-3.2-3b-base-sq-math` | Llama-3.2-3B | ScaleQuest Math |
| `dyna-3.2-3b-instruct-code` | Llama-3.2-3B-Instruct | Magicoder code |
| `dyna-3.2-3b-instruct-math` | Llama-3.2-3B-Instruct | ScaleQuest Math |
| `dyna2-3.2-3b-instruct-code` | Llama-3.2-3B-Instruct | RWCLS48 |
| `dyna-3.1-8b-base-code` | Llama-3.1-8B | Magicoder code |
| `dyna-3.1-8b-base-math` | Llama-3.1-8B | MathInstruct |
| `neftune-dyna-3.2-1b-base-code` | Llama-3.2-1B | Magicoder code |
| `neftune-dyna-3.2-1b-base-math` | Llama-3.2-1B | MathInstruct |
| `neftune-dyna-3.2-3b-base-code` | Llama-3.2-3B | Magicoder code |
| `neftune-dyna-3.2-3b-base-math` | Llama-3.2-3B | MathInstruct |
| `neftune-dyna-3.1-8b-base-code` | Llama-3.1-8B | Magicoder code |
| `neftune-dyna-3.1-8b-base-math` | Llama-3.1-8B | MathInstruct |

`general` has an original SFT baseline, not an original DYNA config. Do not add
it to the DYNA matrix unless a new experiment is intentionally designed.

`finance` is not part of the original DYNA matrix in this workspace. The
existing finance files are plain SFT/LoRA launchers and should not be treated as
DYNA unless a new experiment is intentionally designed.

## Running DYNA Locally

Use the local ignored runner from inside `LLaMA-Factory`:

```bash
cd /Users/lihaoze/Desktop/DYNA/LLaMA-Factory
exec/dyna/run_dyna.sh list
```

Short aliases run local helper configs:

```bash
DRY_RUN=1 exec/dyna/run_dyna.sh math
DRY_RUN=1 exec/dyna/run_dyna.sh code
```

To run a config from the current matrix, pass its DYNA config stem:

```bash
DRY_RUN=1 exec/dyna/run_dyna.sh dyna-3.2-1b-base-math
DRY_RUN=1 exec/dyna/run_dyna.sh dyna-3.2-3b-instruct-code
DRY_RUN=1 exec/dyna/run_dyna.sh dyna-3.1-8b-base-code
DRY_RUN=1 exec/dyna/run_dyna.sh neftune-dyna-3.2-3b-base-math
```

To actually launch:

```bash
DRY_RUN=0 DEVICES=4,5,6,7 exec/dyna/run_dyna.sh dyna-3.2-3b-base-math
```

The runner creates `config.json` under the target `output_dir` when missing.
That config supplies DYNA-specific parameters:

- `model_type=rwcls` for compatibility with the historical internal model class
- `local_window_size`
- `mean`, `std`
- `epoch`
- `local_numel_ratio`
- `global_numel_ratio`
- `eos`

Override them with environment variables such as:

```bash
DYNA_LOCAL_WINDOW_SIZE=5 \
DYNA_REFRESH_STEPS=1737 \
DYNA_LOCAL_RATIO=0.05 \
DYNA_GLOBAL_RATIO=0.25 \
DRY_RUN=0 \
exec/dyna/run_dyna.sh dyna-3.2-3b-base-code
```

The runner refuses plain SFT configs and historical `rwcls-*` names, because it
is intended for `stage: dyna` configs.

## Evaluation

Evaluation helpers live in:

```text
llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval/
```

Use the reusable example launcher there instead of committing one-off
`lm_eval_*.sh` scripts. Generated `eval_results*`, `work_dir*`, cache, and
kernel metadata should remain local.

## Common Checks

```bash
# Top-level workspace
git status --short --branch

# LLaMA-Factory subtree
git status --short --branch --ignored=matching -- LLaMA-Factory/examples/dyna LLaMA-Factory/exec/dyna
bash -n LLaMA-Factory/exec/dyna/*.sh
LLaMA-Factory/exec/dyna/run_dyna.sh list

# evaluation helpers
git status --short --branch -- llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval
```
