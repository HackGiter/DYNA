# DYNA

DYNA is a local research workspace for the paper algorithm "Dynamic Rescaling
Anomalous Token with 1D Convolution". The top-level repository is an umbrella
repo. The actual code lives in two child repositories:

- `LLaMA-Factory`: training code, DYNA/rwcls implementation, and local launch
  configs.
- `llama-recipes`: evaluation framework and reusable evaluation launcher
  examples.

The child directories are tracked from the top-level repo as submodules. Make
code changes inside the child repo that owns the file.

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

The algorithm path is `stage: rwcls`. The implementation computes token-level
cross entropy, smooths token losses with a 1D Gaussian convolution, selects
local top-k anomalous tokens, maintains a global anomalous-token pool, and
rescales losses after warmup.

Important: the `rwcls` training stage must be registered in the LLaMA-Factory
hparams/parser code before running `stage: rwcls` configs. In this workspace,
that wiring is in the LLaMA-Factory working tree.

## Current DYNA Config Matrix

The original DYNA/rwcls matrix currently present in
`LLaMA-Factory/examples/train_full/instruct_tuning/` covers these model sizes
and domains:

| Config stem | Model | Dataset/domain |
| --- | --- | --- |
| `rwcls-3.2-1b-base-code` | Llama-3.2-1B | Magicoder code |
| `rwcls-3.2-1b-base-math` | Llama-3.2-1B | MathInstruct |
| `rwcls-3.2-1b-base-sq-math` | Llama-3.2-1B | ScaleQuest Math |
| `rwcls-3.2-1b-instruct-code` | Llama-3.2-1B-Instruct | Magicoder code |
| `rwcls-3.2-1b-instruct-math` | Llama-3.2-1B-Instruct | ScaleQuest Math |
| `rwcls-3.2-3b-base-code` | Llama-3.2-3B | Magicoder code |
| `rwcls-3.2-3b-base-math` | Llama-3.2-3B | MathInstruct |
| `rwcls-3.2-3b-base-sq-math` | Llama-3.2-3B | ScaleQuest Math |
| `rwcls-3.2-3b-instruct-code` | Llama-3.2-3B-Instruct | Magicoder code |
| `rwcls-3.2-3b-instruct-math` | Llama-3.2-3B-Instruct | ScaleQuest Math |
| `rwcls2-3.2-3b-instruct-code` | Llama-3.2-3B-Instruct | RWCLS48 |
| `rwcls-3.1-8b-base-code` | Llama-3.1-8B | Magicoder code |
| `rwcls-3.1-8b-base-math` | Llama-3.1-8B | MathInstruct |
| `neftune-rwcls-3.2-1b-base-code` | Llama-3.2-1B | Magicoder code |
| `neftune-rwcls-3.2-1b-base-math` | Llama-3.2-1B | MathInstruct |
| `neftune-rwcls-3.2-3b-base-code` | Llama-3.2-3B | Magicoder code |
| `neftune-rwcls-3.2-3b-base-math` | Llama-3.2-3B | MathInstruct |
| `neftune-rwcls-3.1-8b-base-code` | Llama-3.1-8B | Magicoder code |
| `neftune-rwcls-3.1-8b-base-math` | Llama-3.1-8B | MathInstruct |

`general` has an original SFT baseline, not an original DYNA/rwcls config. The
local `dyna_general.yaml` helper is an explicit adaptation from that SFT baseline
with `stage` changed to `rwcls`.

`finance` is not part of the original DYNA/rwcls matrix in this workspace. The
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
DRY_RUN=1 exec/dyna/run_dyna.sh general
```

To run an original config from the full current matrix, pass its config stem:

```bash
DRY_RUN=1 exec/dyna/run_dyna.sh rwcls-3.2-1b-base-math
DRY_RUN=1 exec/dyna/run_dyna.sh rwcls-3.2-3b-instruct-code
DRY_RUN=1 exec/dyna/run_dyna.sh rwcls-3.1-8b-base-code
DRY_RUN=1 exec/dyna/run_dyna.sh neftune-rwcls-3.2-3b-base-math
```

To actually launch:

```bash
DRY_RUN=0 DEVICES=4,5,6,7 exec/dyna/run_dyna.sh rwcls-3.2-3b-base-math
```

The runner creates `config.json` under the target `output_dir` when missing.
That config supplies DYNA-specific parameters:

- `model_type=rwcls`
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
exec/dyna/run_dyna.sh rwcls-3.2-3b-base-code
```

The runner refuses plain SFT configs, because it is intended for `stage: rwcls`.

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

# LLaMA-Factory child repo
cd LLaMA-Factory
git status --short --branch --ignored=matching -- examples/dyna exec/dyna
bash -n exec/dyna/*.sh
exec/dyna/run_dyna.sh list

# llama-recipes child repo
cd ../llama-recipes
git status --short --branch
```
