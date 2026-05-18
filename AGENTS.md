# DYNA Repo Notes

- DYNA is a single monorepo that vendors two upstream codebases:
  `LLaMA-Factory` and `llama-recipes`.
- Make code changes in the directory that owns the file, but commit everything
  from the top-level `DYNA` repo.
- Do not commit generated training or evaluation artifacts. Keep directories such
  as `eval_results*`, `work_dir*`, `kernel_meta`, `output(s)`, `logs`, `runs`, `wandb`,
  `artifacts`, `checkpoints`, `ckpt`, and `saves` local-only.
- In `llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval/`, keep reusable
  config files and generic launcher examples as Git candidates. Treat one-off
  `lm_eval_*.sh` launch scripts plus generated result/work/cache directories as
  local artifacts.
- In `LLaMA-Factory`, treat large one-off experiment matrices under
  `examples/train_full/instruct_tuning/`, `examples/train_full/lora/`,
  `examples/train_full/sae/`, `exec/ift/`, and ad hoc `exec/*.sh` as local
  artifacts. Keep reusable DYNA launch templates under `examples/dyna/` and
  `exec/dyna/`.
- DYNA launch helpers should expose DYNA-facing local names such as
  `dyna-3.2-1b-base-math` under `examples/dyna/` and `exec/dyna/`, with
  `stage: dyna`. Treat `rwcls` as a historical internal implementation name,
  not the public experiment/config name. Do not add finance as a DYNA domain
  unless the user explicitly designs a new finance DYNA experiment; current
  finance files are plain SFT/LoRA launchers.
