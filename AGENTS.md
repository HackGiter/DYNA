# DYNA Repo Notes

- DYNA is an umbrella workspace containing two independent upstream Git repos:
  `LLaMA-Factory` and `llama-recipes`.
- Track the child repos from the top-level repo as submodules. Make code changes
  inside the child repo that owns the file.
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
