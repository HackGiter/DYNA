# DYNA Repo Notes

- DYNA is an umbrella workspace containing two independent upstream Git repos:
  `LLaMA-Factory` and `llama-recipes`.
- Track the child repos from the top-level repo as submodules. Make code changes
  inside the child repo that owns the file.
- Do not commit generated training or evaluation artifacts. Keep directories such
  as `eval_results*`, `work_dir*`, `output(s)`, `logs`, `runs`, `wandb`,
  `artifacts`, `checkpoints`, `ckpt`, and `saves` local-only.
- In `llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval/`, treat shell
  scripts and config files as candidates for Git, but treat generated result and
  work directories as local artifacts.
