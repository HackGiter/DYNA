#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Generic lm-evaluation-harness launcher for the local Meta eval templates.

Prepare custom Meta tasks first, for example:
  python prepare_meta_eval.py --config_path ./eval_config_3B_instruct.yaml

Run an HF/accelerate evaluation:
  MODEL_PATH=/path/to/model \
  TASKS=meta_math \
  INCLUDE_PATH=./work_dir_3B \
  NUM_PROCESSES=4 \
  ./run_meta_eval_example.sh

Run with vLLM directly:
  RUNNER=direct \
  MODEL_BACKEND=vllm \
  MODEL_PATH=/path/to/model \
  TASKS=meta_instruct \
  INCLUDE_PATH=./work_dir \
  TENSOR_PARALLEL_SIZE=1 \
  DATA_PARALLEL_SIZE=4 \
  ./run_meta_eval_example.sh

Common environment variables:
  MODEL_PATH              Required unless MODEL_ARGS is set.
  MODEL_ARGS              Full lm_eval --model_args override.
  MODEL_BACKEND           hf or vllm. Default: hf.
  RUNNER                  accelerate or direct. Default: accelerate.
  TASKS                   Comma-separated lm_eval tasks. Default: meta_math.
  INCLUDE_PATH            Task yaml directory. Default: ./work_dir.
  OUTPUT_PATH             Result directory. Default: eval_results.
  NUM_PROCESSES           accelerate process count. Default: 4.
  DEVICES                 Optional CUDA/NPU visible device list, e.g. 0,1,2,3.
  BATCH_SIZE              lm_eval batch size. Default: auto.
  LIMIT                   Optional lm_eval --limit value.
  LOG_SAMPLES             1/0. Default: 1.
  APPLY_CHAT_TEMPLATE     1/0. Default: 0.
  FEWSHOT_AS_MULTITURN    1/0. Default: 0.
  GEN_KWARGS              Optional lm_eval --gen_kwargs value.
  CACHE_DIR               Optional lm_eval request cache path.
  DRY_RUN                 1 prints the command without running it.

Any positional arguments are appended to lm_eval, for example:
  ./run_meta_eval_example.sh -- --show_config
USAGE
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUNNER="${RUNNER:-accelerate}"
MODEL_BACKEND="${MODEL_BACKEND:-hf}"
MODEL_PATH="${MODEL_PATH:-}"
SEED="${SEED:-42}"

if [[ -z "${MODEL_ARGS:-}" ]]; then
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required unless MODEL_ARGS is set." >&2
    usage >&2
    exit 2
  fi

  case "${MODEL_BACKEND}" in
    hf)
      MODEL_ARGS="pretrained=${MODEL_PATH},parallelize=False,dtype=auto,add_bos_token=True"
      ;;
    vllm)
      TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
      DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
      MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE},max_model_len=${MAX_MODEL_LEN},add_bos_token=True,seed=${SEED}"
      ;;
    *)
      echo "Unsupported MODEL_BACKEND: ${MODEL_BACKEND}" >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${DEVICES}"
  export NPU_VISIBLE_DEVICES="${DEVICES}"
  export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
fi

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"

TASKS="${TASKS:-meta_math}"
INCLUDE_PATH="${INCLUDE_PATH:-./work_dir}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_results}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

lm_eval_args=(
  --model "${MODEL_BACKEND}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
  --seed "${SEED}"
)

if [[ -n "${INCLUDE_PATH}" ]]; then
  lm_eval_args+=(--include_path "${INCLUDE_PATH}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  lm_eval_args+=(--limit "${LIMIT}")
fi

if is_true "${LOG_SAMPLES:-1}"; then
  lm_eval_args+=(--log_samples)
fi

if is_true "${APPLY_CHAT_TEMPLATE:-0}"; then
  lm_eval_args+=(--apply_chat_template)
fi

if is_true "${FEWSHOT_AS_MULTITURN:-0}"; then
  lm_eval_args+=(--fewshot_as_multiturn)
fi

if [[ -n "${GEN_KWARGS:-}" ]]; then
  lm_eval_args+=(--gen_kwargs "${GEN_KWARGS}")
fi

if [[ -n "${CACHE_DIR:-}" ]]; then
  lm_eval_args+=(--use_cache "${CACHE_DIR}" --cache_requests true)
fi

lm_eval_args+=("$@")

case "${RUNNER}" in
  accelerate)
    NUM_PROCESSES="${NUM_PROCESSES:-4}"
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
    accelerate_args=(launch --main_process_port "${MAIN_PROCESS_PORT}" --num_processes "${NUM_PROCESSES}")
    if [[ "${NUM_PROCESSES}" != "1" ]]; then
      accelerate_args+=(--multi_gpu)
    fi
    cmd=(accelerate "${accelerate_args[@]}" -m lm_eval "${lm_eval_args[@]}")
    ;;
  direct)
    cmd=(lm_eval "${lm_eval_args[@]}")
    ;;
  *)
    echo "Unsupported RUNNER: ${RUNNER}" >&2
    exit 2
    ;;
esac

print_command "${cmd[@]}"

if is_true "${DRY_RUN:-0}"; then
  exit 0
fi

exec "${cmd[@]}"
