#!/usr/bin/env bash
#
# SGLang benchmark runner — perf sweep and GSM8K accuracy in one script.
# Pick one mode via --mode; perf and acc configs are independent.
#
# Usage:
#   bash sweep_bench.sh --model-key glm              # perf (default mode)
#   bash sweep_bench.sh --model-key glm --mode acc   # GSM8K x3 by default
#   bash perf_bench.sh --model-key glm               # same as --mode perf
#   bash acc_bench.sh  --model-key glm               # same as --mode acc
#
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "ERROR: run with bash, not sh:  bash $0 ..." >&2
  exit 1
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench_common.sh"
bench_common_init

SUMMARIZE_PERF="${BENCH_SCRIPT_DIR}/summarize_perf.py"
SUMMARIZE_ACC="${BENCH_SCRIPT_DIR}/summarize_acc.py"

usage() {
  cat <<'EOF'
Usage: sweep_bench.sh --model-key KEY [options]

Shared:
  --model-key KEY          qwen | dsv4 | glm  (required)
  --mode MODE              perf (throughput sweep) or acc (GSM8K accuracy)  (default: perf)
  --model-override PATH    override resolved model path
  --gpu-vendor VENDOR      cuda | amd  (default: auto-detect)
  --gpu TAG                short GPU tag, e.g. mi355  (default: auto-detect)
  --node NAME              node name tag  (default: hostname)
  --num-gpus N             TP size for per-GPU throughput in summary
  --sglang-root PATH       sglang checkout  (default: ../../sglang)
  --server-host HOST       server host for MTP auto-detect  (default: 127.0.0.1)
  --server-port PORT       server port for MTP auto-detect  (default: 30000)
  --mtp                    tag/folder suffix: force mtp (skip auto-detect)
  --no-mtp                 tag/folder suffix: force nomtp (skip auto-detect)

Perf (--mode perf):
  --perf-io-pairs PAIRS    comma-separated input:output pairs  (default: 70000:300)
  --perf-concurrencies N   comma-separated max-concurrency values  (default: 4,8,16,32,64)
  --perf-range-ratio F     random length range ratio  (default: 0.8)
  --perf-prompts-per-conc N  num-prompts = concurrency * N  (default: 5)
  --perf-result-dir PATH   output directory  (default: perf_<model>_<mtp|nomtp>_<gpu>_<node>_<ts>)
  --profile                enable torch profiling (output=16, prompts=conc*2)
  --profile-output-dir DIR profile trace directory  (default: /sgl-workspace/profiles)
  --profile-num-steps N    profile decode steps  (default: 10)
  --profile-prefix PREFIX  trace filename prefix  (default: auto per run)

Accuracy (--mode acc):
  --acc-num-questions N    (default: 2000)
  --acc-parallel N         (default: 64)
  --acc-num-shots N        (default: 5)
  --acc-host HOST          server host  (default: 127.0.0.1)
  --acc-port PORT          server port  (default: 30000)
  --acc-runs N             repeat GSM8K eval N times  (default: 3)
  --acc-result-dir PATH    output directory  (default: acc_<model>_<mtp|nomtp>_<gpu>_<node>_<ts>)

  -h, --help               show this help
EOF
}

# ---- Defaults (overridden by CLI flags) ------------------------------------
BENCH_MODE="perf"
MODEL_KEY=""
MODEL_OVERRIDE=""
GPU_VENDOR=""
GPU=""
NODE=""
NUM_GPUS=""
SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$BENCH_SCRIPT_DIR/../../sglang" 2>/dev/null && pwd)}"
BENCH_SERVER_HOST="127.0.0.1"
BENCH_SERVER_PORT="30000"
MTP_OVERRIDE=""

#PERF_IO_PAIRS_STR="70000:300"
PERF_IO_PAIRS_STR="8192:1024"
PERF_CONCURRENCIES_STR="4,8,16,32,64"
PERF_RANGE_RATIO="0.8"
PERF_PROMPTS_PER_CONC="5"
PERF_RESULT_DIR=""
PERF_PROFILE=false
PERF_PROFILE_OUTPUT_DIR="/sgl-workspace/profiles"
PERF_PROFILE_NUM_STEPS="10"
PERF_PROFILE_PREFIX=""
PERF_PROFILE_OUTPUT_LEN="16"
PERF_PROFILE_PROMPT_MULTIPLIER="2"

ACC_NUM_QUESTIONS="2000"
ACC_PARALLEL="64"
ACC_NUM_SHOTS="5"
ACC_HOST="127.0.0.1"
ACC_PORT="30000"
ACC_RUNS="3"
ACC_RESULT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)              BENCH_MODE="$2"; shift 2 ;;
    --model-key)         MODEL_KEY="$2"; shift 2 ;;
    --model-override)    MODEL_OVERRIDE="$2"; shift 2 ;;
    --gpu-vendor)        GPU_VENDOR="$2"; shift 2 ;;
    --gpu)               GPU="$2"; shift 2 ;;
    --node)              NODE="$2"; shift 2 ;;
    --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
    --sglang-root)       SGLANG_ROOT="$2"; shift 2 ;;
    --server-host)       BENCH_SERVER_HOST="$2"; shift 2 ;;
    --server-port)       BENCH_SERVER_PORT="$2"; shift 2 ;;
    --mtp)               MTP_OVERRIDE="mtp"; shift ;;
    --no-mtp)            MTP_OVERRIDE="nomtp"; shift ;;
    --perf-io-pairs)     PERF_IO_PAIRS_STR="$2"; shift 2 ;;
    --perf-concurrencies) PERF_CONCURRENCIES_STR="$2"; shift 2 ;;
    --perf-range-ratio)  PERF_RANGE_RATIO="$2"; shift 2 ;;
    --perf-prompts-per-conc) PERF_PROMPTS_PER_CONC="$2"; shift 2 ;;
    --perf-result-dir)   PERF_RESULT_DIR="$2"; shift 2 ;;
    --profile)           PERF_PROFILE=true; shift ;;
    --profile-output-dir) PERF_PROFILE_OUTPUT_DIR="$2"; shift 2 ;;
    --profile-num-steps) PERF_PROFILE_NUM_STEPS="$2"; shift 2 ;;
    --profile-prefix)    PERF_PROFILE_PREFIX="$2"; shift 2 ;;
    --acc-num-questions) ACC_NUM_QUESTIONS="$2"; shift 2 ;;
    --acc-parallel)      ACC_PARALLEL="$2"; shift 2 ;;
    --acc-num-shots)     ACC_NUM_SHOTS="$2"; shift 2 ;;
    --acc-host)          ACC_HOST="$2"; shift 2 ;;
    --acc-port)          ACC_PORT="$2"; shift 2 ;;
    --acc-runs)          ACC_RUNS="$2"; shift 2 ;;
    --acc-result-dir)    ACC_RESULT_DIR="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_KEY" ]]; then
  echo "ERROR: --model-key is required (qwen | dsv4 | glm)" >&2
  usage >&2
  exit 1
fi

if ! [[ "$ACC_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --acc-runs must be a positive integer (got: '$ACC_RUNS')" >&2
  exit 1
fi

if [[ "$PERF_PROFILE" == true ]]; then
  if ! [[ "$PERF_PROFILE_NUM_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --profile-num-steps must be a positive integer (got: '$PERF_PROFILE_NUM_STEPS')" >&2
    exit 1
  fi
  mkdir -p "$PERF_PROFILE_OUTPUT_DIR"
fi

if [[ -z "$GPU_VENDOR" ]]; then
  GPU_VENDOR="$(detect_gpu_vendor)"
fi

bench_resolve_model
bench_detect_node_gpu

if [[ "$BENCH_MODE" == "acc" ]]; then
  BENCH_SERVER_HOST="$ACC_HOST"
  BENCH_SERVER_PORT="$ACC_PORT"
fi
bench_resolve_mtp_tag
TAG="${TAG}_${MTP_TAG}"

IFS=',' read -ra _perf_io_raw <<< "$PERF_IO_PAIRS_STR"
PERF_IO_PAIRS=("${_perf_io_raw[@]}")

IFS=',' read -ra PERF_CONCURRENCIES <<< "$PERF_CONCURRENCIES_STR"

_ts="$(date +%Y%m%d_%H%M%S)"
PERF_RESULT_DIR="${PERF_RESULT_DIR:-perf_${MODEL_KEY}_${MTP_TAG}_${GPU}_${NODE}_${_ts}}"
ACC_RESULT_DIR="${ACC_RESULT_DIR:-acc_${MODEL_KEY}_${MTP_TAG}_${GPU}_${NODE}_${_ts}}"

echo "Model: $MODEL  (MODEL_KEY=$MODEL_KEY, GPU_VENDOR=$GPU_VENDOR)"
[[ -n "${NUM_GPUS:-}" ]] && echo "Num GPUs (TP): $NUM_GPUS"
echo "Node: $NODE  GPU: $GPU  (detected: ${GPU_RAW:-none})"
echo "MTP tag: $MTP_TAG  (server=${BENCH_SERVER_HOST}:${BENCH_SERVER_PORT}$([ -n "${MTP_OVERRIDE:-}" ] && printf ', override=%s' "$MTP_OVERRIDE"))"
echo "File tag prefix: $TAG"
echo "Mode: $BENCH_MODE"
if [[ "$PERF_PROFILE" == true ]]; then
  echo "Profile: enabled  output_len=${PERF_PROFILE_OUTPUT_LEN}  prompts=conc*${PERF_PROFILE_PROMPT_MULTIPLIER}"
  echo "Profile dir: $PERF_PROFILE_OUTPUT_DIR  num_steps=${PERF_PROFILE_NUM_STEPS}"
fi

run_perf_sweep() {
  local result_dir="$PERF_RESULT_DIR"
  mkdir -p "$result_dir"
  result_dir="$(cd "$result_dir" && pwd)"
  echo "Writing perf results to: $result_dir"

  run_one() {
    local ilen=$1 olen=$2 conc=$3
    local run_olen="$olen"
    local run_range_ratio="$PERF_RANGE_RATIO"
    local prompts_per_conc="$PERF_PROMPTS_PER_CONC"
    local nprompts profile_args=() profile_prefix

    if [[ "$PERF_PROFILE" == true ]]; then
      run_olen="$PERF_PROFILE_OUTPUT_LEN"
      run_range_ratio="1.0"
      prompts_per_conc="$PERF_PROFILE_PROMPT_MULTIPLIER"
    fi

    nprompts=$((conc * prompts_per_conc))
    local tag="${ilen}_o${run_olen}_c${conc}"
    local jsonl="${result_dir}/${TAG}_${tag}.jsonl"
    local log="${result_dir}/${TAG}_${tag}.log"

    echo
    echo "==================================================================="
    echo ">>> input_len=${ilen}  output_len=${run_olen}  max_concurrency=${conc}  num_prompts=${nprompts}"
    if [[ "$PERF_PROFILE" == true ]]; then
      profile_prefix="${PERF_PROFILE_PREFIX:-${TAG}_${MODEL_KEY}_${GPU}_i${ilen}_c${conc}}"
      echo ">>> profile_prefix=${profile_prefix}  profile_num_steps=${PERF_PROFILE_NUM_STEPS}"
    fi
    echo "==================================================================="

    if [[ "$PERF_PROFILE" == true ]]; then
      profile_args=(
        --profile
        --profile-output-dir "$PERF_PROFILE_OUTPUT_DIR"
        --profile-by-stage
        --profile-num-steps "$PERF_PROFILE_NUM_STEPS"
        --profile-prefix "$profile_prefix"
      )
    fi

    python3 -m sglang.bench_serving \
      --model "$MODEL" \
      --dataset-name random \
      --random-input "$ilen" \
      --random-output "$run_olen" \
      --random-range-ratio "$run_range_ratio" \
      --max-concurrency "$conc" \
      --num-prompts "$nprompts" \
      --output-file "$jsonl" \
      "${profile_args[@]}" \
      2>&1 | tee "$log"
  }

  local pair ilen olen conc
  for pair in "${PERF_IO_PAIRS[@]}"; do
    ilen="${pair%%:*}"
    olen="${pair##*:}"
    for conc in "${PERF_CONCURRENCIES[@]}"; do
      run_one "$ilen" "$olen" "$conc" \
        || echo "WARN: run failed for input_len=${ilen} output_len=${olen} conc=${conc}, continuing..."
    done
  done

  echo
  echo "Perf runs done. Summarizing -> ${result_dir}/summary.txt"
  if ! python3 "$SUMMARIZE_PERF" "$result_dir" "$TAG" "$MODEL" "$NODE" "$GPU" "${NUM_GPUS:-}" \
      | tee "${result_dir}/summary.txt"; then
    echo "WARN: perf summary failed (see above); raw logs/jsonl are still in ${result_dir}" >&2
  fi
}

run_accuracy() {
  bench_require_sglang_root

  local result_dir="$ACC_RESULT_DIR"
  mkdir -p "$result_dir"
  result_dir="$(cd "$result_dir" && pwd)"
  echo "SGLang root: $SGLANG_ROOT"
  echo "Server: ${ACC_HOST}:${ACC_PORT}"
  echo "Accuracy runs: ${ACC_RUNS}"
  echo "Writing accuracy results to: $result_dir"

  local run_idx run_tag log result_jsonl
  for ((run_idx = 1; run_idx <= ACC_RUNS; run_idx++)); do
    run_tag="$(printf 'run%03d' "$run_idx")"
    log="${result_dir}/${TAG}_${run_tag}.log"
    result_jsonl="${result_dir}/${TAG}_${run_tag}.jsonl"

    echo
    echo "==================================================================="
    echo ">>> GSM8K ${run_tag} (${run_idx}/${ACC_RUNS})"
    echo ">>> num_questions=${ACC_NUM_QUESTIONS}  parallel=${ACC_PARALLEL}  num_shots=${ACC_NUM_SHOTS}"
    echo "==================================================================="

    if ! (
      cd "$SGLANG_ROOT"
      python3 benchmark/gsm8k/bench_sglang.py \
        --num-questions "$ACC_NUM_QUESTIONS" \
        --parallel "$ACC_PARALLEL" \
        --num-shots "$ACC_NUM_SHOTS" \
        --host "$ACC_HOST" \
        --port "$ACC_PORT" \
        --result-file "$result_jsonl"
    ) 2>&1 | tee "$log"; then
      echo "WARN: GSM8K ${run_tag} (${run_idx}/${ACC_RUNS}) failed, continuing..." >&2
    fi
  done

  echo
  echo "Summarizing -> ${result_dir}/summary.txt"
  if ! python3 "$SUMMARIZE_ACC" \
      "$result_dir" "$TAG" "$MODEL" "$NODE" "$GPU" \
      "$ACC_NUM_SHOTS" "$ACC_NUM_QUESTIONS" "$ACC_PARALLEL" "$ACC_RUNS" \
      | tee "${result_dir}/summary.txt"; then
    echo "WARN: accuracy summary failed (see above); raw logs/jsonl are still in ${result_dir}" >&2
  fi
}

case "$BENCH_MODE" in
  perf) run_perf_sweep ;;
  acc)  run_accuracy ;;
  *)
    echo "ERROR: unknown --mode '$BENCH_MODE' (expected: perf | acc)" >&2
    exit 1
    ;;
esac
