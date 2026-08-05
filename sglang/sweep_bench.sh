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

Perf (--mode perf):
  --perf-io-pairs PAIRS    comma-separated input:output pairs  (default: 70000:300)
  --perf-concurrencies N   comma-separated max-concurrency values  (default: 4,8,16,32,64)
  --perf-range-ratio F     random length range ratio  (default: 0.8)
  --perf-prompts-per-conc N  num-prompts = concurrency * N  (default: 5)
  --perf-result-dir PATH   output directory  (default: perf_<model>_<gpu>_<node>_<ts>)

Accuracy (--mode acc):
  --acc-num-questions N    (default: 2000)
  --acc-parallel N         (default: 64)
  --acc-num-shots N        (default: 5)
  --acc-host HOST          server host  (default: 127.0.0.1)
  --acc-port PORT          server port  (default: 30000)
  --acc-runs N             repeat GSM8K eval N times  (default: 3)
  --acc-result-dir PATH    output directory  (default: acc_<model>_<gpu>_<node>_<ts>)

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

PERF_IO_PAIRS_STR="70000:300"
PERF_CONCURRENCIES_STR="4,8,16,32,64"
PERF_RANGE_RATIO="0.8"
PERF_PROMPTS_PER_CONC="5"
PERF_RESULT_DIR=""

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
    --perf-io-pairs)     PERF_IO_PAIRS_STR="$2"; shift 2 ;;
    --perf-concurrencies) PERF_CONCURRENCIES_STR="$2"; shift 2 ;;
    --perf-range-ratio)  PERF_RANGE_RATIO="$2"; shift 2 ;;
    --perf-prompts-per-conc) PERF_PROMPTS_PER_CONC="$2"; shift 2 ;;
    --perf-result-dir)   PERF_RESULT_DIR="$2"; shift 2 ;;
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

if [[ -z "$GPU_VENDOR" ]]; then
  GPU_VENDOR="$(detect_gpu_vendor)"
fi

bench_resolve_model
bench_detect_node_gpu

IFS=',' read -ra _perf_io_raw <<< "$PERF_IO_PAIRS_STR"
PERF_IO_PAIRS=("${_perf_io_raw[@]}")

IFS=',' read -ra PERF_CONCURRENCIES <<< "$PERF_CONCURRENCIES_STR"

_ts="$(date +%Y%m%d_%H%M%S)"
PERF_RESULT_DIR="${PERF_RESULT_DIR:-perf_${MODEL_KEY}_${GPU}_${NODE}_${_ts}}"
ACC_RESULT_DIR="${ACC_RESULT_DIR:-acc_${MODEL_KEY}_${GPU}_${NODE}_${_ts}}"

echo "Model: $MODEL  (MODEL_KEY=$MODEL_KEY, GPU_VENDOR=$GPU_VENDOR)"
[[ -n "${NUM_GPUS:-}" ]] && echo "Num GPUs (TP): $NUM_GPUS"
echo "Node: $NODE  GPU: $GPU  (detected: ${GPU_RAW:-none})"
echo "Mode: $BENCH_MODE"

run_perf_sweep() {
  local result_dir="$PERF_RESULT_DIR"
  mkdir -p "$result_dir"
  result_dir="$(cd "$result_dir" && pwd)"
  echo "Writing perf results to: $result_dir"

  run_one() {
    local ilen=$1 olen=$2 conc=$3
    local nprompts=$((conc * PERF_PROMPTS_PER_CONC))
    local tag="${ilen}_o${olen}_c${conc}"
    local jsonl="${result_dir}/${TAG}_${tag}.jsonl"
    local log="${result_dir}/${TAG}_${tag}.log"

    echo
    echo "==================================================================="
    echo ">>> input_len=${ilen}  output_len=${olen}  max_concurrency=${conc}  num_prompts=${nprompts}"
    echo "==================================================================="

    python3 -m sglang.bench_serving \
      --model "$MODEL" \
      --dataset-name random \
      --random-input "$ilen" \
      --random-output "$olen" \
      --random-range-ratio "$PERF_RANGE_RATIO" \
      --max-concurrency "$conc" \
      --num-prompts "$nprompts" \
      --output-file "$jsonl" \
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

  bench_package_results "$result_dir"
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

  bench_package_results "$result_dir"
}

case "$BENCH_MODE" in
  perf) run_perf_sweep ;;
  acc)  run_accuracy ;;
  *)
    echo "ERROR: unknown --mode '$BENCH_MODE' (expected: perf | acc)" >&2
    exit 1
    ;;
esac
