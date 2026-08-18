#!/usr/bin/env bash
#
# GLM env suite (sglang/suites/glm):
#   accuracy verify + perf sweep + 8k-only trace collect + trace analysis
#   for both non-MTP and MTP (EAGLE).
#
# Workload matrix (per mode nomtp|mtp):
#   1024:1024           perf only  (baseline server)
#   8192:1024           perf + DECODE trace + analyze  (baseline, conc 4-64)
#   70000:300           perf only  (server --max-running-requests)
#
# Order among selected stages: acc → perf → profile → analyze
# (default --phases acc,perf,profile,analyze).
#
# Usage:
#   bash run_env_suite.sh
#   bash run_env_suite.sh --phases profile,analyze --suite-dir ...
#   bash run_env_suite.sh --only-nomtp --dry-run
#
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "ERROR: run with bash, not sh:  bash $0 ..." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"          # sglang/
REPO_ROOT="$(cd "${SGLANG_DIR}/.." && pwd)"              # caro-playbox
LIB_DIR="${SGLANG_DIR}/lib"
SWEEP_BENCH="${LIB_DIR}/sweep_bench.sh"
RECIPE_GLM="${SGLANG_DIR}/recipes/glm.sh"
PROFILE_ANALYZER_ROOT_DEFAULT="${REPO_ROOT}/analysis"

# shellcheck source=../../lib/bench_common.sh
source "${LIB_DIR}/bench_common.sh"
# shellcheck source=../../recipes/glm.sh
source "$RECIPE_GLM"

usage() {
  cat <<'EOF'
Usage: run_env_suite.sh [options]

GLM suite stages (select with --phases; run in this order when included):
  [acc]     GSM8K accuracy (per mode)
  [perf]    nomtp then mtp: 1024:1024 + 8192:1024, then 70000:300
  [profile] cuda-graph ON DECODE for 8192:1024 at conc 4,8,16,32,64,
            then conc=4 --disable-cuda-graph (nomtp then mtp)
  [analyze] decode_profile Excel + hierarchy .txt under analyze/

IO matrix:
  1024:1024     perf only
  8192:1024     perf + trace + analyze
  70000:300     perf only (--max-running-requests)

Options:
  --model PATH                 model path  (default: vendor auto
                               amd/GLM-5.2-MXFP4 | nvidia/GLM-5.2-NVFP4)
  --model-key KEY              passed to sweep_bench  (default: glm)
  --gpu-vendor VENDOR          cuda | amd  (default: auto-detect)
  --gpus LIST                  CUDA_VISIBLE_DEVICES  (default: 4,5,6,7)
  --tp N                       tensor parallel size  (default: 4)
  --host HOST                  server host  (default: 127.0.0.1)
  --port PORT                  server port  (default: 30000)
  --ready-timeout SEC          wait for /v1/models  (default: 3600)
  --sglang-root PATH           sglang checkout for PYTHONPATH
                               (default: /sgl-workspace/sglang)
  --hf-home PATH               HF_HOME  (default: /data/huggingface or $HF_HOME)
  --short-io PAIRS             short-ctx perf IOs  (default: 1024:1024,8192:1024)
  --long-io PAIR               long-ctx perf IO    (default: 70000:300)
  --trace-io PAIR              DECODE profile IO   (default: 8192:1024)
  --long-max-running N         --max-running-requests for long-ctx  (default: 8)
  --phases LIST                stages to run, comma-separated
                               (default: acc,perf,profile,analyze)
                               e.g. --phases profile,analyze
  --skip-short-ctx             skip short-ctx perf (1024/8192) within perf
  --skip-long-ctx              skip 70000:300 perf within perf
  --profile-num-steps N        profile decode steps  (default: 20)
  --profile-concurrencies N    conc list for profile only
                               (default: 4,8,16,32,64)
  --profile-nocg-concurrencies N
                               conc for disable-cuda-graph profile
                               (default: 4)
  --nocg-profile               record conc4 --disable-cuda-graph (default)
  --skip-nocg-profile          skip conc4 --disable-cuda-graph profiles
  --profile-retries N          retries after fail / bad trace  (default: 2)
  --profile-min-steps N        min acceptable DECODE steps
                               (default: max(5, 75% of num-steps))
  --profile-max-wall-ms MS     max decode/phase wall ms/step for 8k
                               (default: 300; scaled by input length)
  --profile-watchdog-sec N     while recording, poll server health
                               (default: 10; 0=disable poll only)
  --profile-sweep-timeout-sec N
                               per-conc wall timeout; then kill client,
                               restart server, retry  (default: auto
                               max(600, conc*30), nocg 2x; 0=no wall
                               timeout)
  --profile-analyzer-root PATH (default: <repo>/analysis)
  --suite-dir PATH             reuse an existing suite dir (required for
                               --phases analyze alone; no new timestamp)
  --only-nomtp                 only nomtp (no EAGLE)
  --only-mtp                   only mtp / EAGLE
  --keep-server                do not stop server at the very end
  --dry-run                    print plan only (no suite dir / outputs)
  --extra-server-args "..."    extra args appended to every launch_server
  --extra-sweep-args "..."     extra args appended to every sweep_bench.sh
  -h, --help                   show this help

Environment (set before launch if unset):
  HF_HOME, SGLANG_ROCM_FUSED_DECODE_MLA=0, ROCM_QUICK_REDUCE_QUANTIZATION=INT4,
  SGLANG_OPT_USE_TOPK_V2=0, PYTHONPATH=<sglang>/python
EOF
}

MODEL=""                      # resolve after --gpu-vendor / auto-detect
MODEL_KEY="glm"
GPU_VENDOR=""                 # cuda | amd; empty -> auto-detect
GPUS="4,5,6,7"
TP="4"
HOST="127.0.0.1"
PORT="30000"
READY_TIMEOUT="3600"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
HF_HOME_DEFAULT="${HF_HOME:-/data/huggingface}"
SHORT_IO="1024:1024,8192:1024"
LONG_IO="70000:300"
TRACE_IO="8192:1024"
LONG_MAX_RUNNING="8"
PHASES="acc,perf,profile,analyze"
WANT_ACC=false
WANT_PERF=false
WANT_PROFILE=false
WANT_ANALYZE=false
SKIP_SHORT=false
SKIP_LONG=false
SKIP_NOCG_PROFILE=false
PROFILE_NUM_STEPS="20"
PROFILE_CONCURRENCIES="4,8,16,32,64"
PROFILE_NOCG_CONCURRENCIES="4"
PROFILE_RETRIES="2"
PROFILE_MIN_STEPS=""          # empty -> auto from PROFILE_NUM_STEPS
PROFILE_MAX_WALL_MS="300"     # short-ctx default; long IO scaled in helper
PROFILE_WATCHDOG_SEC="10"     # 0 = disable in-sweep health poll
PROFILE_SWEEP_TIMEOUT_SEC=""  # empty = auto; 0 = no wall timeout
PROFILE_ANALYZER_ROOT="$PROFILE_ANALYZER_ROOT_DEFAULT"
SUITE_DIR_OVERRIDE=""
ONLY_NOMTP=false
ONLY_MTP=false
KEEP_SERVER=false
DRY_RUN=false
EXTRA_SERVER_ARGS=""
EXTRA_SWEEP_ARGS=""

SERVER_PID=""
SERVER_LOG=""
SUITE_ROOT=""
SUITE_LOG=""
CURRENT_PHASE=""
CURRENT_MODE=""   # nomtp | mtp
PROFILE_FAILS=0
ANALYZE_FAILS=0
SWEEP_PID=""
declare -a CURRENT_SERVER_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)              MODEL="$2"; shift 2 ;;
    --model-key)          MODEL_KEY="$2"; shift 2 ;;
    --gpu-vendor)         GPU_VENDOR="$2"; shift 2 ;;
    --gpus)               GPUS="$2"; shift 2 ;;
    --tp)                 TP="$2"; shift 2 ;;
    --host)               HOST="$2"; shift 2 ;;
    --port)               PORT="$2"; shift 2 ;;
    --ready-timeout)      READY_TIMEOUT="$2"; shift 2 ;;
    --sglang-root)        SGLANG_ROOT="$2"; shift 2 ;;
    --hf-home)            HF_HOME_DEFAULT="$2"; shift 2 ;;
    --short-io)           SHORT_IO="$2"; shift 2 ;;
    --long-io)            LONG_IO="$2"; shift 2 ;;
    --trace-io)           TRACE_IO="$2"; shift 2 ;;
    --long-max-running)   LONG_MAX_RUNNING="$2"; shift 2 ;;
    --phases)             PHASES="$2"; shift 2 ;;
    --skip-short-ctx)     SKIP_SHORT=true; shift ;;
    --skip-long-ctx)      SKIP_LONG=true; shift ;;
    --skip-nocg-profile)  SKIP_NOCG_PROFILE=true; shift ;;
    --nocg-profile)       SKIP_NOCG_PROFILE=false; shift ;;
    --profile-num-steps)  PROFILE_NUM_STEPS="$2"; shift 2 ;;
    --profile-concurrencies) PROFILE_CONCURRENCIES="$2"; shift 2 ;;
    --profile-nocg-concurrencies) PROFILE_NOCG_CONCURRENCIES="$2"; shift 2 ;;
    --profile-retries)    PROFILE_RETRIES="$2"; shift 2 ;;
    --profile-min-steps)  PROFILE_MIN_STEPS="$2"; shift 2 ;;
    --profile-max-wall-ms) PROFILE_MAX_WALL_MS="$2"; shift 2 ;;
    --profile-watchdog-sec) PROFILE_WATCHDOG_SEC="$2"; shift 2 ;;
    --profile-sweep-timeout-sec) PROFILE_SWEEP_TIMEOUT_SEC="$2"; shift 2 ;;
    --profile-analyzer-root) PROFILE_ANALYZER_ROOT="$2"; shift 2 ;;
    --suite-dir)          SUITE_DIR_OVERRIDE="$2"; shift 2 ;;
    --from-phase|--skip-acc|--skip-perf|--skip-profile|--skip-analyze)
      echo "ERROR: $1 removed; use --phases acc,perf,profile,analyze" >&2
      exit 1
      ;;
    --only-nomtp)         ONLY_NOMTP=true; shift ;;
    --only-mtp)           ONLY_MTP=true; shift ;;
    --keep-server)        KEEP_SERVER=true; shift ;;
    --dry-run)            DRY_RUN=true; shift ;;
    --extra-server-args)  EXTRA_SERVER_ARGS="$2"; shift 2 ;;
    --extra-sweep-args)   EXTRA_SWEEP_ARGS="$2"; shift 2 ;;
    -h|--help)            usage; exit 0 ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

parse_phases() {
  local raw="$1"
  local -a parts=()
  local p
  WANT_ACC=false
  WANT_PERF=false
  WANT_PROFILE=false
  WANT_ANALYZE=false
  IFS=',' read -ra parts <<< "$raw"
  for p in "${parts[@]}"; do
    p="${p// /}"
    p="${p,,}"
    case "$p" in
      "") ;;
      acc) WANT_ACC=true ;;
      perf) WANT_PERF=true ;;
      profile) WANT_PROFILE=true ;;
      analyze) WANT_ANALYZE=true ;;
      *)
        echo "ERROR: unknown phase '${p}' in --phases (want: acc,perf,profile,analyze)" >&2
        exit 1
        ;;
    esac
  done
  if [[ "$WANT_ACC" != true && "$WANT_PERF" != true && "$WANT_PROFILE" != true && "$WANT_ANALYZE" != true ]]; then
    echo "ERROR: --phases must include at least one of: acc,perf,profile,analyze" >&2
    exit 1
  fi
}

parse_phases "$PHASES"

if [[ "$ONLY_NOMTP" == true && "$ONLY_MTP" == true ]]; then
  echo "ERROR: --only-nomtp and --only-mtp are mutually exclusive" >&2
  exit 1
fi

# analyze-only reuses existing traces; never mint an empty suite_glm_env_<ts>/.
if [[ "$WANT_ANALYZE" == true && "$WANT_ACC" != true && "$WANT_PERF" != true && "$WANT_PROFILE" != true ]]; then
  if [[ -z "$SUITE_DIR_OVERRIDE" ]]; then
    echo "ERROR: --phases analyze requires --suite-dir <existing suite_glm_env_*>" >&2
    echo "  example: bash run_env_suite.sh --suite-dir suite_glm_env_20260818_115622 --phases analyze" >&2
    exit 1
  fi
fi

# Resolve GPU vendor + default model (AMD MXFP4 / NVIDIA NVFP4).
if [[ -z "$GPU_VENDOR" ]]; then
  GPU_VENDOR="$(detect_gpu_vendor)"
fi
case "$GPU_VENDOR" in
  cuda|amd) ;;
  nv|nvidia) GPU_VENDOR="cuda" ;;
  rocm) GPU_VENDOR="amd" ;;
  *)
    echo "ERROR: unknown or undetected GPU_VENDOR='${GPU_VENDOR}' (set --gpu-vendor cuda|amd)" >&2
    exit 1
    ;;
esac
if [[ -n "$MODEL" ]]; then
  MODEL_OVERRIDE="$MODEL"
fi
bench_resolve_model

if [[ -n "$SUITE_DIR_OVERRIDE" && ! -d "$SUITE_DIR_OVERRIDE" ]]; then
  echo "ERROR: --suite-dir does not exist: ${SUITE_DIR_OVERRIDE}" >&2
  exit 1
fi
if [[ ! -x "$SWEEP_BENCH" && ! -f "$SWEEP_BENCH" ]]; then
  echo "ERROR: missing ${SWEEP_BENCH}" >&2
  exit 1
fi
if ! [[ "$PROFILE_NUM_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --profile-num-steps must be a positive integer" >&2
  exit 1
fi
if ! [[ "$PROFILE_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --profile-retries must be a non-negative integer" >&2
  exit 1
fi
if [[ -n "$PROFILE_MIN_STEPS" ]] && ! [[ "$PROFILE_MIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --profile-min-steps must be a positive integer" >&2
  exit 1
fi
if ! [[ "$PROFILE_MAX_WALL_MS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: --profile-max-wall-ms must be a positive number" >&2
  exit 1
fi
if ! [[ "$PROFILE_WATCHDOG_SEC" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --profile-watchdog-sec must be a non-negative integer" >&2
  exit 1
fi
if [[ -n "$PROFILE_SWEEP_TIMEOUT_SEC" ]] && ! [[ "$PROFILE_SWEEP_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --profile-sweep-timeout-sec must be a non-negative integer" >&2
  exit 1
fi
# Default min steps: 75% of requested profile steps (at least 5).
if [[ -z "$PROFILE_MIN_STEPS" ]]; then
  PROFILE_MIN_STEPS=$(( (PROFILE_NUM_STEPS * 75 + 99) / 100 ))
  if (( PROFILE_MIN_STEPS < 5 )); then
    PROFILE_MIN_STEPS=5
  fi
  if (( PROFILE_MIN_STEPS > PROFILE_NUM_STEPS )); then
    PROFILE_MIN_STEPS="$PROFILE_NUM_STEPS"
  fi
fi

die() { echo "ERROR: $*" >&2; exit 1; }

log() {
  local msg="[$(date '+%F %T')] $*"
  echo "$msg"
  if [[ -n "${SUITE_LOG:-}" ]]; then
    echo "$msg" >> "$SUITE_LOG"
  fi
}

run_cmd() {
  log "+ $*"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  "$@"
}

setup_env() {
  export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
  case "$GPU_VENDOR" in
    amd)
      export SGLANG_ROCM_FUSED_DECODE_MLA="${SGLANG_ROCM_FUSED_DECODE_MLA:-0}"
      export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
      export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
      ;;
  esac
}

base_server_args() {
  glm_base_server_args
}

eagle_server_args() {
  glm_eagle_server_args
}

# Kill listeners on $PORT only (never host-wide pkill of all SGLang).
_pids_listening_on_port() {
  ss -lntp 2>/dev/null | grep -E ":${PORT}\\b" | grep -oE 'pid=[0-9]+' \
    | cut -d= -f2 | sort -u || true
}

_kill_pids() {
  local sig="$1"
  shift
  local pid
  for pid in "$@"; do
    [[ -z "$pid" ]] && continue
    kill "$sig" "$pid" 2>/dev/null || true
  done
}

kill_existing_server() {
  log "Stopping SGLang server on :${PORT} (tracked PID=${SERVER_PID:-none})"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  local -a pids=()
  local pid

  if [[ -n "${SERVER_PID:-}" ]]; then
    pids+=("$SERVER_PID")
    # Child workers often share the process group of the launch_server parent.
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  fi

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    pids+=("$pid")
  done < <(_pids_listening_on_port)

  if ((${#pids[@]})); then
    _kill_pids -TERM "${pids[@]}"
    sleep 2
    _kill_pids -KILL "${pids[@]}"
    # Re-check port in case workers respawned under new PIDs.
    local -a leftover=()
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      leftover+=("$pid")
    done < <(_pids_listening_on_port)
    if ((${#leftover[@]})); then
      _kill_pids -KILL "${leftover[@]}"
    fi
  fi
  SERVER_PID=""

  local i
  for i in $(seq 1 30); do
    if ! ss -lntp 2>/dev/null | grep -qE ":${PORT}\\b"; then
      return 0
    fi
    sleep 1
  done
  die "port ${PORT} still busy after stop"
}

# Marker printed by launch_server once the new process is fully serving.
SERVER_READY_MARKER="The server is fired up and ready to roll!"

wait_server_ready() {
  local url="http://${HOST}:${PORT}/v1/models"
  local start now elapsed
  start="$(date +%s)"
  log "Waiting for server ready: ${url} (timeout ${READY_TIMEOUT}s)"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  # Gate on the fresh $SERVER_LOG marker (truncated each start_server via '>'),
  # not just HTTP: a lingering/old endpoint can answer /v1/models before the
  # new process has loaded, producing a false "ready" right after a restart.
  while true; do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      die "server process exited early; see ${SERVER_LOG}"
    fi
    if [[ -f "$SERVER_LOG" ]] && grep -qF "$SERVER_READY_MARKER" "$SERVER_LOG"; then
      if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
        log "Server is ready"
        return 0
      fi
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed >= READY_TIMEOUT )); then
      die "server not ready after ${READY_TIMEOUT}s; see ${SERVER_LOG}"
    fi
    sleep 5
  done
}

# Allocate server_logs/{phase}.server.log or {phase}.server.N.log (preserve prior runs).
allocate_server_log() {
  local phase_name="$1"
  local dir="${SUITE_ROOT}/server_logs"
  local base n
  mkdir -p "$dir"
  base="${dir}/${phase_name}.server.log"
  if [[ ! -e "$base" ]]; then
    echo "$base"
    return 0
  fi
  n=1
  while [[ -e "${dir}/${phase_name}.server.${n}.log" ]]; do
    n=$((n + 1))
  done
  echo "${dir}/${phase_name}.server.${n}.log"
}

start_server() {
  local phase_name="$1"
  shift
  local -a extra=("$@")
  local -a args=()
  local line extra_summary=""

  CURRENT_PHASE="$phase_name"
  CURRENT_SERVER_EXTRA=("${extra[@]}")

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # shellcheck disable=SC2206
    args+=($line)
  done < <(base_server_args)

  if ((${#extra[@]})); then
    args+=("${extra[@]}")
    extra_summary=" extra=[${extra[*]}]"
  fi
  if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
    # shellcheck disable=SC2206
    args+=($EXTRA_SERVER_ARGS)
    extra_summary="${extra_summary} EXTRA_SERVER_ARGS"
  fi

  SERVER_LOG="$(allocate_server_log "$phase_name")"
  log "Launching server [${phase_name}] -> ${SERVER_LOG}"
  log "Args: model=${MODEL} tp=${TP}${extra_summary} (full cmdline in server log header)"

  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  kill_existing_server
  {
    echo "=== launch $(date '+%F %T') phase=${phase_name} ==="
    echo "cmdline: python3 -m sglang.launch_server ${args[*]}"
    echo
  } >"$SERVER_LOG"
  # shellcheck disable=SC2086
  python3 -m sglang.launch_server "${args[@]}" >>"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  log "Server PID=${SERVER_PID}"
  wait_server_ready
}

restart_current_server() {
  if [[ -z "${CURRENT_PHASE:-}" ]]; then
    die "restart_current_server: no current phase recorded"
  fi
  log "Restarting server for phase=${CURRENT_PHASE} (after profile/server failure)"
  start_server "$CURRENT_PHASE" "${CURRENT_SERVER_EXTRA[@]}"
}

# Count live (non-defunct) sglang scheduler processes.
_scheduler_count() {
  pgrep -af 'sglang::scheduler' 2>/dev/null | grep -vc '<defunct>' || true
}

# True if /v1/models works, parent alive, and >= TP schedulers (not zombie-only).
server_is_healthy() {
  local n_sched
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  if ! curl -fsS -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    return 1
  fi
  if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    return 1
  fi
  n_sched="$(_scheduler_count)"
  if ! [[ "$n_sched" =~ ^[0-9]+$ ]] || (( n_sched < TP )); then
    return 1
  fi
  if [[ -n "${SERVER_LOG:-}" && -f "$SERVER_LOG" ]]; then
    if grep -qE 'Scheduler watchdog timeout|Scheduler hit an exception' "$SERVER_LOG" 2>/dev/null; then
      # Fresh errors near end of log: check last 80 lines
      if tail -n 80 "$SERVER_LOG" | grep -qE 'Scheduler watchdog timeout|Scheduler hit an exception'; then
        return 1
      fi
    fi
  fi
  return 0
}

# Stronger check used after profile (generate can hang on half-dead servers).
server_smoke_ok() {
  server_is_healthy || return 1
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  curl -fsS -m 90 -X POST "http://${HOST}:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text":"ok","sampling_params":{"max_new_tokens":4,"temperature":0}}' \
    >/dev/null 2>&1
}

# Optional conc: all stage traces (*DECODE* / *EXTEND* / …) matching _cN.
_profile_trace_list() {
  local dir="$1"
  local conc="${2:-}"
  [[ -d "$dir" ]] || return 0
  if [[ -n "$conc" ]]; then
    find "$dir" -type f -name '*.trace.json.gz' 2>/dev/null \
      | grep -E "_c${conc}[-_.]" || true
  else
    find "$dir" -type f -name '*.trace.json.gz' 2>/dev/null || true
  fi
}

_decode_trace_list() {
  local dir="$1"
  local conc="${2:-}"
  [[ -d "$dir" ]] || return 0
  if [[ -n "$conc" ]]; then
    find "$dir" -type f -name '*DECODE.trace.json.gz' 2>/dev/null \
      | grep -E "_c${conc}[-_.]" || true
  else
    find "$dir" -type f -name '*DECODE.trace.json.gz' 2>/dev/null || true
  fi
}

count_decode_traces() {
  local dir="$1"
  local conc="${2:-}"
  _decode_trace_list "$dir" "$conc" | grep -c . || true
}

clear_decode_traces() {
  local dir="$1"
  local conc="${2:-}"
  local f parent
  [[ -d "$dir" ]] || return 0
  # Clear all stages for this conc (DECODE/EXTEND/…), not only DECODE.
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    parent="$(dirname "$f")"
    rm -f "$f"
    # Drop empty timestamp dirs left behind after deleting files.
    if [[ -d "$parent" && "$parent" != "$dir" ]]; then
      rmdir "$parent" 2>/dev/null || true
    fi
  done < <(_profile_trace_list "$dir" "$conc")
  # Also drop empty timestamp dirs from failed attempts that never wrote
  # any *.trace.json.gz (segfault during /start_profile leaves mkdir-only shells).
  find "$dir" -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null || true
}

io_tag_from_pair() {
  local io="$1"
  local in="${io%%:*}"
  local out="${io##*:}"
  echo "i${in}_o${out}"
}

# Scale wall-time ceiling with context length (70k attention is much heavier).
profile_max_wall_for_io() {
  local io="$1"
  local in="${io%%:*}"
  local base="$PROFILE_MAX_WALL_MS"
  # bash arithmetic is integer; keep ms as integers for comparison defaults.
  local base_i="${base%%.*}"
  [[ -z "$base_i" ]] && base_i=300
  if (( in >= 32768 )); then
    echo $(( base_i * 8 ))
  elif (( in >= 4096 )); then
    echo $(( base_i * 2 ))
  else
    echo "$base_i"
  fi
}

# Validate DECODE traces: step count + wall/phase times via decode_profile.single.
# Optional conc: only check that concurrency (avoids stale traces under --suite-dir).
validate_profile_dir() {
  local pdir="$1"
  local io="$2"
  local mode="$3"
  local conc="${4:-}"
  local max_wall
  max_wall="$(profile_max_wall_for_io "$io")"
  local analyzer_root="$PROFILE_ANALYZER_ROOT"
  local rules="${analyzer_root}/rules/glm52.csv"
  local py_path="${analyzer_root}${PYTHONPATH:+:${PYTHONPATH}}"
  local vrc errexit_was=0
  local -a conc_args=()

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN validate: dir=${pdir} mode=${mode} conc=${conc:-all} min_steps=${PROFILE_MIN_STEPS} max_wall_ms=${max_wall}"
    return 0
  fi
  if [[ ! -d "$analyzer_root/decode_profile" ]]; then
    log "WARN: analysis tree missing; skip trace validation"
    return 0
  fi
  if [[ -n "$conc" ]]; then
    conc_args=(--conc "$conc")
  fi

  [[ $- == *e* ]] && errexit_was=1
  set +e
  (
    cd "$analyzer_root"
    PYTHONPATH="$py_path" python3 -m decode_profile.single \
      --dir "$pdir" \
      --label "$mode" \
      --rules "$rules" \
      --validate \
      --expected-steps "$PROFILE_NUM_STEPS" \
      --min-steps "$PROFILE_MIN_STEPS" \
      --max-wall-ms "$max_wall" \
      --mode "$mode" \
      "${conc_args[@]}"
  )
  vrc=$?
  (( errexit_was )) && set -e
  return "$vrc"
}

run_sweep() {
  local mode="$1"
  shift
  local -a sweep_args=(
    --model-key "$MODEL_KEY"
    --mode "$mode"
    --model-override "$MODEL"
    --gpu-vendor "$GPU_VENDOR"
    --server-host "$HOST"
    --server-port "$PORT"
    --sglang-root "$SGLANG_ROOT"
    --num-gpus "$TP"
  )
  if [[ "$CURRENT_MODE" == "mtp" ]]; then
    sweep_args+=(--mtp)
  elif [[ "$CURRENT_MODE" == "nomtp" ]]; then
    sweep_args+=(--no-mtp)
  fi
  if [[ "$mode" == "acc" ]]; then
    sweep_args+=(--acc-host "$HOST" --acc-port "$PORT")
  fi
  if ((${#@})); then
    sweep_args+=("$@")
  fi
  if [[ -n "$EXTRA_SWEEP_ARGS" ]]; then
    # shellcheck disable=SC2206
    sweep_args+=($EXTRA_SWEEP_ARGS)
  fi

  log "Running: bash sweep_bench.sh ${sweep_args[*]}"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  (
    cd "$SCRIPT_DIR"
    bash "$SWEEP_BENCH" "${sweep_args[@]}"
  )
}

# Kill pid and descendants; skip the tracked SGLang server pid.
_kill_pid_tree() {
  local pid="$1"
  local sig="${2:--TERM}"
  local child
  [[ -n "$pid" ]] || return 0
  while IFS= read -r child; do
    [[ -z "$child" ]] && continue
    [[ -n "${SERVER_PID:-}" && "$child" == "$SERVER_PID" ]] && continue
    _kill_pid_tree "$child" "$sig"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill "$sig" "$pid" 2>/dev/null || true
}

stop_sweep_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  log "Stopping profile sweep pid=${pid} and children"
  _kill_pid_tree "$pid" -TERM
  local i
  for i in $(seq 1 8); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  log "Sweep pid=${pid} still alive; SIGKILL tree"
  _kill_pid_tree "$pid" -KILL
}

# Auto wall timeout: max(600, conc*30) for cg-on; nocg uses 2x (slower).
# Explicit --profile-sweep-timeout-sec overrides both. 0 = off.
profile_sweep_timeout_for_conc() {
  local conc="$1"
  local which="${2:-trace}"
  if [[ -n "$PROFILE_SWEEP_TIMEOUT_SEC" ]]; then
    echo "$PROFILE_SWEEP_TIMEOUT_SEC"
    return 0
  fi
  local t=$((conc * 30))
  if (( t < 600 )); then
    t=600
  fi
  if [[ "$which" == "nocg" ]]; then
    t=$((t * 2))
  fi
  echo "$t"
}

# Run sweep in background; abort if server dies or wall timeout hits.
# rc 124 = wall timeout, 125 = server unhealthy mid-sweep.
# PROFILE_WATCHDOG_SEC=0 disables health polls only; wall timeout still applies.
run_sweep_watched() {
  local timeout_sec="$1"
  shift

  if [[ "$DRY_RUN" == true ]]; then
    run_sweep "$@"
    return
  fi

  # No poll and no wall timeout -> plain foreground sweep.
  if [[ "$PROFILE_WATCHDOG_SEC" == "0" && ( "$timeout_sec" == "0" || -z "$timeout_sec" ) ]]; then
    run_sweep "$@"
    return
  fi

  run_sweep "$@" &
  local spid=$!
  SWEEP_PID="$spid"
  local start now elapsed next_check
  start="$(date +%s)"
  if [[ "$PROFILE_WATCHDOG_SEC" != "0" ]]; then
    next_check=$((start + PROFILE_WATCHDOG_SEC))
  else
    next_check=0
  fi

  while kill -0 "$spid" 2>/dev/null; do
    sleep 1
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( timeout_sec > 0 && elapsed >= timeout_sec )); then
      log "WATCHDOG: sweep pid=${spid} exceeded ${timeout_sec}s wall timeout"
      stop_sweep_pid "$spid"
      wait "$spid" 2>/dev/null || true
      SWEEP_PID=""
      return 124
    fi
    if (( next_check > 0 && now >= next_check )); then
      next_check=$((now + PROFILE_WATCHDOG_SEC))
      if ! server_is_healthy; then
        log "WATCHDOG: server unhealthy during sweep pid=${spid} after ${elapsed}s"
        stop_sweep_pid "$spid"
        wait "$spid" 2>/dev/null || true
        SWEEP_PID=""
        return 125
      fi
    fi
  done
  local rc=0
  wait "$spid" || rc=$?
  SWEEP_PID=""
  return "$rc"
}

run_acc_if_needed() {
  if [[ "$WANT_ACC" != true ]]; then
    log "Skip acc (not in --phases)"
    return 0
  fi
  mkdir -p "${SUITE_ROOT}/acc/${CURRENT_MODE}"
  run_sweep acc --acc-result-dir "${SUITE_ROOT}/acc/${CURRENT_MODE}"
}

# The baseline (no --max-running-requests) server only serves acc + short-ctx
# perf. Skip launching it entirely when neither is requested (e.g. resuming a
# suite with only the long-ctx 70000:300 sweep left).
baseline_server_needed() {
  if [[ "$WANT_ACC" == true ]]; then
    return 0
  fi
  if [[ "$WANT_PERF" == true && "$SKIP_SHORT" != true ]]; then
    return 0
  fi
  return 1
}

run_short_perf_if_needed() {
  if [[ "$WANT_PERF" != true || "$SKIP_SHORT" == true ]]; then
    log "Skip short-ctx perf (${SHORT_IO})"
    return 0
  fi
  mkdir -p "${SUITE_ROOT}/perf/${CURRENT_MODE}"
  run_sweep perf \
    --perf-io-pairs "$SHORT_IO" \
    --perf-result-dir "${SUITE_ROOT}/perf/${CURRENT_MODE}"
}

run_long_perf_if_needed() {
  if [[ "$WANT_PERF" != true || "$SKIP_LONG" == true ]]; then
    log "Skip long-ctx perf (${LONG_IO})"
    return 0
  fi
  # Same dir as short: summarize_perf globs all glm_{mode}_*.jsonl into one summary.
  mkdir -p "${SUITE_ROOT}/perf/${CURRENT_MODE}"
  run_sweep perf \
    --perf-io-pairs "$LONG_IO" \
    --perf-result-dir "${SUITE_ROOT}/perf/${CURRENT_MODE}"
}

# One IO pair + one concurrency via sweep_bench --profile; restart server on
# failure or when DECODE trace step/wall sanity checks fail.
# Optional: conc override and tag_suffix (e.g. _c4_nocg for op-compare traces).
# which=nocg applies 2x auto wall timeout. Only this conc's traces are cleared.
run_one_profile() {
  local io="$1"
  local conc="${2:-$PROFILE_CONCURRENCIES}"
  local tag_suffix="${3:-}"
  local which="${4:-trace}"
  local tag
  tag="$(io_tag_from_pair "$io")${tag_suffix}"
  local pdir="${SUITE_ROOT}/profiles/${CURRENT_MODE}/${tag}"
  local stub="${SUITE_ROOT}/profile_logs/${CURRENT_MODE}/${tag}/c${conc}"
  local attempts=$((PROFILE_RETRIES + 1))
  local attempt rc ntraces max_wall sweep_timeout profile_olen

  max_wall="$(profile_max_wall_for_io "$io")"
  sweep_timeout="$(profile_sweep_timeout_for_conc "$conc" "$which")"
  if [[ "$CURRENT_MODE" == "mtp" ]]; then
    profile_olen=128
  else
    profile_olen=64
  fi
  log "===== PROFILE mode=${CURRENT_MODE} io=${io} conc=${conc} dir=${pdir} ====="

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN profile: stub=${stub} traces=${pdir} --perf-io-pairs ${io} --perf-concurrencies ${conc} --profile (actual output_len=${profile_olen})"
    return 0
  fi

  mkdir -p "$pdir" "$stub"

  for attempt in $(seq 1 "$attempts"); do
    if ! server_is_healthy; then
      log "WARN: server unhealthy before profile attempt ${attempt}/${attempts}"
      restart_current_server
    fi

    # Drop prior bad/partial traces for this conc so retries do not pick them up.
    clear_decode_traces "$pdir" "$conc"

    set +e
    run_sweep_watched "$sweep_timeout" perf \
      --perf-io-pairs "$io" \
      --perf-concurrencies "$conc" \
      --profile \
      --profile-output-dir "$pdir" \
      --profile-num-steps "$PROFILE_NUM_STEPS" \
      --perf-result-dir "$stub"
    rc=$?
    # Keep errexit off until this function returns. Re-enabling set -e here made
    # a later `return 1` abort the whole suite (bash errexit + non-zero return),
    # skipping mtp/analyze. Caller wraps us in set +e / set -e.

    ntraces="$(count_decode_traces "$pdir" "$conc")"
    log "Profile attempt ${attempt}/${attempts}: sweep_rc=${rc} decode_traces=${ntraces} conc=${conc}"

    if (( rc == 124 || rc == 125 )); then
      if (( rc == 124 )); then
        log "WARN: profile wall timeout for ${CURRENT_MODE}/${tag} conc=${conc} (attempt ${attempt}/${attempts})"
      else
        log "WARN: server died during profile for ${CURRENT_MODE}/${tag} conc=${conc} (attempt ${attempt}/${attempts})"
      fi
      clear_decode_traces "$pdir" "$conc"
      if (( attempt < attempts )); then
        restart_current_server
      fi
      continue
    fi

    if (( ntraces <= 0 )); then
      log "WARN: no DECODE traces for ${CURRENT_MODE}/${tag} (attempt ${attempt}/${attempts})"
      clear_decode_traces "$pdir" "$conc"
      if (( attempt < attempts )); then
        restart_current_server
      fi
      continue
    fi

    if ! validate_profile_dir "$pdir" "$io" "$CURRENT_MODE" "$conc"; then
      log "WARN: trace sanity check failed for ${CURRENT_MODE}/${tag} conc=${conc} (attempt ${attempt}/${attempts})"
      # Keep the last rejected traces for post-mortem; only clear before a retry.
      if (( attempt < attempts )); then
        clear_decode_traces "$pdir" "$conc"
        restart_current_server
      fi
      continue
    fi

    # Traces are good; do not cold-restart just because smoke fails (next conc
    # still checks health before starting). Avoids useless reboot after last nocg.
    if ! server_smoke_ok; then
      log "WARN: server unhealthy after profile (traces kept=${ntraces}); not restarting"
    fi
    log "Profile OK for ${CURRENT_MODE}/${tag} (decode_traces=${ntraces})"
    return 0
  done

  log "ERROR: profile failed for ${CURRENT_MODE}/${tag} after ${attempts} attempt(s)"
  return 1
}

run_profile_ios_if_needed() {
  local ios="$1"
  local which="$2"   # short|long|trace|nocg
  local conc="${3:-$PROFILE_CONCURRENCIES}"
  local tag_suffix="${4:-}"
  if [[ "$WANT_PROFILE" != true ]]; then
    log "Skip profile (${which})"
    return 0
  fi
  if [[ "$which" == "short" && "$SKIP_SHORT" == true ]]; then
    log "Skip short-ctx profile (${ios})"
    return 0
  fi
  if [[ "$which" == "long" && "$SKIP_LONG" == true ]]; then
    log "Skip long-ctx profile (${ios})"
    return 0
  fi
  if [[ "$which" == "nocg" && "$SKIP_NOCG_PROFILE" == true ]]; then
    log "Skip nocg profile (${ios})"
    return 0
  fi

  local -a pairs=() concs=()
  local io c prc
  IFS=',' read -ra pairs <<< "$ios"
  IFS=',' read -ra concs <<< "$conc"
  for io in "${pairs[@]}"; do
    io="${io// /}"
    [[ -z "$io" ]] && continue
    for c in "${concs[@]}"; do
      c="${c// /}"
      [[ -z "$c" ]] && continue
      # Do not abort the whole suite if one shape/conc profile fails.
      set +e
      if [[ "$which" == "nocg" ]]; then
        # Per-conc suffix so --profile-nocg-concurrencies 4,8 -> _c4_nocg, _c8_nocg
        run_one_profile "$io" "$c" "_c${c}_nocg" nocg
      else
        run_one_profile "$io" "$c" "$tag_suffix" "$which"
      fi
      prc=$?
      set -e
      if (( prc != 0 )); then
        PROFILE_FAILS=$((PROFILE_FAILS + 1))
      fi
    done
  done
}

# DECODE for TRACE_IO (default 8k) at PROFILE_CONCURRENCIES (default 4,8,16,32,64),
# then conc4 --disable-cuda-graph (skip with --skip-nocg-profile).
run_all_profiles_after_perf() {
  if [[ "$WANT_PROFILE" != true ]]; then
    log "Skip all profiles (not in --phases)"
    return 0
  fi

  local max_wall profile_olen_note
  max_wall="$(profile_max_wall_for_io "$TRACE_IO")"
  profile_olen_note="nomtp=64 mtp=128"
  log "===== PROFILE PHASE (TRACE_IO=${TRACE_IO}) ====="
  log "Trace checks: expected_steps=${PROFILE_NUM_STEPS} min_steps=${PROFILE_MIN_STEPS} max_wall_ms=${max_wall} (scaled by IO) watchdog=${PROFILE_WATCHDOG_SEC}s profile_output_len=${profile_olen_note}"
  log "Watchdog: poll=${PROFILE_WATCHDOG_SEC}s wall_timeout=auto max(600,conc*30) (nocg 2x)"

  local -a eagle_args=()
  local line
  if [[ "$ONLY_NOMTP" != true ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      # shellcheck disable=SC2206
      eagle_args+=($line)
    done < <(eagle_server_args)
  fi

  if [[ "$ONLY_MTP" != true ]]; then
    CURRENT_MODE="nomtp"
    log "===== PROFILE nomtp / TRACE_IO=${TRACE_IO} (cuda-graph ON, conc=${PROFILE_CONCURRENCIES}) ====="
    start_server "nomtp_profile"
    run_profile_ios_if_needed "$TRACE_IO" trace

    if [[ "$SKIP_NOCG_PROFILE" == true ]]; then
      log "Skip nomtp nocg profile (--skip-nocg-profile)"
    else
      log "===== PROFILE nomtp / TRACE_IO=${TRACE_IO} (disable-cuda-graph, conc=${PROFILE_NOCG_CONCURRENCIES}) ====="
      start_server "nomtp_profile_nocg" --disable-cuda-graph
      run_profile_ios_if_needed "$TRACE_IO" nocg "$PROFILE_NOCG_CONCURRENCIES"
    fi
  fi

  if [[ "$ONLY_NOMTP" != true ]]; then
    CURRENT_MODE="mtp"
    log "===== PROFILE mtp / TRACE_IO=${TRACE_IO} (cuda-graph ON, conc=${PROFILE_CONCURRENCIES}) ====="
    start_server "mtp_profile" "${eagle_args[@]}"
    run_profile_ios_if_needed "$TRACE_IO" trace

    if [[ "$SKIP_NOCG_PROFILE" == true ]]; then
      log "Skip mtp nocg profile (--skip-nocg-profile)"
    else
      log "===== PROFILE mtp / TRACE_IO=${TRACE_IO} (disable-cuda-graph, conc=${PROFILE_NOCG_CONCURRENCIES}) ====="
      start_server "mtp_profile_nocg" "${eagle_args[@]}" --disable-cuda-graph
      run_profile_ios_if_needed "$TRACE_IO" nocg "$PROFILE_NOCG_CONCURRENCIES"
    fi
  fi
}

analyze_all_profiles() {
  if [[ "$WANT_ANALYZE" != true ]]; then
    log "Skip analysis (not in --phases)"
    return 0
  fi

  local analyzer_root="$PROFILE_ANALYZER_ROOT"
  local out_dir="${SUITE_ROOT}/analyze"
  local rules="${analyzer_root}/rules/glm52.csv"
  mkdir -p "$out_dir"

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN: would single-side analyze each profiles/{nomtp,mtp}/<io> -> ${out_dir}/*.xlsx + *.txt"
    return 0
  fi
  if [[ ! -d "$analyzer_root/decode_profile" ]]; then
    log "WARN: analysis not found at ${analyzer_root}; skip analyze"
    return 0
  fi

  log "===== PROFILE ANALYZE (single-side hierarchical) ====="
  log "Layers: decode wall/kernel -> phases -> kernel categories -> top kernels"
  log "MTP phases expected: draft / target_verify / draft_extend"

  local py_path="${analyzer_root}${PYTHONPATH:+:${PYTHONPATH}}"
  local mode_dir mode tag_dir tag ntraces out_xlsx out_txt

  shopt -s nullglob
  for mode_dir in "${SUITE_ROOT}/profiles/"*; do
    [[ -d "$mode_dir" ]] || continue
    mode="$(basename "$mode_dir")"
    for tag_dir in "${mode_dir}/"*; do
      [[ -d "$tag_dir" ]] || continue
      tag="$(basename "$tag_dir")"
      ntraces="$(count_decode_traces "$tag_dir")"
      if (( ntraces <= 0 )); then
        log "Skip ${mode}/${tag}: no DECODE traces"
        continue
      fi
      out_xlsx="${out_dir}/${mode}_${tag}.xlsx"
      out_txt="${out_dir}/${mode}_${tag}.txt"
      log "Analyze ${mode}/${tag} (decode_traces=${ntraces}) -> ${out_xlsx} + ${out_txt}"
      set +e
      (
        cd "$analyzer_root"
        PYTHONPATH="$py_path" python3 -m decode_profile.single \
          --dir "$tag_dir" \
          --label "$mode" \
          --rules "$rules" \
          -o "$out_xlsx"
      ) 2>&1 | tee "$out_txt"
      local arc
      arc=${PIPESTATUS[0]}
      set -e
      if (( arc != 0 )); then
        log "WARN: analyze failed for ${mode}/${tag} (rc=${arc})"
        ANALYZE_FAILS=$((ANALYZE_FAILS + 1))
      fi
    done
  done
  shopt -u nullglob

  log "Analyze outputs under: ${out_dir}"
}

write_manifest() {
  local mf="${SUITE_ROOT}/manifest.txt"
  {
    echo "suite_root=${SUITE_ROOT}"
    echo "started_or_resumed=$(date '+%F %T')"
    echo "phases=${PHASES}"
    echo "want_acc=${WANT_ACC} want_perf=${WANT_PERF} want_profile=${WANT_PROFILE} want_analyze=${WANT_ANALYZE}"
    echo "model=${MODEL}"
    echo "model_key=${MODEL_KEY}"
    echo "gpu_vendor=${GPU_VENDOR}"
    echo "gpus=${GPUS}"
    echo "tp=${TP}"
    echo "host=${HOST}:${PORT}"
    echo "short_io=${SHORT_IO}"
    echo "long_io=${LONG_IO}"
    echo "profile_num_steps=${PROFILE_NUM_STEPS}"
    echo "profile_min_steps=${PROFILE_MIN_STEPS}"
    echo "profile_max_wall_ms=${PROFILE_MAX_WALL_MS}"
    echo "profile_watchdog_sec=${PROFILE_WATCHDOG_SEC}"
    echo "profile_sweep_timeout_sec=${PROFILE_SWEEP_TIMEOUT_SEC:-auto}"
    echo "profile_concurrencies=${PROFILE_CONCURRENCIES}"
    echo "profile_nocg_concurrencies=${PROFILE_NOCG_CONCURRENCIES}"
    echo "skip_nocg_profile=${SKIP_NOCG_PROFILE}"
    echo "only_nomtp=${ONLY_NOMTP}"
    echo "only_mtp=${ONLY_MTP}"
    echo "profile_fails=${PROFILE_FAILS}"
    echo "analyze_fails=${ANALYZE_FAILS}"
    echo "paths:"
    echo "  acc=${SUITE_ROOT}/acc/"
    echo "  perf=${SUITE_ROOT}/perf/"
    echo "  profiles=${SUITE_ROOT}/profiles/"
    echo "  profile_logs=${SUITE_ROOT}/profile_logs/"
    echo "  server_logs=${SUITE_ROOT}/server_logs/"
    echo "  analyze=${SUITE_ROOT}/analyze/"
    echo "  suite_log=${SUITE_LOG}"
  } >"$mf"
  log "Wrote manifest: ${mf}"
}

cleanup() {
  local ec=$?
  stop_sweep_pid "${SWEEP_PID:-}" || true
  if [[ "$DRY_RUN" == true ]]; then
    exit "$ec"
  fi
  if [[ "$KEEP_SERVER" == true ]]; then
    log "Keeping server running (PID=${SERVER_PID:-unknown})"
    exit "$ec"
  fi
  kill_existing_server || true
  exit "$ec"
}

main() {
  if [[ -n "$SUITE_DIR_OVERRIDE" ]]; then
    mkdir -p "$SUITE_DIR_OVERRIDE"
    SUITE_ROOT="$(cd "$SUITE_DIR_OVERRIDE" && pwd)"
    mkdir -p "$SUITE_ROOT" "${SUITE_ROOT}/server_logs"
    SUITE_LOG="${SUITE_ROOT}/suite.log"
    touch "$SUITE_LOG"
  elif [[ "$DRY_RUN" == true ]]; then
    # Do not create a timestamped suite dir for dry-run (avoids empty leftovers).
    SUITE_ROOT="/tmp/glm_suite_dryrun"
    SUITE_LOG=""
  else
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    SUITE_ROOT="${SCRIPT_DIR}/suite_glm_env_${ts}"
    mkdir -p "$SUITE_ROOT" "${SUITE_ROOT}/server_logs"
    SUITE_LOG="${SUITE_ROOT}/suite.log"
    touch "$SUITE_LOG"
  fi

  setup_env
  trap cleanup EXIT INT TERM

  log "=== GLM env suite start ==="
  log "Suite dir: ${SUITE_ROOT}"
  log "phases=${PHASES}  dry-run=${DRY_RUN}"
  log "Model=${MODEL}  GPU_VENDOR=${GPU_VENDOR}  GPUS=${GPUS}  TP=${TP}  HOST=${HOST}:${PORT}"
  log "HF_HOME=${HF_HOME}  SGLANG_ROOT=${SGLANG_ROOT}"
  if [[ "$WANT_ACC" == true || "$WANT_PERF" == true ]]; then
    log "Short IO=${SHORT_IO}  Long IO=${LONG_IO} (max-running-requests=${LONG_MAX_RUNNING})"
  fi
  if [[ "$WANT_PROFILE" == true ]]; then
    log "Trace IO=${TRACE_IO}  profile_conc=${PROFILE_CONCURRENCIES}  nocg_conc=${PROFILE_NOCG_CONCURRENCIES} skip_nocg=${SKIP_NOCG_PROFILE}"
  fi
  log "Order: selected phases only (acc → perf → profile → analyze)"
  log "Profile: want=${WANT_PROFILE} steps=${PROFILE_NUM_STEPS} min_steps=${PROFILE_MIN_STEPS} max_wall_ms=${PROFILE_MAX_WALL_MS} conc=${PROFILE_CONCURRENCIES} nocg_conc=${PROFILE_NOCG_CONCURRENCIES} skip_nocg=${SKIP_NOCG_PROFILE} retries=${PROFILE_RETRIES} watchdog=${PROFILE_WATCHDOG_SEC}s sweep_timeout=${PROFILE_SWEEP_TIMEOUT_SEC:-auto} (nocg auto 2x)"
  log "Analyze: want=${WANT_ANALYZE} analyzer=${PROFILE_ANALYZER_ROOT}"

  # ----- ACC + PERF (shared servers) -----
  if [[ "$WANT_ACC" == true || "$WANT_PERF" == true ]]; then
    if [[ "$ONLY_MTP" != true ]]; then
      CURRENT_MODE="nomtp"
      if baseline_server_needed; then
        log "===== PHASE nomtp / baseline ====="
        start_server "nomtp_baseline"
        run_acc_if_needed
        run_short_perf_if_needed
      else
        log "Skip nomtp / baseline server (no acc, short-ctx skipped)"
      fi

      if [[ "$WANT_PERF" != true || "$SKIP_LONG" == true ]]; then
        log "Skip nomtp / long-ctx phase"
      else
        log "===== PHASE nomtp / long-ctx (max-running-requests=${LONG_MAX_RUNNING}) ====="
        start_server "nomtp_longctx" --max-running-requests "$LONG_MAX_RUNNING"
        run_long_perf_if_needed
      fi
    fi

    if [[ "$ONLY_NOMTP" != true ]]; then
      CURRENT_MODE="mtp"
      local -a eagle_args=()
      local line
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        # shellcheck disable=SC2206
        eagle_args+=($line)
      done < <(eagle_server_args)

      if baseline_server_needed; then
        log "===== PHASE mtp / EAGLE baseline ====="
        start_server "mtp_baseline" "${eagle_args[@]}"
        run_acc_if_needed
        run_short_perf_if_needed
      else
        log "Skip mtp / EAGLE baseline server (no acc, short-ctx skipped)"
      fi

      if [[ "$WANT_PERF" != true || "$SKIP_LONG" == true ]]; then
        log "Skip mtp / long-ctx phase"
      else
        log "===== PHASE mtp / EAGLE long-ctx (max-running-requests=${LONG_MAX_RUNNING}) ====="
        start_server "mtp_longctx" "${eagle_args[@]}" --max-running-requests "$LONG_MAX_RUNNING"
        run_long_perf_if_needed
      fi
    fi
  else
    log "Skip acc/perf stages (not in --phases)"
  fi

  # ----- PROFILE -----
  run_all_profiles_after_perf

  # ----- ANALYZE -----
  analyze_all_profiles

  if [[ "$DRY_RUN" != true ]]; then
    write_manifest
  else
    log "Skip manifest (dry-run)"
  fi

  if (( PROFILE_FAILS > 0 || ANALYZE_FAILS > 0 )); then
    log "=== GLM env suite finished WITH FAILURES ==="
    log "profile_fails=${PROFILE_FAILS} analyze_fails=${ANALYZE_FAILS}"
    if [[ "$DRY_RUN" != true ]]; then
      log "Logs under: ${SUITE_ROOT}"
    fi
    exit 1
  fi

  log "=== GLM env suite finished OK ==="
  if [[ "$DRY_RUN" == true ]]; then
    log "Dry-run only; no suite dirs or outputs written"
    log "Would write: server_logs/ profile_logs/ profiles/ analyze/ under ${SUITE_ROOT}"
    return 0
  fi
  log "Logs under: ${SUITE_ROOT}"
  if [[ "$WANT_PERF" == true ]]; then
    log "Perf:     ${SUITE_ROOT}/perf"
  fi
  if [[ "$WANT_ACC" == true ]]; then
    log "Acc:      ${SUITE_ROOT}/acc"
  fi
  if [[ "$WANT_PROFILE" == true ]]; then
    log "Profiles: ${SUITE_ROOT}/profiles"
    log "Profile stubs: ${SUITE_ROOT}/profile_logs"
    log "Server logs: ${SUITE_ROOT}/server_logs"
  fi
  if [[ "$WANT_ANALYZE" == true ]]; then
    log "Analyze:  ${SUITE_ROOT}/analyze"
  fi
}

main "$@"
