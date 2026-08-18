#!/usr/bin/env bash
#
# GLM env suite (sglang/suites/glm):
#   accuracy verify + perf sweep + 8k-only trace collect + trace analysis
#   for both non-MTP and MTP (EAGLE).
#
# Workload matrix (per mode nomtp|mtp):
#   1024:1024           perf only  (baseline server)
#   8192:1024           perf + DECODE trace + analyze  (baseline)
#   70000:300           perf only  (server --max-running-requests)
#
# Order: all perf (nomtp then mtp) -> profile 8k (cg-on, then c4 nocg) -> analyze.
#
# Usage:
#   bash run_env_suite.sh
#   bash run_env_suite.sh --skip-acc --only-nomtp
#   bash run_env_suite.sh --dry-run
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

# shellcheck source=../../recipes/glm.sh
source "$RECIPE_GLM"

usage() {
  cat <<'EOF'
Usage: run_env_suite.sh [options]

GLM suite stages:
  [acc]     GSM8K accuracy (per mode)
  [perf]    nomtp then mtp: 1024:1024 + 8192:1024, then 70000:300
  [profile] after all perf: baseline DECODE for 8192:1024, then
            conc=4 --disable-cuda-graph (op compare) for nomtp+mtp
  [analyze] decode_profile.single hierarchical Excel (8k traces)

IO matrix:
  1024:1024     perf only
  8192:1024     perf + trace + analyze
  70000:300     perf only (--max-running-requests)

Options:
  --model PATH                 model path  (default: amd/GLM-5.2-MXFP4)
  --model-key KEY              passed to sweep_bench  (default: glm)
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
  --skip-acc                   skip GSM8K accuracy phases
  --skip-perf                  skip all perf sweeps
  --skip-short-ctx             skip short-ctx perf (1024/8192)
  --skip-long-ctx              skip 70000:300 perf
  --skip-profile               skip torch profiling (after all perf)
  --skip-analyze               skip analysis after profiles
  --profile-num-steps N        profile decode steps  (default: 20)
  --profile-concurrencies N    conc list for profile only  (default: 8)
  --profile-nocg-concurrencies N
                               conc for disable-cuda-graph profile
                               (default: 4)
  --skip-nocg-profile          skip conc4 --disable-cuda-graph profiles
  --profile-retries N          retries after fail / bad trace  (default: 1)
  --profile-min-steps N        min acceptable DECODE steps
                               (default: max(5, 75% of num-steps))
  --profile-max-wall-ms MS     max decode/phase wall ms/step for 8k
                               (default: 300; scaled by input length)
  --profile-analyzer-root PATH (default: <repo>/analysis)
  --suite-dir PATH             reuse an existing suite dir (no new timestamp)
  --from-phase PHASE           perf|profile|analyze  (default: perf)
  --only-nomtp                 only nomtp (no EAGLE)
  --only-mtp                   only mtp / EAGLE
  --keep-server                do not stop server at the very end
  --dry-run                    print commands only
  --extra-server-args "..."    extra args appended to every launch_server
  --extra-sweep-args "..."     extra args appended to every sweep_bench.sh
  -h, --help                   show this help

Environment (set before launch if unset):
  HF_HOME, SGLANG_ROCM_FUSED_DECODE_MLA=0, ROCM_QUICK_REDUCE_QUANTIZATION=INT4,
  SGLANG_OPT_USE_TOPK_V2=0, PYTHONPATH=<sglang>/python
EOF
}

MODEL="amd/GLM-5.2-MXFP4"
MODEL_KEY="glm"
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
SKIP_ACC=false
SKIP_PERF=false
SKIP_SHORT=false
SKIP_LONG=false
SKIP_PROFILE=false
SKIP_ANALYZE=false
SKIP_NOCG_PROFILE=false
PROFILE_NUM_STEPS="20"
PROFILE_CONCURRENCIES="8"
PROFILE_NOCG_CONCURRENCIES="4"
PROFILE_RETRIES="1"
PROFILE_MIN_STEPS=""          # empty -> auto from PROFILE_NUM_STEPS
PROFILE_MAX_WALL_MS="300"     # short-ctx default; long IO scaled in helper
PROFILE_ANALYZER_ROOT="$PROFILE_ANALYZER_ROOT_DEFAULT"
SUITE_DIR_OVERRIDE=""
FROM_PHASE="perf"   # perf | profile | analyze
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
declare -a CURRENT_SERVER_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)              MODEL="$2"; shift 2 ;;
    --model-key)          MODEL_KEY="$2"; shift 2 ;;
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
    --skip-acc)           SKIP_ACC=true; shift ;;
    --skip-perf)          SKIP_PERF=true; shift ;;
    --skip-short-ctx)     SKIP_SHORT=true; shift ;;
    --skip-long-ctx)      SKIP_LONG=true; shift ;;
    --skip-profile)       SKIP_PROFILE=true; shift ;;
    --skip-analyze)       SKIP_ANALYZE=true; shift ;;
    --skip-nocg-profile)  SKIP_NOCG_PROFILE=true; shift ;;
    --profile-num-steps)  PROFILE_NUM_STEPS="$2"; shift 2 ;;
    --profile-concurrencies) PROFILE_CONCURRENCIES="$2"; shift 2 ;;
    --profile-nocg-concurrencies) PROFILE_NOCG_CONCURRENCIES="$2"; shift 2 ;;
    --profile-retries)    PROFILE_RETRIES="$2"; shift 2 ;;
    --profile-min-steps)  PROFILE_MIN_STEPS="$2"; shift 2 ;;
    --profile-max-wall-ms) PROFILE_MAX_WALL_MS="$2"; shift 2 ;;
    --profile-analyzer-root) PROFILE_ANALYZER_ROOT="$2"; shift 2 ;;
    --suite-dir)          SUITE_DIR_OVERRIDE="$2"; shift 2 ;;
    --from-phase)         FROM_PHASE="$2"; shift 2 ;;
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

if [[ "$ONLY_NOMTP" == true && "$ONLY_MTP" == true ]]; then
  echo "ERROR: --only-nomtp and --only-mtp are mutually exclusive" >&2
  exit 1
fi
case "$FROM_PHASE" in
  perf|profile|analyze) ;;
  *)
    echo "ERROR: --from-phase must be perf|profile|analyze (got: ${FROM_PHASE})" >&2
    exit 1
    ;;
esac
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
  export SGLANG_ROCM_FUSED_DECODE_MLA="${SGLANG_ROCM_FUSED_DECODE_MLA:-0}"
  export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
  export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
  export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
  export CUDA_VISIBLE_DEVICES="$GPUS"
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

wait_server_ready() {
  local url="http://${HOST}:${PORT}/v1/models"
  local start now elapsed
  start="$(date +%s)"
  log "Waiting for server ready: ${url} (timeout ${READY_TIMEOUT}s)"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  while true; do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
      log "Server is ready"
      return 0
    fi
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      die "server process exited early; see ${SERVER_LOG}"
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed >= READY_TIMEOUT )); then
      die "server not ready after ${READY_TIMEOUT}s; see ${SERVER_LOG}"
    fi
    sleep 5
  done
}

start_server() {
  local phase_name="$1"
  shift
  local -a extra=("$@")
  local -a args=()
  local line

  CURRENT_PHASE="$phase_name"
  CURRENT_SERVER_EXTRA=("${extra[@]}")

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # shellcheck disable=SC2206
    args+=($line)
  done < <(base_server_args)

  if ((${#extra[@]})); then
    args+=("${extra[@]}")
  fi
  if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
    # shellcheck disable=SC2206
    args+=($EXTRA_SERVER_ARGS)
  fi

  SERVER_LOG="${SUITE_ROOT}/${phase_name}.server.log"
  log "Launching server [${phase_name}] -> ${SERVER_LOG}"
  log "Args: python3 -m sglang.launch_server ${args[*]}"

  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  kill_existing_server
  # shellcheck disable=SC2086
  python3 -m sglang.launch_server "${args[@]}" >"$SERVER_LOG" 2>&1 &
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

# True if /v1/models works and scheduler workers are alive (not zombie-only).
server_is_healthy() {
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  if ! curl -fsS -m 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    return 1
  fi
  if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    return 1
  fi
  # launch_server parent alive but schedulers crashed -> zombie HTTP
  if ! pgrep -f 'sglang::scheduler' >/dev/null 2>&1; then
    return 1
  fi
  if pgrep -af 'sglang::schedul' 2>/dev/null | grep -q '<defunct>'; then
    return 1
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

count_decode_traces() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -type f -name '*DECODE.trace.json.gz' 2>/dev/null | wc -l
}

clear_decode_traces() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  find "$dir" -type f -name '*DECODE.trace.json.gz' -delete 2>/dev/null || true
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
validate_profile_dir() {
  local pdir="$1"
  local io="$2"
  local mode="$3"
  local max_wall
  max_wall="$(profile_max_wall_for_io "$io")"
  local analyzer_root="$PROFILE_ANALYZER_ROOT"
  local rules="${analyzer_root}/rules/glm52.csv"
  local py_path="${analyzer_root}${PYTHONPATH:+:${PYTHONPATH}}"
  local vrc

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN validate: dir=${pdir} mode=${mode} min_steps=${PROFILE_MIN_STEPS} max_wall_ms=${max_wall}"
    return 0
  fi
  if [[ ! -d "$analyzer_root/decode_profile" ]]; then
    log "WARN: analysis tree missing; skip trace validation"
    return 0
  fi

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
      --mode "$mode"
  )
  vrc=$?
  set -e
  return "$vrc"
}

run_sweep() {
  local mode="$1"
  shift
  local -a sweep_args=(
    --model-key "$MODEL_KEY"
    --mode "$mode"
    --model-override "$MODEL"
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

run_acc_if_needed() {
  if [[ "$SKIP_ACC" == true ]]; then
    log "Skip acc"
    return 0
  fi
  mkdir -p "${SUITE_ROOT}/acc/${CURRENT_MODE}"
  run_sweep acc --acc-result-dir "${SUITE_ROOT}/acc/${CURRENT_MODE}"
}

run_short_perf_if_needed() {
  if [[ "$SKIP_PERF" == true || "$SKIP_SHORT" == true ]]; then
    log "Skip short-ctx perf (${SHORT_IO})"
    return 0
  fi
  mkdir -p "${SUITE_ROOT}/perf/${CURRENT_MODE}/short"
  run_sweep perf \
    --perf-io-pairs "$SHORT_IO" \
    --perf-result-dir "${SUITE_ROOT}/perf/${CURRENT_MODE}/short"
}

run_long_perf_if_needed() {
  if [[ "$SKIP_PERF" == true || "$SKIP_LONG" == true ]]; then
    log "Skip long-ctx perf (${LONG_IO})"
    return 0
  fi
  mkdir -p "${SUITE_ROOT}/perf/${CURRENT_MODE}/long"
  run_sweep perf \
    --perf-io-pairs "$LONG_IO" \
    --perf-result-dir "${SUITE_ROOT}/perf/${CURRENT_MODE}/long"
}

# One IO pair profile via sweep_bench --profile; restart server on failure
# or when DECODE trace step/wall sanity checks fail.
# Optional: conc override and tag_suffix (e.g. _c4_nocg for op-compare traces).
run_one_profile() {
  local io="$1"
  local conc="${2:-$PROFILE_CONCURRENCIES}"
  local tag_suffix="${3:-}"
  local tag
  tag="$(io_tag_from_pair "$io")${tag_suffix}"
  local pdir="${SUITE_ROOT}/profiles/${CURRENT_MODE}/${tag}"
  local stub="${SUITE_ROOT}/profile_sweep_logs/${CURRENT_MODE}/${tag}"
  local attempts=$((PROFILE_RETRIES + 1))
  local attempt rc ntraces max_wall

  max_wall="$(profile_max_wall_for_io "$io")"
  mkdir -p "$pdir" "$stub"
  log "===== PROFILE mode=${CURRENT_MODE} io=${io} conc=${conc} dir=${pdir} ====="
  log "Trace checks: expected_steps=${PROFILE_NUM_STEPS} min_steps=${PROFILE_MIN_STEPS} max_wall_ms=${max_wall}"

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN profile: --perf-io-pairs ${io} --perf-concurrencies ${conc} --profile --profile-output-dir ${pdir}"
    return 0
  fi

  for attempt in $(seq 1 "$attempts"); do
    if ! server_is_healthy; then
      log "WARN: server unhealthy before profile attempt ${attempt}/${attempts}"
      restart_current_server
    fi

    # Drop prior bad/partial traces so retries do not pick them up.
    clear_decode_traces "$pdir"

    set +e
    run_sweep perf \
      --perf-io-pairs "$io" \
      --perf-concurrencies "$conc" \
      --profile \
      --profile-output-dir "$pdir" \
      --profile-num-steps "$PROFILE_NUM_STEPS" \
      --perf-result-dir "$stub"
    rc=$?
    set -e

    ntraces="$(count_decode_traces "$pdir")"
    log "Profile attempt ${attempt}/${attempts}: sweep_rc=${rc} decode_traces=${ntraces}"

    if (( ntraces <= 0 )); then
      log "WARN: no DECODE traces for ${CURRENT_MODE}/${tag} (attempt ${attempt}/${attempts})"
      if (( attempt < attempts )); then
        restart_current_server
      fi
      continue
    fi

    if ! validate_profile_dir "$pdir" "$io" "$CURRENT_MODE"; then
      log "WARN: trace sanity check failed for ${CURRENT_MODE}/${tag} (attempt ${attempt}/${attempts})"
      # Keep the last rejected traces for post-mortem; only clear before a retry.
      if (( attempt < attempts )); then
        clear_decode_traces "$pdir"
        restart_current_server
      fi
      continue
    fi

    if ! server_smoke_ok; then
      log "WARN: server unhealthy after profile (traces kept=${ntraces}); will restart"
      restart_current_server
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
  if [[ "$SKIP_PROFILE" == true ]]; then
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

  local -a pairs=()
  local io prc
  IFS=',' read -ra pairs <<< "$ios"
  for io in "${pairs[@]}"; do
    io="${io// /}"
    [[ -z "$io" ]] && continue
    # Do not abort the whole suite if one shape's profile fails.
    set +e
    run_one_profile "$io" "$conc" "$tag_suffix"
    prc=$?
    set -e
    if (( prc != 0 )); then
      PROFILE_FAILS=$((PROFILE_FAILS + 1))
    fi
  done
}

# After all perf: baseline (cuda-graph ON) then conc4 --disable-cuda-graph
# for op comparison. DECODE for TRACE_IO (default 8k) only.
run_all_profiles_after_perf() {
  if [[ "$SKIP_PROFILE" == true ]]; then
    log "Skip all profiles"
    return 0
  fi

  log "===== PROFILE PHASE (after all perf; TRACE_IO=${TRACE_IO}) ====="

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
    start_server "nomtp_baseline"
    run_profile_ios_if_needed "$TRACE_IO" trace

    if [[ "$SKIP_NOCG_PROFILE" == true ]]; then
      log "Skip nomtp nocg profile (--skip-nocg-profile)"
    else
      log "===== PROFILE nomtp / TRACE_IO=${TRACE_IO} (disable-cuda-graph, conc=${PROFILE_NOCG_CONCURRENCIES}) ====="
      start_server "nomtp_profile_nocg" --disable-cuda-graph
      run_profile_ios_if_needed "$TRACE_IO" nocg \
        "$PROFILE_NOCG_CONCURRENCIES" "_c${PROFILE_NOCG_CONCURRENCIES}_nocg"
    fi
  fi

  if [[ "$ONLY_NOMTP" != true ]]; then
    CURRENT_MODE="mtp"
    log "===== PROFILE mtp / TRACE_IO=${TRACE_IO} (cuda-graph ON, conc=${PROFILE_CONCURRENCIES}) ====="
    start_server "mtp_baseline" "${eagle_args[@]}"
    run_profile_ios_if_needed "$TRACE_IO" trace

    if [[ "$SKIP_NOCG_PROFILE" == true ]]; then
      log "Skip mtp nocg profile (--skip-nocg-profile)"
    else
      log "===== PROFILE mtp / TRACE_IO=${TRACE_IO} (disable-cuda-graph, conc=${PROFILE_NOCG_CONCURRENCIES}) ====="
      start_server "mtp_profile_nocg" "${eagle_args[@]}" --disable-cuda-graph
      run_profile_ios_if_needed "$TRACE_IO" nocg \
        "$PROFILE_NOCG_CONCURRENCIES" "_c${PROFILE_NOCG_CONCURRENCIES}_nocg"
    fi
  fi
}

analyze_all_profiles() {
  if [[ "$SKIP_ANALYZE" == true || "$SKIP_PROFILE" == true ]]; then
    log "Skip analysis"
    return 0
  fi

  local analyzer_root="$PROFILE_ANALYZER_ROOT"
  local out_dir="${SUITE_ROOT}/analyze"
  local rules="${analyzer_root}/rules/glm52.csv"
  mkdir -p "$out_dir"

  if [[ "$DRY_RUN" == true ]]; then
    log "DRY-RUN: would single-side analyze each profiles/{nomtp,mtp}/<io> -> ${out_dir}"
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
  local mode_dir mode tag_dir tag ntraces out_xlsx

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
      log "Analyze ${mode}/${tag} (decode_traces=${ntraces}) -> ${out_xlsx}"
      set +e
      (
        cd "$analyzer_root"
        PYTHONPATH="$py_path" python3 -m decode_profile.single \
          --dir "$tag_dir" \
          --label "$mode" \
          --rules "$rules" \
          -o "$out_xlsx"
      )
      local arc=$?
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
    echo "from_phase=${FROM_PHASE}"
    echo "model=${MODEL}"
    echo "model_key=${MODEL_KEY}"
    echo "gpus=${GPUS}"
    echo "tp=${TP}"
    echo "host=${HOST}:${PORT}"
    echo "short_io=${SHORT_IO}"
    echo "long_io=${LONG_IO}"
    echo "profile_num_steps=${PROFILE_NUM_STEPS}"
    echo "profile_min_steps=${PROFILE_MIN_STEPS}"
    echo "profile_max_wall_ms=${PROFILE_MAX_WALL_MS}"
    echo "profile_concurrencies=${PROFILE_CONCURRENCIES}"
    echo "profile_nocg_concurrencies=${PROFILE_NOCG_CONCURRENCIES}"
    echo "skip_nocg_profile=${SKIP_NOCG_PROFILE}"
    echo "only_nomtp=${ONLY_NOMTP}"
    echo "only_mtp=${ONLY_MTP}"
    echo "skip_acc=${SKIP_ACC} skip_perf=${SKIP_PERF} skip_profile=${SKIP_PROFILE} skip_analyze=${SKIP_ANALYZE}"
    echo "profile_fails=${PROFILE_FAILS}"
    echo "analyze_fails=${ANALYZE_FAILS}"
    echo "paths:"
    echo "  acc=${SUITE_ROOT}/acc/"
    echo "  perf=${SUITE_ROOT}/perf/"
    echo "  profiles=${SUITE_ROOT}/profiles/"
    echo "  profile_sweep_logs=${SUITE_ROOT}/profile_sweep_logs/"
    echo "  analyze=${SUITE_ROOT}/analyze/"
    echo "  suite_log=${SUITE_LOG}"
  } >"$mf"
  log "Wrote manifest: ${mf}"
}

cleanup() {
  local ec=$?
  if [[ "$KEEP_SERVER" == true ]]; then
    log "Keeping server running (PID=${SERVER_PID:-unknown})"
    exit "$ec"
  fi
  kill_existing_server || true
  exit "$ec"
}

main() {
  if [[ -n "$SUITE_DIR_OVERRIDE" ]]; then
    SUITE_ROOT="$(cd "$SUITE_DIR_OVERRIDE" && pwd)"
  else
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    SUITE_ROOT="${SCRIPT_DIR}/suite_glm_env_${ts}"
    mkdir -p "$SUITE_ROOT"
  fi
  mkdir -p "$SUITE_ROOT"
  SUITE_LOG="${SUITE_ROOT}/suite.log"
  touch "$SUITE_LOG"

  setup_env
  trap cleanup EXIT INT TERM

  log "=== GLM env suite start ==="
  log "Suite dir: ${SUITE_ROOT}"
  log "from-phase=${FROM_PHASE}"
  log "Model=${MODEL}  GPUS=${GPUS}  TP=${TP}  HOST=${HOST}:${PORT}"
  log "HF_HOME=${HF_HOME}  SGLANG_ROOT=${SGLANG_ROOT}"
  log "Short IO=${SHORT_IO}  Long IO=${LONG_IO}  Trace IO=${TRACE_IO} (max-running-requests=${LONG_MAX_RUNNING})"
  log "Order: all perf first, then profile TRACE_IO only, then analyze"
  log "Profile: skip=${SKIP_PROFILE} steps=${PROFILE_NUM_STEPS} min_steps=${PROFILE_MIN_STEPS} max_wall_ms=${PROFILE_MAX_WALL_MS} conc=${PROFILE_CONCURRENCIES} nocg_conc=${PROFILE_NOCG_CONCURRENCIES} skip_nocg=${SKIP_NOCG_PROFILE} retries=${PROFILE_RETRIES}"
  log "Analyze: skip=${SKIP_ANALYZE} analyzer=${PROFILE_ANALYZER_ROOT}"

  # ----- PERF: non-MTP -----
  if [[ "$FROM_PHASE" == "perf" ]]; then
    if [[ "$ONLY_MTP" != true ]]; then
      CURRENT_MODE="nomtp"
      log "===== PHASE nomtp / baseline (perf only) ====="
      start_server "nomtp_baseline"
      run_acc_if_needed
      run_short_perf_if_needed

      if [[ "$SKIP_LONG" == true ]]; then
        log "Skip nomtp / long-ctx phase (--skip-long-ctx)"
      else
        log "===== PHASE nomtp / long-ctx (perf only, max-running-requests=${LONG_MAX_RUNNING}) ====="
        start_server "nomtp_longctx" --max-running-requests "$LONG_MAX_RUNNING"
        run_long_perf_if_needed
      fi
    fi

    # ----- PERF: MTP / EAGLE -----
    if [[ "$ONLY_NOMTP" != true ]]; then
      CURRENT_MODE="mtp"
      local -a eagle_args=()
      local line
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        # shellcheck disable=SC2206
        eagle_args+=($line)
      done < <(eagle_server_args)

      log "===== PHASE mtp / EAGLE baseline (perf only) ====="
      start_server "mtp_baseline" "${eagle_args[@]}"
      run_acc_if_needed
      run_short_perf_if_needed

      if [[ "$SKIP_LONG" == true ]]; then
        log "Skip mtp / long-ctx phase (--skip-long-ctx)"
      else
        log "===== PHASE mtp / EAGLE long-ctx (perf only, max-running-requests=${LONG_MAX_RUNNING}) ====="
        start_server "mtp_longctx" "${eagle_args[@]}" --max-running-requests "$LONG_MAX_RUNNING"
        run_long_perf_if_needed
      fi
    fi
  else
    log "Skip perf stages (--from-phase=${FROM_PHASE})"
  fi

  # ----- PROFILE (after all perf) -----
  if [[ "$FROM_PHASE" == "perf" || "$FROM_PHASE" == "profile" ]]; then
    run_all_profiles_after_perf
  else
    log "Skip profile stage (--from-phase=${FROM_PHASE})"
  fi

  analyze_all_profiles

  write_manifest

  if (( PROFILE_FAILS > 0 || ANALYZE_FAILS > 0 )); then
    log "=== GLM env suite finished WITH FAILURES ==="
    log "profile_fails=${PROFILE_FAILS} analyze_fails=${ANALYZE_FAILS}"
    log "Logs under: ${SUITE_ROOT}"
    exit 1
  fi

  log "=== GLM env suite finished OK ==="
  log "Logs under: ${SUITE_ROOT}"
  log "Perf:     ${SUITE_ROOT}/perf"
  log "Acc:      ${SUITE_ROOT}/acc"
  log "Profiles: ${SUITE_ROOT}/profiles"
  log "Analyze:  ${SUITE_ROOT}/analyze"
}

main "$@"
