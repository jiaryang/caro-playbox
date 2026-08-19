#!/usr/bin/env bash
# Collect machine fingerprints + run RCCL all_reduce_perf for cross-node compare.
#
#   bash experiments/rccl/run_allreduce_bench.sh
#   GPUS=4,5,6,7 OUT_DIR=/tmp/rccl_m11 bash experiments/rccl/run_allreduce_bench.sh
#   bash experiments/rccl/run_allreduce_bench.sh --info-only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS="${GPUS:-4,5,6,7}"
INFO_ONLY=0
SKIP_BUILD="${SKIP_BUILD:-0}"
FIXED_SIZES="${FIXED_SIZES:-1M 4M 16M 64M}"
SWEEP_ARGS="${SWEEP_ARGS:--b 8 -e 128M -f 2 -n 50 -w 20}"
RCCL_TESTS_GIT="${RCCL_TESTS_GIT:-https://github.com/ROCm/rccl-tests.git}"

usage() {
  sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "Env: GPUS OUT_DIR RCCL_TESTS_DIR RCCL_TESTS_SRC SKIP_BUILD FIXED_SIZES SWEEP_ARGS"
  echo "Flags: --info-only   collect fingerprints only"
  echo "       -h|--help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --info-only) INFO_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Visible devices for both HIP and CUDA-style bindings.
export HIP_VISIBLE_DEVICES="$GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS"
# Count GPUs (commas + 1)
NGPU=$(( $(grep -o ',' <<<"$GPUS" | wc -l) + 1 ))

HOST="$(hostname -s 2>/dev/null || hostname)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/rccl/results/rccl_bench_${HOST}_${TS}}"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

INFO_DIR="${OUT_DIR}/info"
BENCH_DIR="${OUT_DIR}/bench"
mkdir -p "$INFO_DIR" "$BENCH_DIR"

echo "=== RCCL allreduce bench ==="
echo "OUT_DIR=${OUT_DIR}"
echo "GPUS=${GPUS} (ngpu=${NGPU})"
echo "REPO_ROOT=${REPO_ROOT}"

# --- fingerprints ---
bash "${SCRIPT_DIR}/collect_info.sh" "$INFO_DIR"

{
  echo "suite=rccl_allreduce_bench"
  echo "started=$(date -Is)"
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "gpus=${GPUS}"
  echo "ngpu=${NGPU}"
  echo "info_only=${INFO_ONLY}"
  echo "repo_root=${REPO_ROOT}"
  echo "hip_visible=${HIP_VISIBLE_DEVICES}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES}"
  echo "note=Plain all_reduce_perf is a fabric proxy; SGLang GLM EXTEND may use quickreduce."
} >"${OUT_DIR}/manifest.txt"

if [[ "$INFO_ONLY" -eq 1 ]]; then
  echo "INFO_ONLY=1 — skip build/bench."
  echo "Done. See ${OUT_DIR}/info/"
  exit 0
fi

# --- locate / build all_reduce_perf ---
find_binary() {
  if [[ -n "${RCCL_TESTS_DIR:-}" && -x "${RCCL_TESTS_DIR}/all_reduce_perf" ]]; then
    echo "${RCCL_TESTS_DIR}/all_reduce_perf"
    return 0
  fi
  if command -v all_reduce_perf >/dev/null 2>&1; then
    command -v all_reduce_perf
    return 0
  fi
  local cand
  for cand in \
      "${OUT_DIR}/rccl-tests/build/all_reduce_perf" \
      "${OUT_DIR}/src/rccl-tests/build/all_reduce_perf" \
      "${REPO_ROOT}/experiments/rccl/.cache/rccl-tests/build/all_reduce_perf"; do
    if [[ -x "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

BIN=""
if BIN="$(find_binary)"; then
  echo "Using existing binary: ${BIN}"
elif [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "ERROR: SKIP_BUILD=1 but all_reduce_perf not found. Set RCCL_TESTS_DIR." >&2
  exit 1
else
  SRC="${RCCL_TESTS_SRC:-${OUT_DIR}/src/rccl-tests}"
  if [[ ! -d "$SRC/.git" ]]; then
    echo "Cloning rccl-tests -> ${SRC}"
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 "$RCCL_TESTS_GIT" "$SRC"
  else
    echo "Reusing source: ${SRC}"
  fi
  echo "Building rccl-tests (MPI=0) ..."
  (
    cd "$SRC"
    # MPI=0: single-node multi-GPU only (matches SGLang TP on one node)
    make MPI=0 -j"$(nproc)" 2>&1 | tee "${BENCH_DIR}/build.log"
  )
  if [[ -x "${SRC}/build/all_reduce_perf" ]]; then
    BIN="${SRC}/build/all_reduce_perf"
  elif [[ -x "${SRC}/all_reduce_perf" ]]; then
    BIN="${SRC}/all_reduce_perf"
  else
    # some Makefiles drop binary in cwd
    BIN="$(find "$SRC" -type f -name all_reduce_perf -perm -111 | head -n 1 || true)"
  fi
  if [[ -z "$BIN" || ! -x "$BIN" ]]; then
    echo "ERROR: build finished but all_reduce_perf not found. See ${BENCH_DIR}/build.log" >&2
    exit 1
  fi
  echo "Built: ${BIN}"
fi

echo "binary=${BIN}" >>"${OUT_DIR}/manifest.txt"
cp -L "$BIN" "${BENCH_DIR}/all_reduce_perf" 2>/dev/null || true
"${BIN}" --help >"${BENCH_DIR}/all_reduce_perf_help.txt" 2>&1 || true

run_one() {
  local tag="$1"
  shift
  local log="${BENCH_DIR}/${tag}.log"
  echo
  echo "=== RUN ${tag}: ${BIN} $* -g ${NGPU} ==="
  {
    echo "### CMD: ${BIN} $* -g ${NGPU}"
    echo "### AT: $(date -Is)"
    echo "### HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    echo
    # shellcheck disable=SC2086
    "${BIN}" "$@" -g "${NGPU}"
  } 2>&1 | tee "$log"
}

# Sweep + fixed sizes in the ~ms AR band seen on GLM EXTEND profiles.
# shellcheck disable=SC2086
run_one sweep ${SWEEP_ARGS}

for sz in ${FIXED_SIZES}; do
  run_one "fixed_${sz}" -b "$sz" -e "$sz" -n 100 -w 20
done

# Post GPU snapshot
bash "${SCRIPT_DIR}/collect_info.sh" "${OUT_DIR}/info_after" 2>/dev/null || true
# keep only gpus snapshot after to avoid noise; merge note
if [[ -f "${OUT_DIR}/info_after/gpus.txt" ]]; then
  cp "${OUT_DIR}/info_after/gpus.txt" "${INFO_DIR}/gpus_after.txt"
  rm -rf "${OUT_DIR}/info_after"
fi

# Lightweight summary: grab busbw lines from sweep
{
  echo "RCCL allreduce bench summary"
  echo "host=$(hostname -f 2>/dev/null || hostname)"
  echo "gpus=${GPUS}"
  echo "binary=${BIN}"
  echo "finished=$(date -Is)"
  echo
  echo "=== sweep (busbw / time lines) ==="
  if [[ -f "${BENCH_DIR}/sweep.log" ]]; then
    grep -E '^[0-9]|busbw|out-of-place|in-place|size' "${BENCH_DIR}/sweep.log" | tail -n 80 || true
  fi
  echo
  echo "=== fixed-size tails ==="
  for sz in ${FIXED_SIZES}; do
    f="${BENCH_DIR}/fixed_${sz}.log"
    [[ -f "$f" ]] || continue
    echo "--- ${sz} ---"
    tail -n 15 "$f"
    echo
  done
  echo "Compare two result dirs with:"
  echo "  python ${SCRIPT_DIR}/compare_runs.py <dir_a> <dir_b>"
} | tee "${OUT_DIR}/summary.txt"

echo "finished=$(date -Is)" >>"${OUT_DIR}/manifest.txt"
echo
echo "Done."
echo "  info:    ${INFO_DIR}"
echo "  bench:   ${BENCH_DIR}"
echo "  summary: ${OUT_DIR}/summary.txt"
