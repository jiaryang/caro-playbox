#!/usr/bin/env bash
# Collect machine fingerprints + run RCCL all_reduce_perf for cross-node compare.
#
#   bash experiments/rccl/run_allreduce_bench.sh
#   GPUS=4,5,6,7 OUT_DIR=/tmp/rccl_m11 bash experiments/rccl/run_allreduce_bench.sh
#   bash experiments/rccl/run_allreduce_bench.sh --info-only
#
# Source/binary live under experiments/rccl/.cache/ (gitignored) and are reused
# across runs. Only rebuild when missing, or set FORCE_BUILD=1.
# Shareable compare bundle is written to OUT_DIR/compare/ (no src, no binary).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS="${GPUS:-4,5,6,7}"
INFO_ONLY=0
SKIP_BUILD="${SKIP_BUILD:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
COPY_BINARY="${COPY_BINARY:-0}"
FIXED_SIZES="${FIXED_SIZES:-1M 4M 16M 64M}"
SWEEP_ARGS="${SWEEP_ARGS:--b 8 -e 128M -f 2 -n 50 -w 20}"
RCCL_TESTS_GIT="${RCCL_TESTS_GIT:-https://github.com/ROCm/rccl-tests.git}"
# Shared clone+build tree (not under OUT_DIR). Override with RCCL_TESTS_SRC / RCCL_TESTS_DIR.
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/.cache/rccl-tests}"

usage() {
  sed -n '1,14p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "Env: GPUS OUT_DIR CACHE_DIR RCCL_TESTS_DIR RCCL_TESTS_SRC"
  echo "     SKIP_BUILD FORCE_BUILD COPY_BINARY FIXED_SIZES SWEEP_ARGS"
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
COMPARE_DIR="${OUT_DIR}/compare"
mkdir -p "$INFO_DIR" "$BENCH_DIR"

echo "=== RCCL allreduce bench ==="
echo "OUT_DIR=${OUT_DIR}"
echo "GPUS=${GPUS} (ngpu=${NGPU})"
echo "CACHE_DIR=${CACHE_DIR}"
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
  echo "cache_dir=${CACHE_DIR}"
  echo "hip_visible=${HIP_VISIBLE_DEVICES}"
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES}"
  echo "note=Plain all_reduce_perf is a fabric proxy; SGLang GLM EXTEND may use quickreduce."
} >"${OUT_DIR}/manifest.txt"

write_compare_bundle() {
  local dest="${1:-$COMPARE_DIR}"
  mkdir -p "${dest}/bench" "${dest}/info"
  cp -a "${OUT_DIR}/manifest.txt" "${dest}/"
  [[ -f "${OUT_DIR}/summary.txt" ]] && cp -a "${OUT_DIR}/summary.txt" "${dest}/"
  [[ -f "${BENCH_DIR}/sweep.log" ]] && cp -a "${BENCH_DIR}/sweep.log" "${dest}/bench/"
  local f
  for f in "${BENCH_DIR}"/fixed_*.log; do
    [[ -f "$f" ]] && cp -a "$f" "${dest}/bench/"
  done
  for f in rocm.txt topo.txt rccl_libs.txt host.txt gpus.txt gpus_after.txt \
           env.txt host_tuning.txt docker.txt INDEX.txt; do
    [[ -f "${INFO_DIR}/${f}" ]] && cp -a "${INFO_DIR}/${f}" "${dest}/info/"
  done
  cat >"${dest}/README.txt" <<EOF
RCCL compare bundle (perf + fingerprints only).
Share this directory (or OUT_DIR/compare) across machines.

  python ${SCRIPT_DIR}/compare_runs.py <other>/compare <this>/compare

Includes: manifest, summary, bench/sweep + fixed_*.log, info fingerprints.
Excluded: src/, all_reduce_perf binary, build.log.
EOF
  echo "Wrote compare bundle: ${dest}"
}

if [[ "$INFO_ONLY" -eq 1 ]]; then
  echo "INFO_ONLY=1 — skip build/bench."
  write_compare_bundle
  echo "Done. See ${OUT_DIR}/info/ and ${COMPARE_DIR}/"
  exit 0
fi

# --- locate / build all_reduce_perf (prefer shared cache; never clone into OUT_DIR) ---
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
      "${CACHE_DIR}/build/all_reduce_perf" \
      "${CACHE_DIR}/all_reduce_perf" \
      "${OUT_DIR}/src/rccl-tests/build/all_reduce_perf" \
      "${REPO_ROOT}/experiments/rccl/.cache/rccl-tests/build/all_reduce_perf"; do
    if [[ -x "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

# Progress/logs go to stderr so command-substitution only captures the path.
# Otherwise BIN becomes the whole build log and exec fails with "File name too long".
ensure_binary() {
  local bin=""
  if [[ "$FORCE_BUILD" != "1" ]] && bin="$(find_binary)"; then
    echo "Using existing binary: ${bin}" >&2
    printf '%s\n' "$bin"
    return 0
  fi
  if [[ "$FORCE_BUILD" == "1" ]]; then
    echo "FORCE_BUILD=1 — rebuilding even if a binary exists." >&2
  fi
  if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "ERROR: SKIP_BUILD=1 but all_reduce_perf not found. Set RCCL_TESTS_DIR or unset SKIP_BUILD." >&2
    return 1
  fi

  local src="${RCCL_TESTS_SRC:-$CACHE_DIR}"
  if [[ ! -d "${src}/.git" ]]; then
    echo "Cloning rccl-tests -> ${src} (one-time; reused on later runs)" >&2
    mkdir -p "$(dirname "$src")"
    git clone --depth 1 "$RCCL_TESTS_GIT" "$src" >&2
  else
    echo "Reusing source: ${src}" >&2
  fi
  echo "Building rccl-tests (MPI=0) ..." >&2
  mkdir -p "$BENCH_DIR"
  (
    cd "$src"
    # MPI=0: single-node multi-GPU only (matches SGLang TP on one node)
    # tee to log file; also mirror to stderr (not stdout — stdout is the path only).
    make MPI=0 -j"$(nproc)" 2>&1 | tee "${BENCH_DIR}/build.log" >&2
  )
  if [[ -x "${src}/build/all_reduce_perf" ]]; then
    bin="${src}/build/all_reduce_perf"
  elif [[ -x "${src}/all_reduce_perf" ]]; then
    bin="${src}/all_reduce_perf"
  else
    bin="$(find "$src" -type f -name all_reduce_perf -perm -111 | head -n 1 || true)"
  fi
  if [[ -z "$bin" || ! -x "$bin" ]]; then
    echo "ERROR: build finished but all_reduce_perf not found. See ${BENCH_DIR}/build.log" >&2
    return 1
  fi
  echo "Built: ${bin}" >&2
  printf '%s\n' "$bin"
}

BIN="$(ensure_binary)"
# Defensive: only keep the last non-empty line (the path).
BIN="$(printf '%s\n' "$BIN" | awk 'NF{p=$0} END{print p}')"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "ERROR: resolved binary is not executable: '${BIN}'" >&2
  exit 1
fi

echo "binary=${BIN}" >>"${OUT_DIR}/manifest.txt"
if [[ "$COPY_BINARY" == "1" ]]; then
  cp -L "$BIN" "${BENCH_DIR}/all_reduce_perf" 2>/dev/null || true
fi
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
    # Rows are space-padded; match leading whitespace + digit.
    grep -E '^[[:space:]]*[0-9]|busbw|out-of-place|in-place|size' "${BENCH_DIR}/sweep.log" | tail -n 80 || true
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
  echo "Compare two machines with the slim bundles:"
  echo "  python ${SCRIPT_DIR}/compare_runs.py <dir_a>/compare <dir_b>/compare"
} | tee "${OUT_DIR}/summary.txt"

write_compare_bundle

echo "finished=$(date -Is)" >>"${OUT_DIR}/manifest.txt"
# mirror finished into compare manifest
cp -a "${OUT_DIR}/manifest.txt" "${COMPARE_DIR}/manifest.txt"

echo
echo "Done."
echo "  info:    ${INFO_DIR}"
echo "  bench:   ${BENCH_DIR}"
echo "  summary: ${OUT_DIR}/summary.txt"
echo "  compare: ${COMPARE_DIR}   <-- share this for cross-node compare"
echo "  binary:  ${BIN} (cached; set FORCE_BUILD=1 to rebuild)"
