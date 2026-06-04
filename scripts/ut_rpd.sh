#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUMMARIZE="/dockerx/PerformanceCorrelationDebugging/prof_tools/summarize_kernels_rpd.py"

FRAMEWORK="${1:?Usage: $0 <jax|tf>}"

case "${FRAMEWORK}" in
  jax)
    WORKDIR="${REPO_ROOT}/jax"
    GEMM_SCRIPT="gemm_jax.py"
    RPD_FILE="rpd_tracer_output_trace.rpd"
    ;;
  tf)
    WORKDIR="${REPO_ROOT}/tf"
    GEMM_SCRIPT="tf_gemm.py"
    RPD_FILE="tf_trace.rpd"
    ;;
  *)
    echo "Unknown framework: ${FRAMEWORK} (expected jax or tf)"
    exit 1
    ;;
esac

cd "${WORKDIR}"
rm -f rpd_tracer_output_trace.*
python "${GEMM_SCRIPT}"
python "${SUMMARIZE}" "${RPD_FILE}"
