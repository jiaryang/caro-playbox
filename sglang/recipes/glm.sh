# GLM-5.2 server recipes for SGLang (AMD MXFP4 + NVIDIA NVFP4).
# Sourced by suites/glm/run_env_suite.sh; args expand when functions run
# (MODEL/TP/HOST/PORT/GPU_VENDOR must already be set).

_RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./glm_nv.sh
source "${_RECIPE_DIR}/glm_nv.sh"

glm_amd_base_server_args() {
  cat <<EOF
--model ${MODEL}
--tp ${TP}
--trust-remote-code
--tool-call-parser glm47
--reasoning-parser glm45
--mem-fraction-static 0.85
--kv-cache-dtype fp8_e4m3
--disable-radix-cache
--chunked-prefill-size 16384
--dsa-prefill-backend triton
--dsa-decode-backend triton
--enable-aiter-allreduce-fusion
--tokenizer-worker-num 8
--host ${HOST}
--port ${PORT}
EOF
}

glm_eagle_server_args() {
  cat <<EOF
--speculative-algorithm EAGLE
--speculative-num-draft-tokens 4
--speculative-num-steps 3
--speculative-eagle-topk 1
EOF
}

# Dispatch by GPU_VENDOR (cuda|amd). Empty -> amd for backward compat.
glm_base_server_args() {
  case "${GPU_VENDOR:-amd}" in
    cuda|nv|nvidia)
      glm_nv_base_server_args
      ;;
    amd|rocm)
      glm_amd_base_server_args
      ;;
    *)
      echo "ERROR: unknown GPU_VENDOR='${GPU_VENDOR}' (want: cuda|amd)" >&2
      return 1
      ;;
  esac
}
