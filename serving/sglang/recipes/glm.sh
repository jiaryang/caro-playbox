# GLM-5.2 MXFP4 server recipes for SGLang.
# Sourced by suites/glm/run_env_suite.sh after MODEL/TP/HOST/PORT are set.

glm_base_server_args() {
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
