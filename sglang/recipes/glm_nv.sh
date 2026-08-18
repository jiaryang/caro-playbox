# GLM-5.2 NVFP4 (NVIDIA) server recipe for SGLang.
# Sourced by recipes/glm.sh; args expand when functions run
# (MODEL/TP/HOST/PORT must already be set).

glm_nv_base_server_args() {
  cat <<EOF
--model ${MODEL}
--tp ${TP}
--trust-remote-code
--data-parallel-size 1
--expert-parallel-size 1
--disable-radix-cache
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3
--nsa-decode-backend trtllm
--nsa-prefill-backend trtllm
--moe-runner-backend flashinfer_trtllm
--enable-flashinfer-allreduce-fusion
--cuda-graph-max-bs 256
--max-prefill-tokens 32768
--chunked-prefill-size 32768
--mem-fraction-static 0.8
--stream-interval 30
--scheduler-recv-interval 10
--tokenizer-worker-num 6
--tokenizer-path ${MODEL}
--host ${HOST}
--port ${PORT}
EOF
}
