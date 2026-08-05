# Shared helpers for sweep_bench.sh — do not run directly.

bench_common_init() {
  BENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$BENCH_SCRIPT_DIR/../../sglang" 2>/dev/null && pwd)}"
}

bench_require_sglang_root() {
  if [[ ! -f "${SGLANG_ROOT}/benchmark/gsm8k/bench_sglang.py" ]]; then
    echo "ERROR: cannot find benchmark/gsm8k/bench_sglang.py under SGLANG_ROOT=${SGLANG_ROOT}" >&2
    echo "       Set SGLANG_ROOT to your sglang checkout, e.g. SGLANG_ROOT=/path/to/sglang bash $0" >&2
    exit 1
  fi
}

detect_gpu_vendor() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo cuda
  elif command -v rocm-smi >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1 || [ -e /dev/kfd ]; then
    echo amd
  else
    echo unknown
  fi
}

detect_gpu_raw() {
  local name=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  fi
  if [[ -z "$name" ]] && command -v rocm-smi >/dev/null 2>&1; then
    name="$(rocm-smi --showproductname 2>/dev/null | grep -ioE 'MI[0-9]+[A-Z]*' | head -n1)"
  fi
  if [[ -z "$name" ]] && command -v rocminfo >/dev/null 2>&1; then
    name="$(rocminfo 2>/dev/null | grep -ioE 'MI[0-9]+[A-Z]*' | head -n1)"
  fi
  echo "$name"
}

normalize_gpu() {
  local s
  s="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$s" in
    *mi355*) echo "mi355" ;;
    *mi325*) echo "mi325" ;;
    *mi300*) echo "mi300" ;;
    *b200*)  echo "b200"  ;;
    *h200*)  echo "h200"  ;;
    *h100*)  echo "h100"  ;;
    "")      echo "unknown-gpu" ;;
    *)       echo "${s// /_}" ;;
  esac
}

# Resolve MODEL_KEY -> MODEL, TAG, NUM_GPUS default.
# Requires: MODEL_KEY, GPU_VENDOR (set by caller).
bench_resolve_model() {
  QWEN_MODEL_AMD="${QWEN_MODEL_AMD:-amd/Qwen3.5-397B-A17B-MXFP4}"
  QWEN_MODEL_CUDA="${QWEN_MODEL_CUDA:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
  GLM_MODEL_AMD="${GLM_MODEL_AMD:-amd/GLM-5.2-MXFP4}"
  GLM_MODEL_CUDA="${GLM_MODEL_CUDA:-nvidia/GLM-5.2-NVFP4}"

  case "$MODEL_KEY" in
    qwen)
      case "$GPU_VENDOR" in
        cuda) MODEL="$QWEN_MODEL_CUDA" ;;
        amd)  MODEL="$QWEN_MODEL_AMD" ;;
        *)
          echo "ERROR: MODEL_KEY=qwen needs a known GPU vendor, but detection failed." >&2
          echo "       Set it explicitly, e.g. GPU_VENDOR=cuda bash $0" >&2
          exit 1
          ;;
      esac
      TAG="qwen" ;;
    dsv4) MODEL="sgl-project/DeepSeek-V4-Flash-FP8"; TAG="dsv4" ;;
    glm)
      case "$GPU_VENDOR" in
        cuda) MODEL="$GLM_MODEL_CUDA" ;;
        amd)  MODEL="$GLM_MODEL_AMD" ;;
        *)
          echo "ERROR: MODEL_KEY=glm needs a known GPU vendor, but detection failed." >&2
          echo "       Set it explicitly, e.g. GPU_VENDOR=amd bash $0" >&2
          exit 1
          ;;
      esac
      TAG="glm" ;;
    *)
      echo "Unknown MODEL_KEY: '$MODEL_KEY' (expected: qwen | dsv4 | glm)" >&2
      exit 1
      ;;
  esac

  MODEL="${MODEL_OVERRIDE:-$MODEL}"
  case "$MODEL_KEY" in
    glm) NUM_GPUS="${NUM_GPUS:-4}" ;;
    *)   NUM_GPUS="${NUM_GPUS:-}" ;;
  esac
}

bench_detect_node_gpu() {
  NODE="${NODE:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-node)}"
  GPU_RAW="$(detect_gpu_raw)"
  GPU="${GPU:-$(normalize_gpu "$GPU_RAW")}"
}

# Query /server_info and return "mtp" or "nomtp".
bench_detect_mtp_from_server() {
  local host=$1
  local port=$2
  python3 - "$host" "$port" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(
        f"http://{host}:{port}/server_info", timeout=5
    ) as resp:
        data = json.load(resp)
except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
    print("nomtp", end="")
    raise SystemExit(0)

if isinstance(data, dict) and "decode" in data:
    data = data["decode"][0]

algo = data.get("speculative_algorithm") if isinstance(data, dict) else None
if algo:
    print("mtp", end="")
else:
    print("nomtp", end="")
PY
}

# Sets MTP_TAG to "mtp" or "nomtp".
# Optional MTP_OVERRIDE (from --mtp / --no-mtp) skips server auto-detect.
bench_resolve_mtp_tag() {
  if [[ -n "${MTP_OVERRIDE:-}" ]]; then
    MTP_TAG="$MTP_OVERRIDE"
    return
  fi

  local host="${BENCH_SERVER_HOST:-127.0.0.1}"
  local port="${BENCH_SERVER_PORT:-30000}"
  MTP_TAG="$(bench_detect_mtp_from_server "$host" "$port")"
}
