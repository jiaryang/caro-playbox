#!/usr/bin/env bash
#
# Sweep sglang bench_serving over input lengths x max-concurrency.
#
# Edit the variables below, then run:
#   bash sweep_bench.sh
#
set -euo pipefail

# ---- Config -----------------------------------------------------------------
# Choose which model to benchmark. Set MODEL_KEY to one of: qwen | dsv4 | glm
# (e.g.  MODEL_KEY=dsv4 bash sweep_bench.sh), or override MODEL directly.
MODEL_KEY="${MODEL_KEY:-qwen}"

# Detect GPU vendor (cuda | amd) so vendor-specific weights can be selected.
# Override with:  GPU_VENDOR=amd bash sweep_bench.sh
detect_gpu_vendor() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo cuda
  elif command -v rocm-smi >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1 || [ -e /dev/kfd ]; then
    echo amd
  else
    echo unknown
  fi
}
GPU_VENDOR="${GPU_VENDOR:-$(detect_gpu_vendor)}"

# qwen uses vendor-specific quantized weights: MXFP4 on AMD, NVFP4 on CUDA.
QWEN_MODEL_AMD="${QWEN_MODEL_AMD:-amd/Qwen3.5-397B-A17B-MXFP4}"
QWEN_MODEL_CUDA="${QWEN_MODEL_CUDA:-nvidia/Qwen3.5-397B-A17B-NVFP4}"

case "$MODEL_KEY" in
  qwen)
    case "$GPU_VENDOR" in
      cuda) MODEL="$QWEN_MODEL_CUDA" ;;
      amd)  MODEL="$QWEN_MODEL_AMD" ;;
      *)
        echo "ERROR: MODEL_KEY=qwen needs a known GPU vendor, but detection failed." >&2
        echo "       Set it explicitly, e.g. GPU_VENDOR=cuda bash sweep_bench.sh" >&2
        exit 1
        ;;
    esac
    TAG="qwen" ;;
  dsv4) MODEL="sgl-project/DeepSeek-V4-Flash-FP8";  TAG="dsv4" ;;
  glm)  MODEL="zai-org/GLM-5.2-FP8";                TAG="glm"  ;;
  *)    echo "Unknown MODEL_KEY: '$MODEL_KEY' (expected: qwen | dsv4 | glm)" >&2; exit 1 ;;
esac
# Allow a hard override of the resolved model path.
MODEL="${MODEL_OVERRIDE:-$MODEL}"
echo "Model: $MODEL  (MODEL_KEY=$MODEL_KEY, GPU_VENDOR=$GPU_VENDOR)"

# ---- Node / GPU detection ---------------------------------------------------
# NODE can be overridden:  NODE=my-host bash sweep_bench.sh
NODE="${NODE:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-node)}"

# Detect the raw GPU model string (NVIDIA via nvidia-smi, AMD via rocm-smi/rocminfo).
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

# Normalize the raw string to a short tag (mi355 | mi325 | mi300 | b200 | h200 | h100 ...).
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
    *)       echo "${s// /_}" ;;  # fallback: raw name, spaces -> underscores
  esac
}

GPU_RAW="$(detect_gpu_raw)"
# GPU can be overridden:  GPU=mi355 bash sweep_bench.sh
GPU="${GPU:-$(normalize_gpu "$GPU_RAW")}"
echo "Node: $NODE  GPU: $GPU  (detected: ${GPU_RAW:-none})"

# input:output pairs to sweep (output len is per pair).
#   8192/1024  and  71680/500
IO_PAIRS=("8192:1024" "71680:500")

CONCURRENCIES=(4 8 16 32 64 128)

RANGE_RATIO=0.8
# num-prompts = max-concurrency * this multiplier
PROMPTS_PER_CONC=10

# Where to put per-run logs and JSONL results.
RESULT_DIR="sweep_${MODEL_KEY}_${GPU}_${NODE}_$(date +%Y%m%d_%H%M%S)"
# -----------------------------------------------------------------------------

mkdir -p "$RESULT_DIR"
echo "Writing results to: $RESULT_DIR"

run_one() {
  local ilen=$1
  local olen=$2
  local conc=$3
  local nprompts=$((conc * PROMPTS_PER_CONC))
  local tag="${ilen}_o${olen}_c${conc}"
  local jsonl="${RESULT_DIR}/${TAG}_${tag}.jsonl"
  local log="${RESULT_DIR}/${TAG}_${tag}.log"

  echo
  echo "==================================================================="
  echo ">>> input_len=${ilen}  output_len=${olen}  max_concurrency=${conc}  num_prompts=${nprompts}"
  echo "==================================================================="

  python3 -m sglang.bench_serving \
    --model "$MODEL" \
    --dataset-name random \
    --random-input "$ilen" \
    --random-output "$olen" \
    --random-range-ratio "$RANGE_RATIO" \
    --max-concurrency "$conc" \
    --num-prompts "$nprompts" \
    --output-file "$jsonl" \
    2>&1 | tee "$log"
}

for pair in "${IO_PAIRS[@]}"; do
  ilen="${pair%%:*}"
  olen="${pair##*:}"
  for conc in "${CONCURRENCIES[@]}"; do
    # Don't abort the whole sweep if one run fails; warn and continue.
    run_one "$ilen" "$olen" "$conc" \
      || echo "WARN: run failed for input_len=${ilen} output_len=${olen} conc=${conc}, continuing..."
  done
done

echo
echo "All runs done. Summarizing -> ${RESULT_DIR}/summary.txt"
python3 - "$RESULT_DIR" "$TAG" "$MODEL" "$NODE" "$GPU" <<'PY' | tee "${RESULT_DIR}/summary.txt"
import glob, json, os, sys

result_dir = sys.argv[1]
prefix = sys.argv[2]
model = sys.argv[3]
node = sys.argv[4]
gpu = sys.argv[5]
print(f"Model: {model}")
print(f"Node:  {node}    GPU: {gpu}")
print()
rows = []
for path in glob.glob(os.path.join(result_dir, f"{prefix}_*.jsonl")):
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        continue
    d = json.loads(lines[-1])  # last run in the file
    rows.append(d)

# Sort by input length, then by configured concurrency.
if not rows:
    print("No result JSONL files found; nothing to summarize.")
    sys.exit(0)

rows.sort(key=lambda d: (d.get("random_input_len", 0), d.get("max_concurrency", 0)))

# Metrics of interest:
#   TTFT          -> median time-to-first-token (ms)
#   TPOT          -> median time-per-output-token (ms)
#   out_tok/s     -> output throughput (tokens/s, aggregate)
#   interactivity -> 1000 / median TPOT (tokens/s per request)
hdr = (
    f"{'input':>8} {'conc':>4} {'act_conc':>8} "
    f"{'TTFT_ms':>10} {'TPOT_ms':>9} {'out_tok/s':>10} {'interact':>9}"
)
print(hdr)
print("-" * len(hdr))

table = []
for d in rows:
    tpot = d.get("median_tpot_ms", 0) or 0
    interactivity = 1000.0 / tpot if tpot else 0.0
    print(
        f"{d.get('random_input_len',0):>8} "
        f"{d.get('max_concurrency',0):>4} "
        f"{d.get('concurrency',0):>8.2f} "
        f"{d.get('median_ttft_ms',0):>10.1f} "
        f"{tpot:>9.3f} "
        f"{d.get('output_throughput',0):>10.2f} "
        f"{interactivity:>9.2f}"
    )
    table.append(
        {
            "input_len": d.get("random_input_len", 0),
            "max_concurrency": d.get("max_concurrency", 0),
            "actual_concurrency": round(d.get("concurrency", 0), 2),
            "TTFT_median_ms": round(d.get("median_ttft_ms", 0), 1),
            "TPOT_median_ms": round(tpot, 3),
            "output_tok_per_s": round(d.get("output_throughput", 0), 2),
            "interactivity_tok_per_s": round(interactivity, 2),
        }
    )

# Build per-input-length summaries: rows = max_concurrency, cols = 4 metrics.
import pandas as pd

df = pd.DataFrame(table)

METRIC_COLS = [
    "TTFT_median_ms",
    "TPOT_median_ms",
    "output_tok_per_s",
    "interactivity_tok_per_s",
]


def pivot_for(input_len):
    sub = df[df["input_len"] == input_len].copy()
    sub = sub.sort_values("max_concurrency").set_index("max_concurrency")
    return sub[METRIC_COLS]


input_lens = sorted(df["input_len"].unique())

# CSV: one pivoted file per input length (CSV is single-table only).
for il in input_lens:
    p = os.path.join(result_dir, f"summary_{il}.csv")
    pivot_for(il).to_csv(p, index_label="max_concurrency")
    print(f"\nCSV written: {p}")

# Excel: one workbook, one sheet per input length (rows=concurrency, cols=metrics).
xlsx_path = os.path.join(result_dir, "summary.xlsx")
try:
    with pd.ExcelWriter(xlsx_path) as writer:
        for il in input_lens:
            sheet = f"in_{il}"
            pivot_for(il).to_excel(writer, sheet_name=sheet, index_label="max_concurrency")
    print(f"Excel written: {xlsx_path} (one sheet per input length)")
except ImportError:
    print("Excel (.xlsx) skipped: openpyxl not installed.")
    print("  Install with: pip install openpyxl   (CSV files above already open in Excel)")
PY

# ---- Package everything (xlsx + csv + txt + logs + jsonl) into one archive ----
# Prefer zip; fall back to tar.gz when zip is not installed.
if command -v zip >/dev/null 2>&1; then
  ARCHIVE="${RESULT_DIR}.zip"
  zip -r -q "$ARCHIVE" "$RESULT_DIR"
else
  echo "NOTE: 'zip' not found, using tar.gz instead (install with: apt-get install -y zip)."
  ARCHIVE="${RESULT_DIR}.tar.gz"
  tar -czf "$ARCHIVE" "$RESULT_DIR"
fi
echo
echo "Packaged results -> $(readlink -f "$ARCHIVE")"
echo
echo "To download to your local machine, run this FROM your local PowerShell"
echo "(replace <YOUR_LOCAL_DEST_DIR> with your own target folder):"
echo "  scp $(whoami)@$(hostname):$(readlink -f "$ARCHIVE") \"<YOUR_LOCAL_DEST_DIR>\""
