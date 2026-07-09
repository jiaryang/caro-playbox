#!/usr/bin/env bash
#
# Sweep sglang bench_serving over input lengths x max-concurrency.
#
# Edit the variables below, then run:
#   bash sweep_bench.sh
#
set -euo pipefail

# ---- Config -----------------------------------------------------------------
HOST=127.0.0.1
PORT=8000
MODEL=sgl-project/DeepSeek-V4-Flash-FP8

INPUT_LENS=(131072 262144)
CONCURRENCIES=(1 2 4 6 8)

OUTPUT_LEN=1024
RANGE_RATIO=1
NUM_PROMPTS=20
WARMUP_REQUESTS=2

# Where to put per-run logs and JSONL results.
RESULT_DIR="sweep_$(date +%Y%m%d_%H%M%S)"
# -----------------------------------------------------------------------------

mkdir -p "$RESULT_DIR"
echo "Writing results to: $RESULT_DIR"

run_one() {
  local ilen=$1
  local conc=$2
  local tag="${ilen}_c${conc}"
  local jsonl="${RESULT_DIR}/dsv4_${tag}.jsonl"
  local log="${RESULT_DIR}/dsv4_${tag}.log"

  echo
  echo "==================================================================="
  echo ">>> input_len=${ilen}  max_concurrency=${conc}  num_prompts=${NUM_PROMPTS}"
  echo "==================================================================="

  python3 -m sglang.bench_serving \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ilen" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio "$RANGE_RATIO" \
    --max-concurrency "$conc" \
    --num-prompts "$NUM_PROMPTS" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --output-file "$jsonl" \
    2>&1 | tee "$log"
}

for ilen in "${INPUT_LENS[@]}"; do
  for conc in "${CONCURRENCIES[@]}"; do
    # Don't abort the whole sweep if one run fails; warn and continue.
    run_one "$ilen" "$conc" || echo "WARN: run failed for input_len=${ilen} conc=${conc}, continuing..."
  done
done

echo
echo "All runs done. Summarizing -> ${RESULT_DIR}/summary.txt"
python3 - "$RESULT_DIR" <<'PY' | tee "${RESULT_DIR}/summary.txt"
import glob, json, os, sys

result_dir = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(result_dir, "dsv4_*.jsonl")):
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

# ---- Package everything (xlsx + csv + txt + logs + jsonl) into one zip --------
ARCHIVE="${RESULT_DIR}.zip"
zip -r -q "$ARCHIVE" "$RESULT_DIR"
echo
echo "Packaged results -> $(readlink -f "$ARCHIVE")"
echo
echo "To download to your local machine, run this FROM your local PowerShell:"
echo "  scp $(whoami)@$(hostname):$(readlink -f "$ARCHIVE") \"C:\\Users\\jiaryang\\OneDrive - Advanced Micro Devices Inc\\2_task\\59_featherless\\\""
