#!/usr/bin/env python3
import csv
import glob
import json
import os
import sys


def num_gpus_for(d, default_num_gpus):
    tp = d.get("server_info", {}).get("tp_size")
    if tp:
        return int(tp)
    if default_num_gpus:
        return default_num_gpus
    return 1


def main():
    result_dir = sys.argv[1]
    prefix = sys.argv[2]
    model = sys.argv[3]
    node = sys.argv[4]
    gpu = sys.argv[5]
    default_num_gpus = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None

    print(f"Model: {model}")
    print(f"Node:  {node}    GPU: {gpu}")
    if default_num_gpus:
        print(f"Num GPUs (TP, default): {default_num_gpus}")
    print()

    rows = []
    for path in glob.glob(os.path.join(result_dir, f"{prefix}_*.jsonl")):
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            rows.append(json.loads(lines[-1]))

    if not rows:
        print("No result JSONL files found; nothing to summarize.")
        return 0

    rows.sort(key=lambda d: (d.get("random_input_len", 0), d.get("max_concurrency", 0)))

    hdr = (
        f"{'input':>8} {'conc':>4} {'act_conc':>8} "
        f"{'TTFT_ms':>10} {'TPOT_ms':>9} {'out_tok/s':>10} {'tot/gpu':>10} {'interact':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    table = []
    for d in rows:
        tpot = d.get("median_tpot_ms", 0) or 0
        interactivity = 1000.0 / tpot if tpot else 0.0
        ng = num_gpus_for(d, default_num_gpus)
        total_tput = d.get("total_throughput", 0) or 0
        total_per_gpu = total_tput / ng if ng else 0.0
        print(
            f"{d.get('random_input_len', 0):>8} "
            f"{d.get('max_concurrency', 0):>4} "
            f"{d.get('concurrency', 0):>8.2f} "
            f"{d.get('median_ttft_ms', 0):>10.1f} "
            f"{tpot:>9.3f} "
            f"{d.get('output_throughput', 0):>10.2f} "
            f"{total_per_gpu:>10.2f} "
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
                "total_tok_per_gpu": round(total_per_gpu, 2),
                "interactivity_tok_per_s": round(interactivity, 2),
            }
        )

    metric_cols = [
        "TTFT_median_ms",
        "TPOT_median_ms",
        "output_tok_per_s",
        "total_tok_per_gpu",
        "interactivity_tok_per_s",
    ]
    input_lens = sorted({r["input_len"] for r in table})

    for il in input_lens:
        p = os.path.join(result_dir, f"summary_{il}.csv")
        sub = [r for r in table if r["input_len"] == il]
        sub.sort(key=lambda r: r["max_concurrency"])
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["max_concurrency"] + metric_cols)
            w.writeheader()
            for r in sub:
                w.writerow({"max_concurrency": r["max_concurrency"], **{k: r[k] for k in metric_cols}})
        print(f"\nCSV written: {p}")

    try:
        import pandas as pd

        df = pd.DataFrame(table)

        def pivot_for(input_len):
            sub = df[df["input_len"] == input_len].copy()
            sub = sub.sort_values("max_concurrency").set_index("max_concurrency")
            return sub[metric_cols]

        xlsx_path = os.path.join(result_dir, "summary.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            for il in input_lens:
                pivot_for(il).to_excel(writer, sheet_name=f"in_{il}", index_label="max_concurrency")
        print(f"Excel written: {xlsx_path} (one sheet per input length)")
    except ImportError:
        print("Excel (.xlsx) skipped: pandas/openpyxl not installed.")
        print("  CSV files above already contain the summary tables.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
