#!/usr/bin/env python3
import csv
import glob
import json
import os
import re
import statistics
import sys

PATTERNS = {
    "accuracy": r"Accuracy:\s*([\d.]+)",
    "invalid": r"Invalid:\s*([\d.]+)",
    "latency_s": r"Latency:\s*([\d.]+)\s*s",
    "output_tok_per_s": r"Output throughput:\s*([\d.]+)\s*token/s",
}


def parse_log(path):
    out = {"log": os.path.basename(path), "ok": False}
    if not os.path.isfile(path):
        out["error"] = "log missing"
        return out
    text = open(path).read()
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    out["ok"] = "accuracy" in out
    if not out["ok"]:
        out["error"] = "accuracy not found in log"
    return out


def run_num(path):
    m = re.search(r"run(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def stat_line(name, values, fmt):
    if not values:
        return
    if len(values) == 1:
        print(f"  {name + ':':12s}{values[0]:{fmt}}")
        return
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    print(
        f"  {name + ':':12s}mean={mean:{fmt}}  std={stdev:{fmt}}  "
        f"min={min(values):{fmt}}  max={max(values):{fmt}}  spread={max(values) - min(values):{fmt}}"
    )


def main():
    result_dir = sys.argv[1]
    prefix = sys.argv[2]
    model, node, gpu, num_shots, num_questions, parallel, num_runs = sys.argv[3:10]
    num_runs = int(num_runs)

    log_paths = sorted(
        glob.glob(os.path.join(result_dir, f"{prefix}_run*.log")),
        key=run_num,
    )
    runs = []
    for path in log_paths:
        row = {"run": run_num(path), "log": os.path.basename(path)}
        row.update(parse_log(path))
        jsonl_path = path[:-4] + ".jsonl"
        row["jsonl"] = os.path.basename(jsonl_path)
        if os.path.isfile(jsonl_path):
            with open(jsonl_path) as f:
                lines = [l for l in f if l.strip()]
            if lines:
                row["result"] = json.loads(lines[-1])
        runs.append(row)

    print(f"Model: {model}")
    print(f"Node:  {node}    GPU: {gpu}")
    print(f"num_shots: {num_shots}    num_questions: {num_questions}    parallel: {parallel}")
    print(f"runs: {num_runs}")
    print()

    hdr = f"{'run':>4} {'accuracy':>9} {'invalid':>9} {'latency_s':>10} {'out_tok/s':>10} {'status':>8}"
    print(hdr)
    print("-" * len(hdr))

    ok_runs = []
    for row in runs:
        if row.get("ok"):
            ok_runs.append(row)
            print(
                f"{row['run']:>4} "
                f"{row.get('accuracy', float('nan')):>9.3f} "
                f"{row.get('invalid', float('nan')):>9.3f} "
                f"{row.get('latency_s', float('nan')):>10.1f} "
                f"{row.get('output_tok_per_s', float('nan')):>10.1f} "
                f"{'ok':>8}"
            )
        else:
            print(f"{row['run']:>4} {'-':>9} {'-':>9} {'-':>10} {'-':>10} {'FAIL':>8}")

    if ok_runs:
        print()
        print("Aggregate (successful runs only):")
        for key, fmt in (
            ("accuracy", ".3f"),
            ("invalid", ".3f"),
            ("latency_s", ".1f"),
            ("output_tok_per_s", ".1f"),
        ):
            stat_line(key, [r[key] for r in ok_runs if key in r], fmt)

    csv_path = os.path.join(result_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "log",
                "jsonl",
                "accuracy",
                "invalid",
                "latency_s",
                "output_tok_per_s",
                "ok",
                "error",
            ],
        )
        w.writeheader()
        for row in runs:
            w.writerow(
                {
                    "run": row["run"],
                    "log": row.get("log"),
                    "jsonl": row.get("jsonl"),
                    "accuracy": row.get("accuracy"),
                    "invalid": row.get("invalid"),
                    "latency_s": row.get("latency_s"),
                    "output_tok_per_s": row.get("output_tok_per_s"),
                    "ok": row.get("ok"),
                    "error": row.get("error"),
                }
            )
    print(f"\nCSV written: {csv_path}")

    try:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "run": row["run"],
                    "accuracy": row.get("accuracy"),
                    "invalid": row.get("invalid"),
                    "latency_s": row.get("latency_s"),
                    "output_tok_per_s": row.get("output_tok_per_s"),
                    "log": row.get("log"),
                    "jsonl": row.get("jsonl"),
                    "ok": row.get("ok"),
                }
                for row in runs
            ]
        ).sort_values("run")
        xlsx_path = os.path.join(result_dir, "summary.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, sheet_name="runs", index=False)
            if ok_runs and len(ok_runs) > 1:
                agg = {}
                for key in ("accuracy", "invalid", "latency_s", "output_tok_per_s"):
                    vals = [r[key] for r in ok_runs if key in r]
                    if vals:
                        agg[key + "_mean"] = statistics.mean(vals)
                        agg[key + "_std"] = statistics.stdev(vals)
                        agg[key + "_min"] = min(vals)
                        agg[key + "_max"] = max(vals)
                pd.DataFrame([agg]).to_excel(writer, sheet_name="aggregate", index=False)
        print(f"Excel written: {xlsx_path}")
    except ImportError:
        print("Excel (.xlsx) skipped: pandas/openpyxl not installed.")

    return 0 if runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
