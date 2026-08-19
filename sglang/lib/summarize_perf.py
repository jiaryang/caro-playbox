#!/usr/bin/env python3
import csv
import glob
import json
import os
import re
import statistics
import sys


def num_gpus_for(d, default_num_gpus):
    tp = d.get("server_info", {}).get("tp_size")
    if tp:
        return int(tp)
    if default_num_gpus:
        return default_num_gpus
    return 1


def accept_length_for(d, jsonl_path):
    val = d.get("accept_length")
    if val is not None:
        return float(val)
    log_path = jsonl_path[:-6] + ".log" if jsonl_path.endswith(".jsonl") else jsonl_path + ".log"
    if os.path.isfile(log_path):
        m = re.search(r"Accept length:\s+([\d.]+)", open(log_path).read())
        if m:
            return float(m.group(1))
    return None


def run_num(path):
    m = re.search(r"_run(\d+)\.jsonl$", os.path.basename(path))
    return int(m.group(1)) if m else 0


# Report metric columns (CSV / Excel), in display order.
METRIC_COLS = [
    "Interactivity (tok/s/user)",
    "Token TPUT per GPU",
    "MedianTTFT",
    "MedianTPOT",
    "MedianITL",
]

METRIC_DIGITS = {
    "Interactivity (tok/s/user)": 2,
    "Token TPUT per GPU": 2,
    "MedianTTFT": 1,
    "MedianTPOT": 3,
    "MedianITL": 3,
    "accept_length": 2,
    "actual_concurrency": 2,
}

DISP_SUFFIXES = ("std", "min", "max", "spread", "cv_pct")


def metrics_from(d, path, default_num_gpus):
    tpot = d.get("median_tpot_ms", 0) or 0
    itl = d.get("median_itl_ms", 0) or 0
    interactivity = 1000.0 / tpot if tpot else 0.0
    ng = num_gpus_for(d, default_num_gpus)
    total_tput = d.get("total_throughput", 0) or 0
    total_per_gpu = total_tput / ng if ng else 0.0
    accept_len = accept_length_for(d, path)
    ttft = d.get("median_ttft_ms", 0) or 0
    row = {
        "input_len": d.get("random_input_len", 0),
        "max_concurrency": d.get("max_concurrency", 0),
        "actual_concurrency": round(d.get("concurrency", 0), 2),
        "run": run_num(path),
        "Interactivity (tok/s/user)": round(interactivity, 2),
        "Token TPUT per GPU": round(total_per_gpu, 2),
        "MedianTTFT": round(ttft, 1),
        "MedianTPOT": round(tpot, 3),
        "MedianITL": round(itl, 3),
        "path": path,
    }
    if accept_len is not None:
        row["accept_length"] = round(accept_len, 2)
    return row


def dispersion(vals, ndigits):
    """mean / std / min / max / spread / cv_pct for a list of numbers."""
    mean = statistics.mean(vals)
    amin = min(vals)
    amax = max(vals)
    spread = amax - amin
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    cv_pct = (100.0 * std / abs(mean)) if mean else 0.0
    return {
        "mean": round(mean, ndigits),
        "std": round(std, ndigits),
        "min": round(amin, ndigits),
        "max": round(amax, ndigits),
        "spread": round(spread, ndigits),
        "cv_pct": round(cv_pct, 2),
    }


def attach_metric_stats(agg, col, vals):
    stats = dispersion(vals, METRIC_DIGITS.get(col, 3))
    agg[col] = stats["mean"]
    for suf in DISP_SUFFIXES:
        agg[f"{col}_{suf}"] = stats[suf]
    return stats


def aggregate_rows(rows, show_accept):
    """Aggregate runs that share (input_len, max_concurrency); keep dispersion."""
    groups = {}
    for r in rows:
        key = (r["input_len"], r["max_concurrency"])
        groups.setdefault(key, []).append(r)

    out = []
    for (ilen, conc), group in sorted(groups.items()):
        agg = {
            "input_len": ilen,
            "max_concurrency": conc,
            "n_runs": len(group),
        }
        attach_metric_stats(agg, "actual_concurrency", [g["actual_concurrency"] for g in group])
        for col in METRIC_COLS:
            attach_metric_stats(agg, col, [g[col] for g in group])
        if show_accept:
            accepts = [g["accept_length"] for g in group if g.get("accept_length") is not None]
            if accepts:
                attach_metric_stats(agg, "accept_length", accepts)
            else:
                agg["accept_length"] = None
        out.append(agg)
    return out


def metric_fieldnames(metric_cols, with_dispersion):
    fields = []
    for col in metric_cols:
        fields.append(col)
        if with_dispersion:
            fields.extend(f"{col}_{suf}" for suf in DISP_SUFFIXES)
    return fields


def print_stat_line(name, stats, mean_fmt, disp_fmt=None):
    disp_fmt = disp_fmt or mean_fmt
    print(
        f"  {name + ':':28s}"
        f"mean={stats['mean']:{mean_fmt}}  "
        f"std={stats['std']:{disp_fmt}}  "
        f"min={stats['min']:{disp_fmt}}  "
        f"max={stats['max']:{disp_fmt}}  "
        f"spread={stats['spread']:{disp_fmt}}  "
        f"cv={stats['cv_pct']:.2f}%"
    )


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

    entries = []
    for path in glob.glob(os.path.join(result_dir, f"{prefix}_*.jsonl")):
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            entries.append((json.loads(lines[-1]), path))

    if not entries:
        print("No result JSONL files found; nothing to summarize.")
        return 0

    rows = [metrics_from(d, path, default_num_gpus) for d, path in entries]
    rows.sort(key=lambda r: (r["input_len"], r["max_concurrency"], r["run"]))
    show_accept = any(r.get("accept_length") is not None for r in rows)
    multi_run = any(r["run"] > 0 for r in rows) or len(
        {(r["input_len"], r["max_concurrency"]) for r in rows}
    ) < len(rows)

    hdr = f"{'input':>8} {'conc':>4}"
    if multi_run:
        hdr += f" {'run':>4}"
    hdr += (
        f" {'act_conc':>8} "
        f"{'interact':>9} {'tot/gpu':>10} {'TTFT_ms':>10} {'TPOT_ms':>9} {'ITL_ms':>9}"
    )
    if show_accept:
        hdr += f" {'accept':>7}"
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        line = f"{r['input_len']:>8} {r['max_concurrency']:>4}"
        if multi_run:
            line += f" {r['run'] or 1:>4}"
        line += (
            f" {r['actual_concurrency']:>8.2f} "
            f"{r['Interactivity (tok/s/user)']:>9.2f} "
            f"{r['Token TPUT per GPU']:>10.2f} "
            f"{r['MedianTTFT']:>10.1f} "
            f"{r['MedianTPOT']:>9.3f} "
            f"{r['MedianITL']:>9.3f}"
        )
        if show_accept:
            al = r.get("accept_length")
            line += f" {al:>7.2f}" if al is not None else f" {'-':>7}"
        print(line)

    metric_cols = list(METRIC_COLS)
    if show_accept:
        metric_cols.append("accept_length")

    agg_rows = aggregate_rows(rows, show_accept)
    if multi_run:
        print()
        print("Aggregate (mean + dispersion per input/conc):")
        print("  Rule of thumb: cv < ~5% and relative spread (spread/mean) < ~10% usually OK.")
        ahdr = (
            f"{'input':>8} {'conc':>4} {'n':>3} "
            f"{'interact':>9} {'cv%':>6} {'tot/gpu':>10} {'cv%':>6} "
            f"{'TTFT_ms':>10} {'cv%':>6} {'TPOT_ms':>9} {'cv%':>6}"
        )
        print(ahdr)
        print("-" * len(ahdr))
        for r in agg_rows:
            print(
                f"{r['input_len']:>8} {r['max_concurrency']:>4} {r['n_runs']:>3} "
                f"{r['Interactivity (tok/s/user)']:>9.2f} {r['Interactivity (tok/s/user)_cv_pct']:>5.2f}% "
                f"{r['Token TPUT per GPU']:>10.2f} {r['Token TPUT per GPU_cv_pct']:>5.2f}% "
                f"{r['MedianTTFT']:>10.1f} {r['MedianTTFT_cv_pct']:>5.2f}% "
                f"{r['MedianTPOT']:>9.3f} {r['MedianTPOT_cv_pct']:>5.2f}%"
            )

        print()
        print("Dispersion detail:")
        for r in agg_rows:
            print(f"  input={r['input_len']}  conc={r['max_concurrency']}  n={r['n_runs']}")
            for col, fmt in (
                ("Interactivity (tok/s/user)", ".2f"),
                ("Token TPUT per GPU", ".2f"),
                ("MedianTTFT", ".1f"),
                ("MedianTPOT", ".3f"),
                ("MedianITL", ".3f"),
            ):
                stats = {
                    "mean": r[col],
                    "std": r[f"{col}_std"],
                    "min": r[f"{col}_min"],
                    "max": r[f"{col}_max"],
                    "spread": r[f"{col}_spread"],
                    "cv_pct": r[f"{col}_cv_pct"],
                }
                print_stat_line(col, stats, fmt)
            if show_accept and r.get("accept_length") is not None:
                stats = {
                    "mean": r["accept_length"],
                    "std": r["accept_length_std"],
                    "min": r["accept_length_min"],
                    "max": r["accept_length_max"],
                    "spread": r["accept_length_spread"],
                    "cv_pct": r["accept_length_cv_pct"],
                }
                print_stat_line("accept_length", stats, ".2f")

    input_lens = sorted({r["input_len"] for r in agg_rows})
    summary_fields = ["max_concurrency"] + metric_fieldnames(metric_cols, with_dispersion=multi_run)

    for il in input_lens:
        p = os.path.join(result_dir, f"summary_{il}.csv")
        sub = [r for r in agg_rows if r["input_len"] == il]
        sub.sort(key=lambda r: r["max_concurrency"])
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for r in sub:
                w.writerow({k: r.get(k) for k in summary_fields})
        print(f"\nCSV written: {p}")

    if multi_run:
        runs_csv = os.path.join(result_dir, "summary_runs.csv")
        run_fields = ["input_len", "max_concurrency", "run"] + metric_cols
        with open(runs_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=run_fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in run_fields})
        print(f"CSV written: {runs_csv}")

        disp_csv = os.path.join(result_dir, "summary_dispersion.csv")
        disp_fields = ["input_len", "max_concurrency", "n_runs"] + metric_fieldnames(
            metric_cols, with_dispersion=True
        )
        with open(disp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=disp_fields)
            w.writeheader()
            for r in agg_rows:
                w.writerow({k: r.get(k) for k in disp_fields})
        print(f"CSV written: {disp_csv}")

    try:
        import pandas as pd

        df = pd.DataFrame(agg_rows)

        def pivot_mean(input_len):
            sub = df[df["input_len"] == input_len].copy()
            sub = sub.sort_values("max_concurrency").set_index("max_concurrency")
            return sub[metric_cols]

        def pivot_dispersion(input_len):
            cols = metric_fieldnames(metric_cols, with_dispersion=True)
            sub = df[df["input_len"] == input_len].copy()
            sub = sub.sort_values("max_concurrency").set_index("max_concurrency")
            return sub[[c for c in cols if c in sub.columns]]

        xlsx_path = os.path.join(result_dir, "summary.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            for il in input_lens:
                pivot_mean(il).to_excel(writer, sheet_name=f"in_{il}", index_label="max_concurrency")
                if multi_run:
                    pivot_dispersion(il).to_excel(
                        writer, sheet_name=f"in_{il}_disp", index_label="max_concurrency"
                    )
            if multi_run:
                pd.DataFrame(
                    [
                        {k: r.get(k) for k in (["input_len", "max_concurrency", "run"] + metric_cols)}
                        for r in rows
                    ]
                ).to_excel(writer, sheet_name="runs", index=False)
                pd.DataFrame(
                    [
                        {
                            k: r.get(k)
                            for k in (
                                ["input_len", "max_concurrency", "n_runs"]
                                + metric_fieldnames(metric_cols, with_dispersion=True)
                            )
                        }
                        for r in agg_rows
                    ]
                ).to_excel(writer, sheet_name="dispersion", index=False)
        print(f"Excel written: {xlsx_path} (one sheet per input length)")
    except ImportError:
        print("Excel (.xlsx) skipped: pandas/openpyxl not installed.")
        print("  CSV files above already contain the summary tables.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
