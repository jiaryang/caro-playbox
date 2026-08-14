"""MTP vs non-MTP: what one decode step costs and what it delivers.

A decode step is not comparable across the two modes on wall time alone. An MTP
step is longer but emits ``accept_length`` tokens per request instead of one, so
the honest metric is output tokens per second. This script pairs the GPU traces
of both runs with the client-side benchmark summaries that carry the accepted
length, and writes one workbook holding both views.

    python compare_modes.py --mtp-root 3_0811_MTP --nonmtp-root 3_0811_nonMTP -o out.xlsx

Profile directories are discovered as ``profiles_<context>_<gpu>[_<variant>]``
and matched against ``perf_glm_<mode>_<gpu>[_<variant>]_*/summary_<context>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtp_profile.kernels import GLM52_RULES, KernelClassifier
from mtp_profile.report import r2, ratio, write_excel
from mtp_profile.sweep import build_run, discover

PROFILE_RE = re.compile(r"^profiles_(8k|8192|70000)_(b200|mi355)(?:_(mr\d+))?$")
PERF_RE = re.compile(r"^perf_glm_(mtp|nomtp)_(b200|mi355)(?:_(mr\d+))?_")
CONTEXT_NAMES = {"8k": "8k", "8192": "8k", "70000": "70k"}
CONTEXT_CSV = {"8k": "8192", "70k": "70000"}


def find_profile_dirs(root: str) -> dict:
    """-> {(context, gpu, variant): path}."""
    out = {}
    for entry in sorted(os.listdir(root)):
        m = PROFILE_RE.match(entry)
        if not m:
            continue
        ctx, gpu, variant = m.groups()
        out[(CONTEXT_NAMES[ctx], gpu, variant or "")] = os.path.join(root, entry)
    return out


def find_perf(root: str, mode: str) -> dict:
    """-> {(mode, context, gpu, variant, conc): row} from the benchmark summaries.

    The mode belongs in the key: both roots hold the same GPU and concurrency
    combinations, so a mode-less key would let one run overwrite the other.
    """
    out = {}
    for entry in sorted(os.listdir(root)):
        m = PERF_RE.match(entry)
        if not m or m.group(1) != mode:
            continue
        _, gpu, variant = m.groups()
        for path in glob.glob(os.path.join(root, entry, "summary_*.csv")):
            ctx_raw = os.path.basename(path)[len("summary_") : -len(".csv")]
            ctx = CONTEXT_NAMES.get(ctx_raw)
            if ctx is None:
                continue
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    conc = int(row["max_concurrency"])
                    out[(mode, ctx, gpu, variant or "", conc)] = row
    return out


def measure(root: str, mode: str, classifier) -> list:
    """Load every DECODE trace under ``root`` and return one record per run."""
    records = []
    for (ctx, gpu, variant), directory in sorted(find_profile_dirs(root).items()):
        try:
            found = discover(directory)
        except ValueError as exc:
            print(f"[skip] {directory}: {exc}")
            continue
        label = gpu + (f"_{variant}" if variant else "")
        for conc in sorted(found):
            run = build_run(label, conc, found[conc], classifier)
            records.append(
                {
                    "Context": ctx,
                    "GPU": gpu,
                    "Variant": variant,
                    "Mode": "MTP" if mode == "mtp" else "non-MTP",
                    "Conc": conc,
                    "bs": run.bs,
                    "wall": run.decode_wall,
                    "kernel": run.decode_kernel,
                }
            )
            print(
                f"  {ctx:>4} {label:<12} c{conc:<3} bs={run.bs:<3} "
                f"wall={run.decode_wall:6.2f} kernel={run.decode_kernel:6.2f}"
            )
    return records


def attach_client(records: list, perf: dict) -> None:
    for rec in records:
        mode = "mtp" if rec["Mode"] == "MTP" else "nomtp"
        row = perf.get((mode, rec["Context"], rec["GPU"], rec["Variant"], rec["Conc"]))
        rec["accept_length"] = (
            float(row["accept_length"])
            if row and row.get("accept_length")
            else (1.0 if rec["Mode"] == "non-MTP" else None)
        )
        rec["TPOT_ms"] = float(row["TPOT_median_ms"]) if row else None
        rec["client_out_tok_s"] = float(row["output_tok_per_s"]) if row else None
        accept = rec["accept_length"]
        rec["gpu_out_tok_s"] = (
            rec["bs"] * accept / rec["wall"] * 1000.0 if accept and rec["wall"] else None
        )


def gain(a, b):
    """Ratio that tolerates a missing client summary on either side."""
    return ratio(a, b) if a is not None and b else None


def build_runs_sheet(records: list) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append(
            {
                "Context": rec["Context"],
                "GPU": rec["GPU"],
                "Variant": rec["Variant"] or "default",
                "Mode": rec["Mode"],
                "Conc": rec["Conc"],
                "bs": rec["bs"],
                "decode wall": r2(rec["wall"]),
                "decode kernel": r2(rec["kernel"]),
                "accept length": r2(rec["accept_length"]),
                "GPU out tok/s": r2(rec["gpu_out_tok_s"]),
                "client TPOT ms": r2(rec["TPOT_ms"]),
                "client out tok/s": r2(rec["client_out_tok_s"]),
            }
        )
    return pd.DataFrame(rows)


def build_mode_gain(records: list) -> pd.DataFrame:
    """MTP against non-MTP on the same GPU, context and concurrency."""
    by_key = {}
    for rec in records:
        by_key.setdefault((rec["Context"], rec["GPU"], rec["Conc"]), {})[rec["Mode"]] = rec

    rows = []
    for (ctx, gpu, conc), modes in sorted(by_key.items()):
        mtp, plain = modes.get("MTP"), modes.get("non-MTP")
        if not mtp or not plain:
            continue
        rows.append(
            {
                "Context": ctx,
                "GPU": gpu,
                "Conc": conc,
                "MTP bs": mtp["bs"],
                "non-MTP bs": plain["bs"],
                "bs match": "yes" if mtp["bs"] == plain["bs"] else "NO",
                "MTP wall": r2(mtp["wall"]),
                "non-MTP wall": r2(plain["wall"]),
                "Step cost ratio": ratio(mtp["wall"], plain["wall"]),
                "accept length": r2(mtp["accept_length"]),
                "MTP tok/s": r2(mtp["gpu_out_tok_s"]),
                "non-MTP tok/s": r2(plain["gpu_out_tok_s"]),
                "MTP gain (GPU)": gain(mtp["gpu_out_tok_s"], plain["gpu_out_tok_s"]),
                "MTP TPOT": r2(mtp["TPOT_ms"]),
                "non-MTP TPOT": r2(plain["TPOT_ms"]),
                "MTP gain (client)": gain(plain["TPOT_ms"], mtp["TPOT_ms"]),
            }
        )
    return pd.DataFrame(rows)


def build_gpu_gap(records: list) -> pd.DataFrame:
    """MI355 against B200 within one mode, split by variant."""
    by_key = {}
    for rec in records:
        key = (rec["Context"], rec["Mode"], rec["Conc"])
        by_key.setdefault(key, {})[rec["GPU"] + (f"_{rec['Variant']}" if rec["Variant"] else "")] = rec

    rows = []
    for (ctx, mode, conc), sides in sorted(by_key.items()):
        b200 = sides.get("b200")
        if not b200:
            continue
        for name, rec in sorted(sides.items()):
            if name == "b200":
                continue
            rows.append(
                {
                    "Context": ctx,
                    "Mode": mode,
                    "Conc": conc,
                    "MI355 config": name,
                    "MI355 bs": rec["bs"],
                    "B200 bs": b200["bs"],
                    "bs match": "yes" if rec["bs"] == b200["bs"] else "NO",
                    "MI355 wall": r2(rec["wall"]),
                    "B200 wall": r2(b200["wall"]),
                    "Wall ratio": ratio(rec["wall"], b200["wall"]),
                    "MI355 tok/s": r2(rec["gpu_out_tok_s"]),
                    "B200 tok/s": r2(b200["gpu_out_tok_s"]),
                    "Throughput ratio": gain(rec["gpu_out_tok_s"], b200["gpu_out_tok_s"]),
                }
            )
    return pd.DataFrame(rows)


def build_readme(mtp_root: str, nonmtp_root: str) -> pd.DataFrame:
    rows = [
        ("Report", "MTP vs non-MTP decode, MI355 vs B200"),
        ("MTP profiles", mtp_root),
        ("non-MTP profiles", nonmtp_root),
        ("decode wall", "median GPU wall time of one decode step, ms, from the traces"),
        (
            "accept length",
            "accepted tokens per request per step, from the client benchmark summary; "
            "1.0 by definition without MTP",
        ),
        (
            "GPU out tok/s",
            "bs * accept length / decode wall; what the GPU alone would sustain",
        ),
        (
            "client out tok/s",
            "measured end to end, so it also carries prefill, scheduling and detokenisation",
        ),
        (
            "Step cost ratio",
            "MTP wall divided by non-MTP wall; above 1.0 means an MTP step costs more GPU time",
        ),
        ("MTP gain", "output tokens per second with MTP divided by without"),
        ("Units", "ms and tokens per second, 2 decimals"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mtp-root", required=True)
    ap.add_argument("--nonmtp-root", required=True)
    ap.add_argument("-o", "--output", required=True, metavar="XLSX")
    ap.add_argument("--rules", default=GLM52_RULES)
    args = ap.parse_args(argv)

    classifier = KernelClassifier(args.rules)
    print("[MTP]")
    records = measure(args.mtp_root, "mtp", classifier)
    print("[non-MTP]")
    records += measure(args.nonmtp_root, "nomtp", classifier)

    perf = find_perf(args.mtp_root, "mtp")
    perf.update(find_perf(args.nonmtp_root, "nomtp"))
    attach_client(records, perf)
    missing = [r for r in records if r["accept_length"] is None]
    if missing:
        print(f"[warn] no client summary for {len(missing)} runs; tok/s left blank")

    gain = build_mode_gain(records)
    print("\n" + gain.to_string(index=False))

    sheets = [
        ("README", build_readme(args.mtp_root, args.nonmtp_root)),
        ("Mode_Gain", gain),
        ("GPU_Gap", build_gpu_gap(records)),
        ("All_Runs", build_runs_sheet(records)),
    ]
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_excel(out, sheets)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
