"""Compare suite perf summaries across machines (nonMTP / MTP).

Reads ``perf/{nomtp|mtp}/summary.txt`` from each suite root and writes an Excel
workbook with side-by-side Token TPUT/GPU + Median TPOT ratios.

Example (4_0818 nonMTP: two MI355 + one B200)::

    python tools/compare_suite_perf.py \\
      --suite path/to/suite_..._m11-13=m11-13 \\
      --suite path/to/suite_..._n10-17=n10-17 \\
      --suite path/to/suite_..._dgx-024=dgx-024 \\
      --baseline dgx-024 --mode nomtp \\
      -o compare_nomtp.xlsx
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def parse_summary(path: Path) -> tuple[dict, list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta: dict = {}
    for key, pat in [
        ("model", r"^Model:\s*(.+)"),
        ("node", r"^Node:\s+(\S+)"),
        ("gpu", r"GPU:\s*(\S+)"),
        ("tp", r"Num GPUs \(TP, default\):\s*(\d+)"),
    ]:
        m = re.search(pat, text, re.M)
        if m:
            meta[key] = m.group(1).strip()
    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "input": int(m.group(1)),
                "conc": int(m.group(2)),
                "act_conc": float(m.group(3)),
                "interact": float(m.group(4)),
                "tot_gpu": float(m.group(5)),
                "TTFT": float(m.group(6)),
                "TPOT": float(m.group(7)),
                "ITL": float(m.group(8)),
            }
        )
    return meta, rows


def geomean(vals) -> float:
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        action="append",
        required=True,
        help="suite_root=label  (repeatable; '=' avoids Windows drive 'C:')",
    )
    ap.add_argument("--baseline", required=True, help="label used as denominator")
    ap.add_argument("--mode", default="nomtp", choices=["nomtp", "mtp"])
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    suites: dict[str, Path] = {}
    for item in args.suite:
        if "=" in item:
            root, label = item.rsplit("=", 1)
        elif ":" in item and not re.match(r"^[A-Za-z]:[\\/]", item):
            root, _, label = item.partition(":")
        else:
            raise SystemExit(f"--suite needs suite_root=label, got {item!r}")
        if not label:
            raise SystemExit(f"--suite needs suite_root=label, got {item!r}")
        suites[label] = Path(root)

    if args.baseline not in suites:
        raise SystemExit(f"--baseline {args.baseline!r} not in suite labels")

    meta, perf = {}, {}
    for label, root in suites.items():
        path = root / "perf" / args.mode / "summary.txt"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        m, rows = parse_summary(path)
        meta[label] = m
        perf[label] = {(r["input"], r["conc"]): r for r in rows}
        print(f"{label}: {m.get('node')} / {m.get('gpu')}  n={len(rows)}")

    base = perf[args.baseline]
    others = [l for l in suites if l != args.baseline]
    keys = sorted(base.keys())

    cmp_rows = []
    for inp, conc in keys:
        b = base[(inp, conc)]
        row = {
            "input": inp,
            "conc": conc,
            f"{args.baseline}_tot": b["tot_gpu"],
            f"{args.baseline}_tpot": b["TPOT"],
        }
        for lab in others:
            r = perf[lab][(inp, conc)]
            row[f"{lab}_tot"] = r["tot_gpu"]
            row[f"{lab}_tpot"] = r["TPOT"]
            row[f"{lab}_tot_vs_{args.baseline}"] = round(r["tot_gpu"] / b["tot_gpu"], 4)
            row[f"{lab}_tpot_vs_{args.baseline}"] = round(r["TPOT"] / b["TPOT"], 4)
        if len(others) == 2:
            a, c = others
            row[f"{a}_vs_{c}_tot"] = round(
                perf[a][(inp, conc)]["tot_gpu"] / perf[c][(inp, conc)]["tot_gpu"], 4
            )
            row[f"{a}_vs_{c}_tpot"] = round(
                perf[a][(inp, conc)]["TPOT"] / perf[c][(inp, conc)]["TPOT"], 4
            )
        cmp_rows.append(row)

    df = pd.DataFrame(cmp_rows)
    geo_rows = []
    for isl, label in [(1024, "1k"), (8192, "8k"), (70000, "70k")]:
        sub = df[df.input == isl]
        if sub.empty:
            continue
        g = {"isl": label}
        for lab in others:
            g[f"{lab}_tot_geomean"] = round(
                geomean(sub[f"{lab}_tot_vs_{args.baseline}"]), 4
            )
            g[f"{lab}_tpot_geomean"] = round(
                geomean(sub[f"{lab}_tpot_vs_{args.baseline}"]), 4
            )
        geo_rows.append(g)

    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(
            [
                {
                    "label": lab,
                    "role": "baseline" if lab == args.baseline else "compare",
                    "model": meta[lab].get("model"),
                    "node": meta[lab].get("node"),
                    "gpu": meta[lab].get("gpu"),
                    "tp": meta[lab].get("tp"),
                    "suite": suites[lab].name,
                    "mode": args.mode,
                }
                for lab in suites
            ]
        ).to_excel(w, sheet_name="meta", index=False)
        df.to_excel(w, sheet_name="tput_tpot", index=False)
        pd.DataFrame(geo_rows).to_excel(w, sheet_name="geomean", index=False)

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
