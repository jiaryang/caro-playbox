"""MI355 vs B200 per kernel category, one column per configuration.

The per-concurrency sweeps break categories down by phase, which is the right
view inside one run but makes it hard to see how a category behaves as the
configuration changes. This script collapses each run to whole-step category
totals and lays the configurations out side by side, so a row reads as "what
this category costs on each GPU as context and batch grow".

    python category_matrix.py --root 3_0811_MTP --mode MTP \
                              --root 3_0811_nonMTP --mode non-MTP -o categories.xlsx

Within a mode, MI355 is paired against B200 at equal batch: for every
concurrency the MI355 variant whose running batch matches B200 is preferred, and
the pairing is flagged when no variant matches.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(_TOOLS)
sys.path.insert(0, _ANALYSIS)
sys.path.insert(0, _TOOLS)

from decode_profile.kernels import GLM52_RULES, KernelClassifier
from decode_profile.report import MIN_MS, r2, ratio, write_excel
from decode_profile.sweep import build_run, discover

from compare_modes import find_profile_dirs

CONTEXT_ORDER = {"8k": 0, "70k": 1}


def collect(root: str, classifier) -> dict:
    """-> {(context, gpu, variant, conc): run} for every DECODE trace."""
    runs = {}
    for (ctx, gpu, variant), directory in sorted(find_profile_dirs(root).items()):
        try:
            found = discover(directory)
        except ValueError as exc:
            print(f"[skip] {directory}: {exc}")
            continue
        label = gpu + (f"_{variant}" if variant else "")
        for conc in sorted(found):
            run = build_run(label, conc, found[conc], classifier)
            runs[(ctx, gpu, variant, conc)] = run
            print(f"  {ctx:>4} {label:<12} c{conc:<3} bs={run.bs}")
    return runs


def category_totals(run) -> dict:
    """Per-category kernel ms for the whole decode step, summed over phases."""
    totals = {}
    for stats in run.stats.values():
        for cat, ms in stats.per_category_ms.items():
            totals[cat] = totals.get(cat, 0.0) + ms
    return totals


def pair_runs(runs: dict) -> list:
    """Pair each B200 run with the MI355 variant that ran the same batch."""
    pairs = []
    for (ctx, gpu, variant, conc), b200 in sorted(runs.items()):
        if gpu != "b200" or variant:
            continue
        candidates = [
            (v, r) for (c2, g2, v, c3), r in runs.items()
            if g2 == "mi355" and c2 == ctx and c3 == conc
        ]
        if not candidates:
            continue
        matched = [(v, r) for v, r in candidates if r.bs == b200.bs]
        variant_a, mi355 = (matched or sorted(candidates, key=lambda vr: vr[0] or ""))[0]
        pairs.append(
            {
                "context": ctx,
                "conc": conc,
                "variant": variant_a or "default",
                "bs_match": bool(matched),
                "mi355": mi355,
                "b200": b200,
            }
        )
    pairs.sort(key=lambda p: (CONTEXT_ORDER.get(p["context"], 9), p["conc"]))
    return pairs


def column_label(pair) -> str:
    bs = pair["mi355"].bs if pair["bs_match"] else f"{pair['mi355'].bs}/{pair['b200'].bs}"
    return f"{pair['context']} c{pair['conc']} bs{bs}"


def build_index(all_pairs: dict) -> pd.DataFrame:
    rows = []
    for mode, pairs in all_pairs.items():
        for pair in pairs:
            a, b = pair["mi355"], pair["b200"]
            ka = sum(category_totals(a).values())
            kb = sum(category_totals(b).values())
            rows.append(
                {
                    "Mode": mode,
                    "Column": column_label(pair),
                    "Context": pair["context"],
                    "Conc": pair["conc"],
                    "MI355 config": pair["variant"],
                    "MI355 bs": a.bs,
                    "B200 bs": b.bs,
                    "bs match": "yes" if pair["bs_match"] else "NO",
                    "MI355 wall": r2(a.decode_wall),
                    "B200 wall": r2(b.decode_wall),
                    "MI355 kernel": r2(ka),
                    "B200 kernel": r2(kb),
                    "Kernel diff": r2(ka - kb),
                }
            )
    return pd.DataFrame(rows)


def build_matrix(pairs: list, classifier, value: str) -> pd.DataFrame:
    """Rows are categories, columns are configurations.

    ``value`` selects diff (MI355 minus B200), or the absolute ms of one side.
    """
    cats = set()
    totals = []
    for pair in pairs:
        ta, tb = category_totals(pair["mi355"]), category_totals(pair["b200"])
        cats |= set(ta) | set(tb)
        totals.append((pair, ta, tb))

    rows = []
    for cat in sorted(cats, key=classifier.order_index):
        row = {"Category": cat}
        keep = False
        for pair, ta, tb in totals:
            va, vb = ta.get(cat, 0.0), tb.get(cat, 0.0)
            keep = keep or max(va, vb) >= MIN_MS
            row[column_label(pair)] = r2(
                {"diff": va - vb, "mi355": va, "b200": vb}[value]
            )
        if keep:
            rows.append(row)

    total = {"Category": "TOTAL kernel"}
    for pair, ta, tb in totals:
        va, vb = sum(ta.values()), sum(tb.values())
        total[column_label(pair)] = r2({"diff": va - vb, "mi355": va, "b200": vb}[value])
    rows.append(total)
    return pd.DataFrame(rows)


def build_share(pairs: list, classifier) -> pd.DataFrame:
    """Each category as a percentage of that run's kernel time, both GPUs."""
    cats = set()
    totals = []
    for pair in pairs:
        ta, tb = category_totals(pair["mi355"]), category_totals(pair["b200"])
        cats |= set(ta) | set(tb)
        totals.append((pair, ta, tb, sum(ta.values()), sum(tb.values())))

    rows = []
    for cat in sorted(cats, key=classifier.order_index):
        for side, idx in (("MI355", 1), ("B200", 2)):
            row = {"Category": cat, "GPU": side}
            keep = False
            for pair, ta, tb, sa, sb in totals:
                value, total = (ta.get(cat, 0.0), sa) if idx == 1 else (tb.get(cat, 0.0), sb)
                keep = keep or value >= MIN_MS
                row[column_label(pair)] = r2(value / total * 100) if total else None
            if keep:
                rows.append(row)
    return pd.DataFrame(rows)


def build_detail(all_pairs: dict, classifier) -> pd.DataFrame:
    rows = []
    for mode, pairs in all_pairs.items():
        for pair in pairs:
            ta, tb = category_totals(pair["mi355"]), category_totals(pair["b200"])
            sa, sb = sum(ta.values()), sum(tb.values())
            for cat in sorted(set(ta) | set(tb), key=classifier.order_index):
                va, vb = ta.get(cat, 0.0), tb.get(cat, 0.0)
                if max(va, vb) < MIN_MS:
                    continue
                rows.append(
                    {
                        "Mode": mode,
                        "Context": pair["context"],
                        "Conc": pair["conc"],
                        "MI355 config": pair["variant"],
                        "bs": pair["mi355"].bs if pair["bs_match"] else None,
                        "bs match": "yes" if pair["bs_match"] else "NO",
                        "Category": cat,
                        "MI355 ms/step": r2(va),
                        "B200 ms/step": r2(vb),
                        "Diff": r2(va - vb),
                        "Ratio": ratio(va, vb),
                        "MI355 %": r2(va / sa * 100) if sa else None,
                        "B200 %": r2(vb / sb * 100) if sb else None,
                    }
                )
    return pd.DataFrame(rows)


def build_readme(roots: list, classifier) -> pd.DataFrame:
    rows = [
        ("Report", "MI355 vs B200 by kernel category, one column per configuration"),
        ("Scope", "each mode is compared within itself; MTP and non-MTP are never mixed"),
        (
            "Category totals",
            "kernel ms summed over every phase of the step, so an MTP column covers "
            "draft + target_verify + draft_extend",
        ),
        ("Pairing", "MI355 is matched to B200 at equal running batch where a variant allows it"),
        ("Column label", "context, concurrency and the shared batch, e.g. 8k c32 bs32"),
        ("Diff sheets", "MI355 minus B200 in ms/step; positive means MI355 is slower"),
        ("Share sheets", "each category as a percent of that run's own kernel time"),
        ("Per-phase detail", "see Kernel_Category_Scaling in the per-mode concurrency sweeps"),
        ("Kernel categories", classifier.rules_path),
        ("Units", "ms per decode step, 2 decimals"),
    ]
    for mode, root in roots:
        rows.append((f"{mode} profiles", root))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", action="append", required=True,
                    help="profile root; repeat once per mode, paired with --mode")
    ap.add_argument("--mode", action="append", required=True,
                    help="label for the matching --root, e.g. MTP or non-MTP")
    ap.add_argument("-o", "--output", required=True, metavar="XLSX")
    ap.add_argument("--rules", default=GLM52_RULES)
    args = ap.parse_args(argv)

    if len(args.root) != len(args.mode):
        raise SystemExit("--root and --mode must be given the same number of times")

    classifier = KernelClassifier(args.rules)
    all_pairs = {}
    for mode, root in zip(args.mode, args.root):
        print(f"[{mode}]")
        all_pairs[mode] = pair_runs(collect(root, classifier))

    sheets = [
        ("README", build_readme(list(zip(args.mode, args.root)), classifier)),
        ("Config_Index", build_index(all_pairs)),
    ]
    for mode, pairs in all_pairs.items():
        tag = mode.replace("-", "").replace(" ", "")
        sheets.append((f"{tag}_Diff", build_matrix(pairs, classifier, "diff")))
        sheets.append((f"{tag}_MI355", build_matrix(pairs, classifier, "mi355")))
        sheets.append((f"{tag}_B200", build_matrix(pairs, classifier, "b200")))
        sheets.append((f"{tag}_Share", build_share(pairs, classifier)))
    sheets.append(("Detail", build_detail(all_pairs, classifier)))

    for mode, pairs in all_pairs.items():
        print(f"\n=== {mode}: MI355 minus B200, ms/step ===")
        print(build_matrix(pairs, classifier, "diff").to_string(index=False))

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_excel(out, sheets)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
