"""Concurrency sweep: run the phase analysis over c4..c64 and diff the two GPUs.

Discovers ``*_c<N>-*DECODE.trace.json.gz`` in two profile directories, pairs the
runs by concurrency, and writes one workbook where concurrency is a dimension.
Traces are loaded one concurrency at a time and dropped afterwards so the whole
sweep stays in a few hundred MB.

Handles both MTP runs (draft / target_verify / draft_extend) and plain runs
(a single ``decode`` phase); the phase set comes from the traces themselves.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import sys
from dataclasses import dataclass, field

import pandas as pd

from .kernels import BASE_RULES, GLM52_RULES, KernelClassifier
from .phases import analyze_phases, decode_totals, phase_order
from .report import MIN_MS, r2, ratio, write_excel
from .trace import load_trace

CONC_RE = re.compile(r"_c(\d+)[-_.]")
TOP_KERNELS = 8
DEFAULT_PATTERN = "*DECODE.trace.json.gz"
SHEET_PHASE_NAMES = {"target_verify": "TargetVerify", "decode": "Decode"}


@dataclass
class Run:
    """Everything the report needs from one trace, without keeping the trace."""

    label: str
    conc: int
    stats: dict
    bs: int = None
    basename: str = ""
    phase_tids: tuple = ()
    kernel_tid_counts: dict = field(default_factory=dict)
    n_events: int = 0
    n_kernels: int = 0

    @property
    def steps(self) -> int:
        return max((st.n_steps for st in self.stats.values()), default=0)

    @property
    def decode_wall(self) -> float:
        return decode_totals(self.stats)[0]

    @property
    def decode_kernel(self) -> float:
        return decode_totals(self.stats)[1]


def discover(directory: str, pattern: str = DEFAULT_PATTERN) -> dict:
    """-> {concurrency: trace path}."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        m = CONC_RE.search(os.path.basename(path))
        if not m:
            continue
        conc = int(m.group(1))
        if conc in found:
            raise ValueError(f"two traces for c{conc} in {directory}")
        found[conc] = path
    if not found:
        raise ValueError(f"no traces matching {pattern} in {directory}")
    return found


def build_run(label: str, conc: int, path: str, classifier) -> Run:
    tr = load_trace(path, label)
    stats = analyze_phases(tr, classifier)
    run = Run(
        label=label,
        conc=conc,
        stats=stats,
        bs=tr.batch_size,
        basename=tr.basename,
        phase_tids=tuple(tr.phase_tids),
        kernel_tid_counts=dict(tr.kernel_tid_counts),
        n_events=tr.event_total,
        n_kernels=len(tr.k_names),
    )
    del tr
    gc.collect()
    return run


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------


def build_readme(pairs, label_a, label_b, classifier, dir_a, dir_b, phases) -> pd.DataFrame:
    concs = ", ".join(f"c{c}" for c, _, _ in pairs)
    mismatched = [f"c{c}" for c, a, b in pairs if a.bs != b.bs]
    capped = [
        f"c{c}: {label_a} bs={a.bs}, {label_b} bs={b.bs}"
        for c, a, b in pairs
        if a.bs < c or b.bs < c
    ]
    rows = [
        ("Report", f"GLM DECODE concurrency sweep, {label_a} vs {label_b}"),
        ("Phases", ", ".join(phases)),
        ("Concurrencies", concs),
        (f"{label_a} bs per concurrency", ", ".join(f"c{c}: {a.bs}" for c, a, _ in pairs)),
        (f"{label_b} bs per concurrency", ", ".join(f"c{c}: {b.bs}" for c, _, b in pairs)),
        (
            "Batch size match",
            "identical on both sides at every concurrency"
            if not mismatched
            else "MISMATCH at " + ", ".join(mismatched)
            + " - per-step wall is not comparable there, use req-steps/s instead",
        ),
        (
            "Batch below requested concurrency",
            "; ".join(capped) if capped else "none",
        ),
        ("req-steps/s", "bs divided by decode wall; the throughput view when bs differs"),
        (f"{label_a} directory", dir_a),
        (f"{label_b} directory", dir_b),
        ("Timing source", "CUDA-graph-ON traces"),
        (
            "Phase markers",
            "cat=gpu_user_annotation: draft, step[TARGET_VERIFY], draft_extend for MTP; "
            "step[DECODE bs=N] for a plain run",
        ),
        ("Multi-stream handling", "phase windows unioned across streams; kernel time summed over all streams"),
        ("overlap factor", "kernel ms/step divided by wall ms/step; above 1.0 means streams run concurrently"),
        ("non-kernel gap", "wall ms/step minus kernel ms/step; GPU idle inside the phase"),
        ("wall per request", "decode wall ms/step divided by bs; lower is better at equal bs"),
        ("Kernel categories", classifier.rules_path),
        ("Units", "ms per decode step, median wall, 2 decimals"),
    ]
    for conc, a, b in pairs:
        rows.append((f"c{conc} {label_a} trace", a.basename))
        rows.append((f"c{conc} {label_b} trace", b.basename))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def build_bs_check(pairs, label_a, label_b) -> pd.DataFrame:
    rows = []
    for conc, a, b in pairs:
        rows.append(
            {
                "Concurrency": conc,
                f"{label_a} bs": a.bs,
                f"{label_b} bs": b.bs,
                "bs match": "yes" if a.bs == b.bs else "NO",
                f"{label_a} steps": a.steps,
                f"{label_b} steps": b.steps,
                f"{label_a} kernels in trace": a.n_kernels,
                f"{label_b} kernels in trace": b.n_kernels,
                f"{label_a} phase streams": ", ".join(str(t) for t in a.phase_tids),
                f"{label_b} phase streams": ", ".join(str(t) for t in b.phase_tids),
            }
        )
    return pd.DataFrame(rows)


def build_scaling(pairs, label_a, label_b) -> pd.DataFrame:
    """Decode totals per concurrency: the headline scaling view."""
    rows = []
    for conc, a, b in pairs:
        wa, ka = decode_totals(a.stats)
        wb, kb = decode_totals(b.stats)
        tp_a = a.bs / wa * 1000.0 if a.bs and wa else None
        tp_b = b.bs / wb * 1000.0 if b.bs and wb else None
        rows.append(
            {
                "Concurrency": conc,
                f"{label_a} bs": a.bs,
                f"{label_b} bs": b.bs,
                "bs match": "yes" if a.bs == b.bs else "NO",
                f"{label_a} wall": r2(wa),
                f"{label_b} wall": r2(wb),
                "Wall diff": r2(wa - wb),
                "Wall ratio": ratio(wa, wb),
                f"{label_a} kernel": r2(ka),
                f"{label_b} kernel": r2(kb),
                "Kernel diff": r2(ka - kb),
                "Kernel ratio": ratio(ka, kb),
                f"{label_a} overlap": ratio(ka, wa),
                f"{label_b} overlap": ratio(kb, wb),
                f"{label_a} gap": r2(wa - ka),
                f"{label_b} gap": r2(wb - kb),
                f"{label_a} wall/req": r2(wa / a.bs) if a.bs else None,
                f"{label_b} wall/req": r2(wb / b.bs) if b.bs else None,
                f"{label_a} req-steps/s": r2(tp_a),
                f"{label_b} req-steps/s": r2(tp_b),
                "Throughput ratio": ratio(tp_a, tp_b) if tp_b else None,
            }
        )
    df = pd.DataFrame(rows)
    # Growth relative to the smallest concurrency, so the shape of the curve is
    # visible without leaving the sheet.
    base = df.iloc[0]
    for label in (label_a, label_b):
        col = f"{label} wall"
        df[f"{label} wall vs c{int(base['Concurrency'])}"] = [
            ratio(v, base[col]) for v in df[col]
        ]
    return df


def build_phase_scaling(pairs, label_a, label_b, phases) -> pd.DataFrame:
    rows = []
    for conc, a, b in pairs:
        for phase in phases:
            sa, sb = a.stats[phase], b.stats[phase]
            rows.append(
                {
                    "Concurrency": conc,
                    f"{label_a} bs": a.bs,
                    f"{label_b} bs": b.bs,
                    "Phase": phase,
                    f"{label_a} wall": r2(sa.wall_ms),
                    f"{label_b} wall": r2(sb.wall_ms),
                    "Wall diff": r2(sa.wall_ms - sb.wall_ms),
                    "Wall ratio": ratio(sa.wall_ms, sb.wall_ms),
                    f"{label_a} kernel": r2(sa.kernel_ms),
                    f"{label_b} kernel": r2(sb.kernel_ms),
                    "Kernel diff": r2(sa.kernel_ms - sb.kernel_ms),
                    "Kernel ratio": ratio(sa.kernel_ms, sb.kernel_ms),
                    f"{label_a} overlap": r2(sa.overlap_factor),
                    f"{label_b} overlap": r2(sb.overlap_factor),
                    f"{label_a} gap": r2(sa.non_kernel_gap_ms),
                    f"{label_b} gap": r2(sb.non_kernel_gap_ms),
                }
            )
        wa, ka = decode_totals(a.stats)
        wb, kb = decode_totals(b.stats)
        rows.append(
            {
                "Concurrency": conc,
                f"{label_a} bs": a.bs,
                f"{label_b} bs": b.bs,
                "Phase": "decode total",
                f"{label_a} wall": r2(wa),
                f"{label_b} wall": r2(wb),
                "Wall diff": r2(wa - wb),
                "Wall ratio": ratio(wa, wb),
                f"{label_a} kernel": r2(ka),
                f"{label_b} kernel": r2(kb),
                "Kernel diff": r2(ka - kb),
                "Kernel ratio": ratio(ka, kb),
                f"{label_a} overlap": ratio(ka, wa),
                f"{label_b} overlap": ratio(kb, wb),
                f"{label_a} gap": r2(wa - ka),
                f"{label_b} gap": r2(wb - kb),
            }
        )
    return pd.DataFrame(rows)


def build_phase_wall_pivot(pairs, label_a, label_b, phases) -> pd.DataFrame:
    """Phase wall time with concurrency across the columns."""
    rows = []
    # With a single phase the total would just repeat that phase's row.
    labels = list(phases) + (["decode total"] if len(phases) > 1 else [])
    for phase in labels:
        for label, idx in ((label_a, 1), (label_b, 2)):
            row = {"Phase": phase, "GPU": label}
            for conc, a, b in pairs:
                run = (a, b)[idx - 1]
                value = run.decode_wall if phase == "decode total" else run.stats[phase].wall_ms
                row[f"c{conc}"] = r2(value)
            rows.append(row)
        row = {"Phase": phase, "GPU": f"diff ({label_a}-{label_b})"}
        for conc, a, b in pairs:
            if phase == "decode total":
                va, vb = a.decode_wall, b.decode_wall
            else:
                va, vb = a.stats[phase].wall_ms, b.stats[phase].wall_ms
            row[f"c{conc}"] = r2(va - vb)
        rows.append(row)
    return pd.DataFrame(rows)


def build_category_scaling(pairs, label_a, label_b, classifier, phases) -> pd.DataFrame:
    rows = []
    for conc, a, b in pairs:
        for phase in phases:
            sa, sb = a.stats[phase], b.stats[phase]
            rows.append(
                {
                    "Concurrency": conc,
                    f"{label_a} bs": a.bs,
                    f"{label_b} bs": b.bs,
                    "Phase": phase,
                    "Category": "TOTAL kernel",
                    f"{label_a} ms/step": r2(sa.kernel_ms),
                    f"{label_b} ms/step": r2(sb.kernel_ms),
                    "Diff": r2(sa.kernel_ms - sb.kernel_ms),
                    "Ratio": ratio(sa.kernel_ms, sb.kernel_ms),
                    f"{label_a} %": r2(100.0) if sa.kernel_ms else None,
                    f"{label_b} %": r2(100.0) if sb.kernel_ms else None,
                }
            )
            categories = set(sa.per_category_ms) | set(sb.per_category_ms)
            for cat in sorted(categories, key=classifier.order_index):
                va = sa.per_category_ms.get(cat, 0.0)
                vb = sb.per_category_ms.get(cat, 0.0)
                if max(va, vb) < MIN_MS:
                    continue
                rows.append(
                    {
                        "Concurrency": conc,
                        f"{label_a} bs": a.bs,
                        f"{label_b} bs": b.bs,
                        "Phase": phase,
                        "Category": cat,
                        f"{label_a} ms/step": r2(va),
                        f"{label_b} ms/step": r2(vb),
                        "Diff": r2(va - vb),
                        "Ratio": ratio(va, vb),
                        f"{label_a} %": r2(va / sa.kernel_ms * 100) if sa.kernel_ms else None,
                        f"{label_b} %": r2(vb / sb.kernel_ms * 100) if sb.kernel_ms else None,
                    }
                )
    return pd.DataFrame(rows)


def build_category_pivot(pairs, label_a, label_b, classifier, phase) -> pd.DataFrame:
    """One category per row, concurrency across the columns, for a single phase."""
    categories = set()
    for _, a, b in pairs:
        categories |= set(a.stats[phase].per_category_ms) | set(b.stats[phase].per_category_ms)
    rows = []
    for cat in sorted(categories, key=classifier.order_index):
        row_a = {"Phase": phase, "Category": cat, "Metric": f"{label_a} ms/step"}
        row_b = {"Phase": phase, "Category": cat, "Metric": f"{label_b} ms/step"}
        row_d = {"Phase": phase, "Category": cat, "Metric": f"diff ({label_a}-{label_b})"}
        keep = False
        for conc, a, b in pairs:
            va = a.stats[phase].per_category_ms.get(cat, 0.0)
            vb = b.stats[phase].per_category_ms.get(cat, 0.0)
            keep = keep or max(va, vb) >= MIN_MS
            row_a[f"c{conc}"] = r2(va)
            row_b[f"c{conc}"] = r2(vb)
            row_d[f"c{conc}"] = r2(va - vb)
        if keep:
            rows.extend([row_a, row_b, row_d])
    return pd.DataFrame(rows)


def build_top_kernels(pairs, label_a, label_b, classifier, phases) -> pd.DataFrame:
    rows = []
    for conc, a, b in pairs:
        for phase in phases:
            for run in (a, b):
                st = run.stats[phase]
                ranked = sorted(st.per_kernel_ms.items(), key=lambda kv: -kv[1])[:TOP_KERNELS]
                for rank, (name, ms) in enumerate(ranked, 1):
                    if ms < MIN_MS:
                        continue
                    rows.append(
                        {
                            "Concurrency": conc,
                            "bs": run.bs,
                            "Phase": phase,
                            "GPU": run.label,
                            "Rank": rank,
                            "Kernel": name,
                            "Category": classifier.classify(name),
                            "ms/step": r2(ms),
                            "calls/step": r2(st.per_kernel_calls.get(name, 0)),
                            "% of phase kernel": r2(ms / st.kernel_ms * 100) if st.kernel_ms else None,
                        }
                    )
    return pd.DataFrame(rows)


def build_streams(pairs, label_a, label_b, phases) -> pd.DataFrame:
    rows = []
    for conc, a, b in pairs:
        for run in (a, b):
            phase_tids = set(run.phase_tids)
            totals = {}
            for phase in phases:
                for tid, ms in run.stats[phase].per_stream_ms.items():
                    totals[tid] = totals.get(tid, 0.0) + ms
            grand = sum(totals.values())
            for tid in sorted(totals, key=lambda t: -totals[t]):
                rows.append(
                    {
                        "Concurrency": conc,
                        "GPU": run.label,
                        "Stream tid": tid,
                        "Carries phase markers": "yes" if tid in phase_tids else "no",
                        "Kernels in trace": run.kernel_tid_counts.get(tid, 0),
                        "Kernel ms/step (decode)": r2(totals[tid]),
                        "Share %": r2(totals[tid] / grand * 100) if grand else None,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def print_scaling(pairs, label_a, label_b) -> None:
    header = (
        f"{'conc':>5}{'bs ' + label_a:>12}{'bs ' + label_b:>12}"
        f"{label_a + ' wall':>14}{label_b + ' wall':>14}{'ratio':>8}"
        f"{label_a + ' rps':>13}{label_b + ' rps':>13}{'ratio':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for conc, a, b in pairs:
        wa, _ = decode_totals(a.stats)
        wb, _ = decode_totals(b.stats)
        ta, tb = a.bs / wa * 1000.0, b.bs / wb * 1000.0
        flag = "" if a.bs == b.bs else "  <- bs differs"
        print(
            f"{conc:>5}{a.bs:>12}{b.bs:>12}{wa:>14.2f}{wb:>14.2f}{wa / wb:>8.2f}"
            f"{ta:>13.1f}{tb:>13.1f}{ta / tb:>8.2f}{flag}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", nargs=2, metavar=("LABEL", "DIR"), required=True,
                    help="first side: label and profile directory")
    ap.add_argument("--b", nargs=2, metavar=("LABEL", "DIR"), required=True,
                    help="second side: label and profile directory")
    ap.add_argument("-o", "--output", required=True, metavar="XLSX")
    ap.add_argument("--conc", nargs="*", type=int,
                    help="concurrencies to include (default: all present on both sides)")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--rules", default=GLM52_RULES,
                    help=f"kernel category CSV (default: GLM-5.2 buckets; base set: {BASE_RULES})")
    args = ap.parse_args(argv)

    label_a, dir_a = args.a
    label_b, dir_b = args.b
    found_a = discover(dir_a, args.pattern)
    found_b = discover(dir_b, args.pattern)
    concs = sorted(set(found_a) & set(found_b))
    if args.conc:
        concs = [c for c in concs if c in set(args.conc)]
    if not concs:
        raise SystemExit("no concurrency present in both directories")
    only_a = sorted(set(found_a) - set(found_b))
    only_b = sorted(set(found_b) - set(found_a))
    if only_a or only_b:
        print(f"[skip] unpaired: {label_a} {only_a} {label_b} {only_b}")

    classifier = KernelClassifier(args.rules)
    pairs = []
    for conc in concs:
        print(f"[c{conc}] loading")
        a = build_run(label_a, conc, found_a[conc], classifier)
        b = build_run(label_b, conc, found_b[conc], classifier)
        print(
            f"       {label_a} bs={a.bs} steps={a.steps} streams={a.phase_tids} | "
            f"{label_b} bs={b.bs} steps={b.steps} streams={b.phase_tids}"
        )
        if a.bs != b.bs:
            print(f"       WARNING: batch size differs at c{conc}")
        pairs.append((conc, a, b))

    print_scaling(pairs, label_a, label_b)

    phases = phase_order(*[r.stats for _, a, b in pairs for r in (a, b)])
    # The phase worth pivoting on: the one holding most of the step.
    main_phase = max(
        phases, key=lambda p: sum(r.stats[p].kernel_ms for _, a, b in pairs for r in (a, b))
    )
    pivot_sheet = SHEET_PHASE_NAMES.get(main_phase, main_phase.title()) + "_Categories"
    print(f"\n[phases] {', '.join(phases)} (pivot on {main_phase})")

    sheets = [
        ("README", build_readme(pairs, label_a, label_b, classifier, dir_a, dir_b, phases)),
        ("BatchSize_Check", build_bs_check(pairs, label_a, label_b)),
        ("Decode_Scaling", build_scaling(pairs, label_a, label_b)),
        ("Phase_Wall_ByConc", build_phase_wall_pivot(pairs, label_a, label_b, phases)),
        ("Phase_Scaling", build_phase_scaling(pairs, label_a, label_b, phases)),
        (pivot_sheet, build_category_pivot(pairs, label_a, label_b, classifier, main_phase)),
        (
            "Kernel_Category_Scaling",
            build_category_scaling(pairs, label_a, label_b, classifier, phases),
        ),
        ("Top_Kernels", build_top_kernels(pairs, label_a, label_b, classifier, phases)),
        ("Streams", build_streams(pairs, label_a, label_b, phases)),
    ]

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_excel(out, sheets)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
