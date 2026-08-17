"""CLI: compare two SGLang MTP decode profiles and write an Excel report."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from . import report
from .kernels import BASE_RULES, GLM52_RULES, KernelClassifier
from .phases import analyze_phases, decode_totals, get_phase_stats, phase_order
from .structure import op_calls_per_step
from .trace import Trace, load_trace


@dataclass
class Side:
    label: str
    trace: Trace
    stats: dict


def build_side(label: str, path: str, classifier) -> Side:
    print(f"[load] {label}: {os.path.basename(path)}")
    tr = load_trace(path, label)
    stats = analyze_phases(tr, classifier)
    n = max(st.n_steps for st in stats.values())
    print(
        f"       events={tr.event_total:,} kernels={len(tr.k_names):,} "
        f"phases={','.join(tr.phases)} streams={tr.phase_tids} steps={n}"
    )
    return Side(label=label, trace=tr, stats=stats)


def print_summary(side_a: Side, side_b: Side) -> None:
    print()
    header = f"{'phase':<15}{side_a.label + ' wall':>14}{side_b.label + ' wall':>14}{'diff':>9}{side_a.label + ' kern':>14}{side_b.label + ' kern':>14}{'diff':>9}"
    print(header)
    print("-" * len(header))
    for phase in phase_order(side_a.stats, side_b.stats):
        sa, sb = get_phase_stats(side_a.stats, phase), get_phase_stats(side_b.stats, phase)
        print(
            f"{phase:<15}{sa.wall_ms:>14.2f}{sb.wall_ms:>14.2f}{sa.wall_ms - sb.wall_ms:>9.2f}"
            f"{sa.kernel_ms:>14.2f}{sb.kernel_ms:>14.2f}{sa.kernel_ms - sb.kernel_ms:>9.2f}"
        )
    wa, ka = decode_totals(side_a.stats)
    wb, kb = decode_totals(side_b.stats)
    print("-" * len(header))
    print(
        f"{'decode total':<15}{wa:>14.2f}{wb:>14.2f}{wa - wb:>9.2f}{ka:>14.2f}{kb:>14.2f}{ka - kb:>9.2f}"
    )
    print(f"\nper-step decode wall ratio {side_a.label}/{side_b.label}: {wa / wb:.2f}x" if wb else "")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", nargs=2, metavar=("LABEL", "TRACE"), required=True,
                    help="first side: label and CUDA-graph-ON trace")
    ap.add_argument("--b", nargs=2, metavar=("LABEL", "TRACE"), required=True,
                    help="second side: label and CUDA-graph-ON trace")
    ap.add_argument("--graph-off-a", metavar="TRACE",
                    help="graph-OFF trace of side A, for the op-call cross-check")
    ap.add_argument("--graph-off-b", metavar="TRACE",
                    help="graph-OFF trace of side B, for the op-call cross-check")
    ap.add_argument("-o", "--output", required=True, metavar="XLSX")
    ap.add_argument("--rules", default=GLM52_RULES,
                    help=f"kernel category CSV (default: GLM-5.2 buckets; base set: {BASE_RULES})")
    args = ap.parse_args(argv)

    classifier = KernelClassifier(args.rules)
    side_a = build_side(args.a[0], args.a[1], classifier)
    side_b = build_side(args.b[0], args.b[1], classifier)
    print_summary(side_a, side_b)

    graph_off = []
    ops_a = ops_b = {}
    if args.graph_off_a and args.graph_off_b:
        print("\n[load] graph-OFF traces for op-call cross-check")
        tr_a = load_trace(args.graph_off_a, side_a.label, want_cpu_ops=True)
        tr_b = load_trace(args.graph_off_b, side_b.label, want_cpu_ops=True)
        ops_a = op_calls_per_step(tr_a)
        ops_b = op_calls_per_step(tr_b)
        graph_off = [(side_a.label, tr_a), (side_b.label, tr_b)]
        for label, tr in graph_off:
            print(f"       {label}: cpu_ops={len(tr.cpu_ops):,} cpu phase spans={len(tr.cpu_spans)}")

    sheets = [
        ("README", report.build_readme(side_a, side_b, classifier, graph_off)),
        ("PerStep_Summary", report.build_per_step_summary(side_a, side_b)),
        ("Phase_Diff", report.build_phase_diff(side_a, side_b)),
        ("Kernel_Category_Diff", report.build_category_diff(side_a, side_b, classifier)),
        ("Top_Kernels", report.build_top_kernels(side_a, side_b, classifier)),
        ("Streams", report.build_streams(side_a, side_b)),
    ]
    if graph_off:
        sheets.append(
            ("GraphOFF_Ops", report.build_graph_off_ops(ops_a, ops_b, side_a.label, side_b.label))
        )

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report.write_excel(out, sheets)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
