"""Excel report assembly. Every number is rounded to two decimals."""

from __future__ import annotations

import pandas as pd

from .phases import decode_totals, get_phase_stats, phase_order

TOP_KERNELS = 12
TOP_OPS = 30
MIN_MS = 0.005


def r2(x):
    return None if x is None else round(float(x), 2)


def ratio(a, b):
    return r2(a / b) if b else None


def _phase_label(phase: str) -> str:
    return phase


def _phases(side_a, side_b) -> list:
    return phase_order(side_a.stats, side_b.stats)


def build_readme(side_a, side_b, classifier, graph_off=None) -> pd.DataFrame:
    rows = [
        ("Report", "GLM MTP 8k conc=4 DECODE, MI355 vs B200"),
        ("Timing source", "CUDA-graph-ON traces"),
        (f"{side_a.label} trace", side_a.trace.basename),
        (f"{side_b.label} trace", side_b.trace.basename),
        ("Phase markers", "cat=gpu_user_annotation: draft, step[TARGET_VERIFY], draft_extend"),
        ("Inner markers ignored", "step[DECODE bs=N], step[DRAFT_EXTEND_V2 bs=N] nest inside the above"),
        ("Multi-stream handling", "phase windows unioned across streams; kernel time summed over all streams"),
        ("overlap factor", "kernel ms/step divided by wall ms/step; above 1.0 means streams run concurrently"),
        ("non-kernel gap", "wall ms/step minus kernel ms/step; GPU idle inside the phase"),
        ("Kernel categories", classifier.rules_path),
        ("Units", "ms per decode step, median wall, 2 decimals"),
    ]
    for side in (side_a, side_b):
        rows.append(
            (f"{side.label} phase streams", ", ".join(str(t) for t in side.trace.phase_tids))
        )
        rows.append(
            (
                f"{side.label} kernel streams",
                ", ".join(
                    f"{t}:{c}"
                    for t, c in sorted(
                        side.trace.kernel_tid_counts.items(), key=lambda kv: -kv[1]
                    )
                ),
            )
        )
    if graph_off:
        rows.append(("Structure cross-check", "CUDA-graph-OFF traces, op calls only"))
        for label, tr in graph_off:
            rows.append((f"{label} graph-OFF trace", tr.basename))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def build_per_step_summary(side_a, side_b) -> pd.DataFrame:
    phases = _phases(side_a, side_b)

    def raw(side):
        wall, kernel = decode_totals(side.stats)
        data = {}
        for phase in phases:
            st = get_phase_stats(side.stats, phase)
            data[f"{phase} wall"] = st.wall_ms
            data[f"{phase} kernel"] = st.kernel_ms
        data["decode wall/step"] = wall
        data["decode kernel/step"] = kernel
        return data

    # Diff and ratio come from the raw values, not the rounded ones, so this
    # sheet agrees with Phase_Diff to the last displayed digit.
    ra, rb = raw(side_a), raw(side_b)
    metrics = list(ra)
    steps_a = max(st.n_steps for st in side_a.stats.values())
    steps_b = max(st.n_steps for st in side_b.stats.values())
    rows = [
        {"GPU": side_a.label, "Steps": steps_a,
         **{k: r2(ra[k]) for k in metrics}},
        {"GPU": side_b.label, "Steps": steps_b,
         **{k: r2(rb[k]) for k in metrics}},
        {"GPU": f"Diff ({side_a.label}-{side_b.label})", "Steps": "",
         **{k: r2(ra[k] - rb[k]) for k in metrics}},
        {"GPU": f"Ratio ({side_a.label}/{side_b.label})", "Steps": "",
         **{k: ratio(ra[k], rb[k]) for k in metrics}},
    ]
    return pd.DataFrame(rows)


def build_phase_diff(side_a, side_b) -> pd.DataFrame:
    rows = []
    for phase in _phases(side_a, side_b):
        sa, sb = get_phase_stats(side_a.stats, phase), get_phase_stats(side_b.stats, phase)
        rows.append(
            {
                "Phase": _phase_label(phase),
                f"{side_a.label} steps": sa.n_steps,
                f"{side_b.label} steps": sb.n_steps,
                f"{side_a.label} wall": r2(sa.wall_ms),
                f"{side_b.label} wall": r2(sb.wall_ms),
                "Wall diff": r2(sa.wall_ms - sb.wall_ms),
                "Wall ratio": ratio(sa.wall_ms, sb.wall_ms),
                f"{side_a.label} kernel": r2(sa.kernel_ms),
                f"{side_b.label} kernel": r2(sb.kernel_ms),
                "Kernel diff": r2(sa.kernel_ms - sb.kernel_ms),
                "Kernel ratio": ratio(sa.kernel_ms, sb.kernel_ms),
                f"{side_a.label} overlap": r2(sa.overlap_factor),
                f"{side_b.label} overlap": r2(sb.overlap_factor),
                f"{side_a.label} gap": r2(sa.non_kernel_gap_ms),
                f"{side_b.label} gap": r2(sb.non_kernel_gap_ms),
            }
        )
    wall_a, kern_a = decode_totals(side_a.stats)
    wall_b, kern_b = decode_totals(side_b.stats)
    rows.append(
        {
            "Phase": "decode total",
            f"{side_a.label} steps": "",
            f"{side_b.label} steps": "",
            f"{side_a.label} wall": r2(wall_a),
            f"{side_b.label} wall": r2(wall_b),
            "Wall diff": r2(wall_a - wall_b),
            "Wall ratio": ratio(wall_a, wall_b),
            f"{side_a.label} kernel": r2(kern_a),
            f"{side_b.label} kernel": r2(kern_b),
            "Kernel diff": r2(kern_a - kern_b),
            "Kernel ratio": ratio(kern_a, kern_b),
            f"{side_a.label} overlap": ratio(kern_a, wall_a),
            f"{side_b.label} overlap": ratio(kern_b, wall_b),
            f"{side_a.label} gap": r2(wall_a - kern_a),
            f"{side_b.label} gap": r2(wall_b - kern_b),
        }
    )
    return pd.DataFrame(rows)


def build_category_diff(side_a, side_b, classifier) -> pd.DataFrame:
    rows = []
    for phase in _phases(side_a, side_b):
        sa, sb = get_phase_stats(side_a.stats, phase), get_phase_stats(side_b.stats, phase)
        rows.append(
            {
                "Phase": _phase_label(phase),
                "Category": "TOTAL kernel",
                f"{side_a.label} ms/step": r2(sa.kernel_ms),
                f"{side_b.label} ms/step": r2(sb.kernel_ms),
                "Diff": r2(sa.kernel_ms - sb.kernel_ms),
                "Ratio": ratio(sa.kernel_ms, sb.kernel_ms),
                f"{side_a.label} %": r2(100.0) if sa.kernel_ms else None,
                f"{side_b.label} %": r2(100.0) if sb.kernel_ms else None,
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
                    "Phase": _phase_label(phase),
                    "Category": cat,
                    f"{side_a.label} ms/step": r2(va),
                    f"{side_b.label} ms/step": r2(vb),
                    "Diff": r2(va - vb),
                    "Ratio": ratio(va, vb),
                    f"{side_a.label} %": r2(va / sa.kernel_ms * 100) if sa.kernel_ms else None,
                    f"{side_b.label} %": r2(vb / sb.kernel_ms * 100) if sb.kernel_ms else None,
                }
            )
    return pd.DataFrame(rows)


def build_top_kernels(side_a, side_b, classifier) -> pd.DataFrame:
    rows = []
    for phase in _phases(side_a, side_b):
        for side in (side_a, side_b):
            st = get_phase_stats(side.stats, phase)
            ranked = sorted(st.per_kernel_ms.items(), key=lambda kv: -kv[1])[:TOP_KERNELS]
            for rank, (name, ms) in enumerate(ranked, 1):
                if ms < MIN_MS:
                    continue
                rows.append(
                    {
                        "Phase": _phase_label(phase),
                        "GPU": side.label,
                        "Rank": rank,
                        "Kernel": name,
                        "Category": classifier.classify(name),
                        "ms/step": r2(ms),
                        "calls/step": r2(st.per_kernel_calls.get(name, 0)),
                        "% of phase kernel": r2(ms / st.kernel_ms * 100) if st.kernel_ms else None,
                    }
                )
    return pd.DataFrame(rows)


def build_streams(side_a, side_b) -> pd.DataFrame:
    rows = []
    phases = _phases(side_a, side_b)
    for side in (side_a, side_b):
        phase_tids = set(side.trace.phase_tids)
        totals = {}
        for phase in phases:
            for tid, ms in get_phase_stats(side.stats, phase).per_stream_ms.items():
                totals[tid] = totals.get(tid, 0.0) + ms
        grand = sum(totals.values())
        for tid in sorted(totals, key=lambda t: -totals[t]):
            rows.append(
                {
                    "GPU": side.label,
                    "Stream tid": tid,
                    "Carries phase markers": "yes" if tid in phase_tids else "no",
                    "Kernels in trace": side.trace.kernel_tid_counts.get(tid, 0),
                    "Kernel ms/step (decode)": r2(totals[tid]),
                    "Share %": r2(totals[tid] / grand * 100) if grand else None,
                }
            )
    return pd.DataFrame(rows)


def build_graph_off_ops(ops_a, ops_b, label_a, label_b) -> pd.DataFrame:
    rows = []
    for phase in phase_order(ops_a, ops_b):
        a = ops_a.get(phase, {})
        b = ops_b.get(phase, {})
        for op in set(a) | set(b):
            ca, cb = a.get(op, 0.0), b.get(op, 0.0)
            rows.append(
                {
                    "Phase": _phase_label(phase),
                    "Op": op,
                    f"{label_a} calls/step": r2(ca),
                    f"{label_b} calls/step": r2(cb),
                    "Diff": r2(ca - cb),
                    "Only on": label_a if cb == 0 else (label_b if ca == 0 else ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Phase", "Op", "Note"])
    df = pd.DataFrame(rows)
    df["_rank"] = df["Diff"].abs().fillna(0) + df[[f"{label_a} calls/step", f"{label_b} calls/step"]].max(axis=1) / 1000.0
    df = (
        df.sort_values(["Phase", "_rank"], ascending=[True, False])
        .groupby("Phase", sort=False)
        .head(TOP_OPS)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return df


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets:
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for col_cells in ws.columns:
                width = max((len(str(c.value)) for c in col_cells if c.value is not None), default=8)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(width + 2, 10), 60)
    return path
