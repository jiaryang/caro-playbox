"""Single-side hierarchical profile report.

Layers (top -> bottom):
  1. decode totals  — wall / kernel / overlap / non-kernel gap
  2. phases         — nomtp: ``decode``; mtp: ``draft``, ``target_verify``,
                      ``draft_extend`` (whatever markers the trace carries)
  3. kernel groups  — category ms/step inside each phase
  4. top kernels    — named kernels inside each phase

Accepts either one DECODE trace or a profile directory with ``*_cN-*DECODE*``
files (one row / sheet dimension per concurrency).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .kernels import BASE_RULES, GLM52_RULES, KernelClassifier
from .phases import decode_totals, phase_order
from .report import MIN_MS, r2, write_excel
from .sweep import build_run, discover

TOP_KERNELS = 15
# Preferred MTP phase order when present; unknowns append in first-seen order.
PREFERRED_PHASES = ("draft", "target_verify", "draft_extend", "decode")


def ordered_phases(stats: dict) -> list:
    present = set(stats)
    out = [p for p in PREFERRED_PHASES if p in present]
    for p in phase_order(stats):
        if p not in out:
            out.append(p)
    return out


def load_runs(label: str, path: str, classifier) -> list:
    """-> list[Run]. ``path`` is a file or a directory of DECODE traces."""
    if os.path.isfile(path):
        # Fake conc=0 for a lone file; basename still carries _cN if present.
        from .sweep import CONC_RE

        m = CONC_RE.search(os.path.basename(path))
        conc = int(m.group(1)) if m else 0
        return [build_run(label, conc, path, classifier)]
    try:
        found = discover(path)
    except ValueError as exc:
        raise SystemExit(f"no usable DECODE traces in {path}: {exc}") from exc
    return [build_run(label, c, p, classifier) for c, p in sorted(found.items())]


def build_readme(runs, label, classifier, source: str) -> pd.DataFrame:
    phases = ordered_phases(runs[0].stats) if runs else []
    rows = [
        ("Report", "Single-side hierarchical decode profile"),
        ("Label", label),
        ("Source", source),
        ("Timing", "CUDA-graph-ON DECODE traces; median phase wall, ms/step"),
        ("Layers", "decode totals → phases → kernel categories → top kernels"),
        ("Phases detected", ", ".join(phases) if phases else "(none)"),
        ("MTP expected phases", "draft, target_verify, draft_extend"),
        ("Non-MTP expected phase", "decode"),
        ("Kernel categories", classifier.rules_path),
        ("Concurrencies", ", ".join(str(r.conc) for r in runs)),
    ]
    for r in runs:
        rows.append((f"c{r.conc} trace", r.basename))
        rows.append((f"c{r.conc} bs / steps", f"bs={r.bs} steps={r.steps}"))
        rows.append(
            (
                f"c{r.conc} phase streams",
                ", ".join(str(t) for t in r.phase_tids),
            )
        )
    return pd.DataFrame(rows, columns=["Item", "Value"])


def build_decode_summary(runs, label: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        wall, kern = decode_totals(r.stats)
        rows.append(
            {
                "Label": label,
                "Conc": r.conc,
                "BatchSize": r.bs,
                "Steps": r.steps,
                "Decode wall ms/step": r2(wall),
                "Decode kernel ms/step": r2(kern),
                "Overlap (kern/wall)": r2(kern / wall) if wall else None,
                "Non-kernel gap ms/step": r2(wall - kern),
                "Phases": ",".join(ordered_phases(r.stats)),
            }
        )
    return pd.DataFrame(rows)


def build_phase_breakdown(runs, label: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        wall_tot, kern_tot = decode_totals(r.stats)
        for phase in ordered_phases(r.stats):
            st = r.stats[phase]
            rows.append(
                {
                    "Label": label,
                    "Conc": r.conc,
                    "BatchSize": r.bs,
                    "Phase": phase,
                    "Steps": st.n_steps,
                    "Wall ms/step": r2(st.wall_ms),
                    "Kernel ms/step": r2(st.kernel_ms),
                    "Overlap": r2(st.overlap_factor),
                    "Non-kernel gap": r2(st.non_kernel_gap_ms),
                    "Wall share of decode %": r2(100 * st.wall_ms / wall_tot) if wall_tot else None,
                    "Kernel share of decode %": r2(100 * st.kernel_ms / kern_tot)
                    if kern_tot
                    else None,
                }
            )
        rows.append(
            {
                "Label": label,
                "Conc": r.conc,
                "BatchSize": r.bs,
                "Phase": "decode total",
                "Steps": r.steps,
                "Wall ms/step": r2(wall_tot),
                "Kernel ms/step": r2(kern_tot),
                "Overlap": r2(kern_tot / wall_tot) if wall_tot else None,
                "Non-kernel gap": r2(wall_tot - kern_tot),
                "Wall share of decode %": 100.0,
                "Kernel share of decode %": 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_kernel_category(runs, label: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        for phase in ordered_phases(r.stats):
            st = r.stats[phase]
            phase_kern = st.kernel_ms or 0.0
            cats = sorted(st.per_category_ms.items(), key=lambda kv: -kv[1])
            for cat, ms in cats:
                if ms < MIN_MS:
                    continue
                rows.append(
                    {
                        "Label": label,
                        "Conc": r.conc,
                        "Phase": phase,
                        "Category": cat,
                        "ms/step": r2(ms),
                        "Share of phase kernel %": r2(100 * ms / phase_kern)
                        if phase_kern
                        else None,
                    }
                )
    return pd.DataFrame(rows)


def build_top_kernels(runs, label: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        for phase in ordered_phases(r.stats):
            st = r.stats[phase]
            phase_kern = st.kernel_ms or 0.0
            ranked = sorted(st.per_kernel_ms.items(), key=lambda kv: -kv[1])[:TOP_KERNELS]
            for name, ms in ranked:
                if ms < MIN_MS:
                    continue
                rows.append(
                    {
                        "Label": label,
                        "Conc": r.conc,
                        "Phase": phase,
                        "Kernel": name,
                        "ms/step": r2(ms),
                        "calls/step": r2(st.per_kernel_calls.get(name, 0.0)),
                        "Share of phase kernel %": r2(100 * ms / phase_kern)
                        if phase_kern
                        else None,
                    }
                )
    return pd.DataFrame(rows)


def build_streams(runs, label: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        for phase in ordered_phases(r.stats):
            st = r.stats[phase]
            for tid, ms in sorted(st.per_stream_ms.items(), key=lambda kv: -kv[1]):
                rows.append(
                    {
                        "Label": label,
                        "Conc": r.conc,
                        "Phase": phase,
                        "Stream tid": tid,
                        "Kernel ms/step": r2(ms),
                    }
                )
    return pd.DataFrame(rows)


def print_hierarchy(runs, label: str) -> None:
    for r in runs:
        wall, kern = decode_totals(r.stats)
        print(f"\n=== {label}  c{r.conc}  bs={r.bs}  steps={r.steps} ===")
        print(
            f"  decode  wall={wall:.2f} ms  kernel={kern:.2f} ms  "
            f"overlap={kern / wall if wall else 0:.2f}  gap={wall - kern:.2f} ms"
        )
        for phase in ordered_phases(r.stats):
            st = r.stats[phase]
            print(
                f"    [{phase}] wall={st.wall_ms:.2f}  kern={st.kernel_ms:.2f}  "
                f"overlap={st.overlap_factor:.2f}  gap={st.non_kernel_gap_ms:.2f}"
            )
            cats = sorted(st.per_category_ms.items(), key=lambda kv: -kv[1])[:8]
            for cat, ms in cats:
                if ms < MIN_MS:
                    continue
                print(f"      · {cat}: {ms:.2f} ms")


def analyze_path(path: str, label: str, rules: str, output: str) -> int:
    classifier = KernelClassifier(rules)
    runs = load_runs(label, path, classifier)
    if not runs:
        raise SystemExit(f"no DECODE traces in {path}")
    print_hierarchy(runs, label)
    sheets = [
        ("README", build_readme(runs, label, classifier, os.path.abspath(path))),
        ("Decode_Summary", build_decode_summary(runs, label)),
        ("Phase_Breakdown", build_phase_breakdown(runs, label)),
        ("Kernel_Category", build_kernel_category(runs, label)),
        ("Top_Kernels", build_top_kernels(runs, label)),
        ("Streams", build_streams(runs, label)),
    ]
    out = os.path.abspath(output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_excel(out, sheets)
    print(f"\nWrote {out}")
    return 0


def validate_runs(
    runs,
    *,
    expected_steps: int,
    min_steps: int,
    max_wall_ms: float,
    mode: str,
) -> list[str]:
    """Return human-readable issues; empty list means OK."""
    issues: list[str] = []
    if not runs:
        return ["no DECODE traces loaded"]

    max_steps = max(expected_steps * 3, expected_steps + 10)
    expect_mtp = mode.strip().lower() in ("mtp", "eagle")
    expect_nomtp = mode.strip().lower() in ("nomtp", "non-mtp", "baseline", "")

    for r in runs:
        tag = f"c{r.conc}"
        if r.steps < min_steps:
            issues.append(
                f"{tag}: steps={r.steps} < min={min_steps} "
                f"(expected~{expected_steps})"
            )
        elif r.steps > max_steps:
            issues.append(
                f"{tag}: steps={r.steps} > max={max_steps} "
                f"(expected~{expected_steps}; possible duplicate/corrupt markers)"
            )

        wall, _kern = decode_totals(r.stats)
        if wall <= 0:
            issues.append(f"{tag}: decode wall={wall:.3f} ms (non-positive)")
        elif wall > max_wall_ms:
            issues.append(
                f"{tag}: decode wall={wall:.2f} ms/step > max={max_wall_ms:.0f} "
                f"(abnormally high)"
            )

        for phase, st in r.stats.items():
            if st.n_steps < min_steps:
                issues.append(
                    f"{tag}/{phase}: steps={st.n_steps} < min={min_steps}"
                )
            if st.wall_ms > max_wall_ms:
                issues.append(
                    f"{tag}/{phase}: wall={st.wall_ms:.2f} ms/step > "
                    f"max={max_wall_ms:.0f}"
                )
            # Stuck / pathological: near-zero kernel with large wall.
            # Skip draft_extend — MI355 often spends wall with almost no kernels.
            if (
                phase != "draft_extend"
                and st.wall_ms > 50.0
                and st.kernel_ms < 0.05 * st.wall_ms
            ):
                issues.append(
                    f"{tag}/{phase}: kernel={st.kernel_ms:.2f} vs wall="
                    f"{st.wall_ms:.2f} (overlap suspiciously low)"
                )

        if expect_mtp:
            # analyze_phases always inserts MTP keys; empty means n_steps==0.
            for need in ("draft", "target_verify", "draft_extend"):
                st = r.stats.get(need)
                if st is None or st.n_steps == 0:
                    issues.append(
                        f"{tag}: empty MTP phase '{need}' "
                        f"(n_steps={0 if st is None else st.n_steps})"
                    )
        elif expect_nomtp and r.stats:
            phases = set(r.stats)
            if not any(p == "decode" or p.startswith("decode") for p in phases):
                issues.append(
                    f"{tag}: expected nomtp phase 'decode' "
                    f"(have: {','.join(sorted(phases))})"
                )

    return issues


def validate_path(
    path: str,
    *,
    label: str,
    rules: str,
    expected_steps: int,
    min_steps: int,
    max_wall_ms: float,
    mode: str,
) -> int:
    classifier = KernelClassifier(rules)
    try:
        runs = load_runs(label, path, classifier)
    except SystemExit as exc:
        print(f"TRACE VALIDATION FAILED: {exc}")
        return 1
    except ValueError as exc:
        print(f"TRACE VALIDATION FAILED: {exc}")
        return 1
    issues = validate_runs(
        runs,
        expected_steps=expected_steps,
        min_steps=min_steps,
        max_wall_ms=max_wall_ms,
        mode=mode,
    )
    for r in runs:
        wall, kern = decode_totals(r.stats)
        print(
            f"validate {label} c{r.conc}: steps={r.steps} "
            f"wall={wall:.2f}ms kern={kern:.2f}ms "
            f"phases={','.join(ordered_phases(r.stats))}"
        )
    if issues:
        print("TRACE VALIDATION FAILED:")
        for msg in issues:
            print(f"  - {msg}")
        return 1
    print("TRACE VALIDATION OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", required=True, help="profile directory or one DECODE .json.gz")
    ap.add_argument("--label", default="run", help="side label (e.g. nomtp / mtp)")
    ap.add_argument("-o", "--output", metavar="XLSX", help="required unless --validate")
    ap.add_argument(
        "--rules",
        default=GLM52_RULES,
        help=f"kernel category CSV (default: GLM-5.2; base: {BASE_RULES})",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="sanity-check step count / wall times; exit 1 if bad",
    )
    ap.add_argument("--expected-steps", type=int, default=20)
    ap.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="minimum steps (default: 75%% of --expected-steps, at least 5)",
    )
    ap.add_argument(
        "--max-wall-ms",
        type=float,
        default=500.0,
        help="max decode/phase wall ms/step before treating as abnormal",
    )
    ap.add_argument(
        "--mode",
        default="",
        help="nomtp|mtp — enables expected phase checks",
    )
    args = ap.parse_args(argv)

    if args.validate:
        min_steps = args.min_steps
        if min_steps <= 0:
            min_steps = max(5, int(args.expected_steps * 0.75))
        return validate_path(
            args.dir,
            label=args.label,
            rules=args.rules,
            expected_steps=args.expected_steps,
            min_steps=min_steps,
            max_wall_ms=args.max_wall_ms,
            mode=args.mode or args.label,
        )

    if not args.output:
        ap.error("-o/--output is required unless --validate")
    return analyze_path(args.dir, args.label, args.rules, args.output)


if __name__ == "__main__":
    sys.exit(main())
