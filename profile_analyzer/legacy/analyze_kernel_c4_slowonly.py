import gzip
import json
import os
import re
import statistics
from collections import defaultdict

OUTLIER_MS = 100
SLOW_THRESHOLD_MS = 5.0  # fast/slow split for target_verify & draft_extend

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOTS = [
    os.path.join(os.path.dirname(ROOT), "2_8k_profile_MTP"),
    os.path.join(os.path.dirname(ROOT), "2_70k_profile_MTP"),
]

FILES = {
    ("8k", "b200"): ["glm_mtp_glm_b200_i8192_c4-*-TP-0-DECODE.trace.json.gz"],
    ("8k", "mi355"): ["glm_mtp_glm_mi355_i8192_c4-*-TP-0-DECODE.trace.json.gz"],
    ("70k", "b200"): ["glm_mtp_glm_b200_i70000_c4-*-TP-0-EXTEND.trace.json.gz"],
    ("70k", "mi355"): ["glm_mtp_glm_mi355_i70000_c4-*-TP-0-EXTEND.trace.json.gz"],
}
PHASES = ["draft", "target_verify", "draft_extend"]
CATS = ["attention", "gemm", "comm", "norm_rope", "moe_topk", "kv_cache", "quant", "misc_gpu", "other"]


def find_file(patterns):
    import glob

    best = None
    for pattern in patterns:
        for root in ROOTS:
            for path in glob.glob(os.path.join(root, pattern)):
                try:
                    with gzip.open(path, "rt") as fh:
                        json.load(fh)
                    if best is None or os.path.getmtime(path) > os.path.getmtime(best):
                        best = path
                except (EOFError, OSError, json.JSONDecodeError):
                    continue
    return best


def phase_name(raw):
    if raw == "draft":
        return "draft"
    if raw == "draft_extend":
        return "draft_extend"
    if "TARGET_VERIFY" in raw:
        return "target_verify"
    return None


def categorize_kernel(name):
    n = name.lower()
    if any(k in n for k in ["fmha", "mqa_logits", "sparse_mla", "flash_attn", "attention"]):
        return "attention"
    if any(k in n for k in ["allreduce", "reduce_scatter", "nccl", "mnnvl_allreduce", "quickreduce"]):
        return "comm"
    if any(
        k in n
        for k in ["gemm", "bmm", "nvjet", "hgemm", "deep_gemm", "cijk_", "moe_gemm", "a4w4", "a8w8", "batched_gemm"]
    ):
        return "gemm"
    if any(k in n for k in ["rmsnorm", "norm", "rope", "hadamard"]):
        return "norm_rope"
    if any(k in n for k in ["topk", "moe", "finalize", "sort_quant"]):
        return "moe_topk"
    if any(k in n for k in ["cache", "paged", "kv", "indexer"]):
        return "kv_cache"
    if any(k in n for k in ["quant", "fp4", "fp8", "mxfp", "e2m1", "e4m3"]):
        return "quant"
    if any(k in n for k in ["verify_tree", "copy", "elementwise", "vectorized"]):
        return "misc_gpu"
    return "other"


def shorten_kernel(name, max_len=68):
    name = re.sub(r"\s+", " ", name.strip())
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def is_slow_step(phase, dur_ms):
    if dur_ms > OUTLIER_MS:
        return False
    if phase in ("target_verify", "draft_extend"):
        return dur_ms >= SLOW_THRESHOLD_MS
    # draft: keep all non-outlier (no clear fast/slow split on B200)
    return True


def analyze(path):
    with gzip.open(path, "rt") as fh:
        events = json.load(fh)["traceEvents"]

    all_steps = defaultdict(list)
    slow_phases = []
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        pname = phase_name(e.get("name", ""))
        if not pname:
            continue
        dur_ms = e.get("dur", 0) / 1000
        all_steps[pname].append(dur_ms)
        if is_slow_step(pname, dur_ms):
            slow_phases.append(
                {"phase": pname, "start": e["ts"], "end": e["ts"] + e.get("dur", 0), "dur_ms": dur_ms}
            )

    kernels = []
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0)
        if dur <= 0:
            continue
        kernels.append(
            {"name": e.get("name", "unknown"), "start": e["ts"], "end": e["ts"] + dur, "dur_ms": dur / 1000}
        )

    phase_marker_ms = defaultdict(float)
    phase_marker_n = defaultdict(int)
    phase_kernel = defaultdict(lambda: defaultdict(float))
    phase_cat = defaultdict(lambda: defaultdict(float))

    for p in slow_phases:
        phase_marker_ms[p["phase"]] += p["dur_ms"]
        phase_marker_n[p["phase"]] += 1

    for k in kernels:
        kdur, kstart, kend = k["dur_ms"], k["start"], k["end"]
        span = kend - kstart
        if span <= 0:
            continue
        for p in slow_phases:
            os_ = max(kstart, p["start"])
            oe = min(kend, p["end"])
            if oe > os_:
                frac = (oe - os_) / span
                part = kdur * frac
                phase_kernel[p["phase"]][k["name"]] += part
                phase_cat[p["phase"]][categorize_kernel(k["name"])] += part

    return {
        "path": os.path.basename(path),
        "all_steps": dict(all_steps),
        "phase_marker_ms": dict(phase_marker_ms),
        "phase_marker_n": dict(phase_marker_n),
        "phase_kernel": phase_kernel,
        "phase_cat": phase_cat,
    }


def per_step(total, n):
    return total / n if n else 0.0


def main():
    results = {}
    for key, patterns in FILES.items():
        path = find_file(patterns)
        if not path:
            print(f"MISSING {key}")
            continue
        results[key] = analyze(path)

    print(f"=== Slow-step only analysis (threshold: target_verify/draft_extend >= {SLOW_THRESHOLD_MS}ms) ===")
    print("draft: all non-outlier steps kept\n")

    for workload in ["8k", "70k"]:
        print("=" * 88)
        print(f"{workload.upper()} conc=4 — step counts (all vs slow-only)")
        print("=" * 88)
        print(f"{'GPU':<6} {'phase':<14} {'all':>5} {'slow':>5} {'all_med':>9} {'slow_med':>9} {'slow_mean':>9}")
        print("-" * 64)
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            for ph in PHASES:
                all_d = [d for d in r["all_steps"].get(ph, []) if d <= OUTLIER_MS]
                slow_d = [d for d in all_d if is_slow_step(ph, d)]
                ns = r["phase_marker_n"].get(ph, 0)
                print(
                    f"{gpu:<6} {ph:<14} {len(all_d):>5} {ns:>5} "
                    f"{statistics.median(all_d) if all_d else 0:>9.3f} "
                    f"{statistics.median(slow_d) if slow_d else 0:>9.3f} "
                    f"{per_step(r['phase_marker_ms'].get(ph, 0), ns):>9.3f}"
                )

    for workload in ["8k", "70k"]:
        print()
        print("=" * 88)
        print(f"{workload.upper()} conc=4 — Phase marker per-step (SLOW only, ms/step)")
        print("=" * 88)
        print(f"{'GPU':<6} {'phase':<14} {'#slow':>6} {'per_step':>10}")
        print("-" * 36)
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            row_sum = 0
            parts = []
            for ph in PHASES:
                n = r["phase_marker_n"].get(ph, 0)
                ps = per_step(r["phase_marker_ms"].get(ph, 0), n)
                parts.append(f"{ph}={ps:.2f}")
                row_sum += ps
                print(f"{gpu:<6} {ph:<14} {n:>6} {ps:>10.3f}")
            print(f"{'':6} {'TOTAL':<14} {'':6} {row_sum:>10.3f}")
            print()

    for workload in ["8k", "70k"]:
        print("=" * 88)
        print(f"{workload.upper()} conc=4 — Kernel per-step by category (SLOW only)")
        print("=" * 88)
        hdr = f"{'GPU/phase':<22}" + "".join(f"{c:>9}" for c in CATS) + f"{'TOTAL':>9}"
        print(hdr)
        print("-" * len(hdr))
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            for ph in PHASES:
                n = r["phase_marker_n"].get(ph, 0)
                if n == 0:
                    continue
                row = f"{gpu}/{ph:<14}"
                total = 0.0
                for c in CATS:
                    v = per_step(r["phase_cat"].get(ph, {}).get(c, 0), n)
                    row += f"{v:>9.3f}"
                    total += v
                row += f"{total:>9.3f}"
                print(row)

    for workload in ["8k", "70k"]:
        print()
        print(f"{workload.upper()} — B200 vs MI355 slow-step ratio")
        b200, mi355 = results.get((workload, "b200")), results.get((workload, "mi355"))
        if not b200 or not mi355:
            continue
        print(f"{'phase/category':<24} {'B200':>8} {'MI355':>8} {'MI/B200':>8}")
        print("-" * 52)
        for ph in PHASES:
            nb, nm = b200["phase_marker_n"].get(ph, 0), mi355["phase_marker_n"].get(ph, 0)
            if nb == 0 or nm == 0:
                continue
            bt = per_step(sum(b200["phase_cat"].get(ph, {}).values()), nb)
            mt = per_step(sum(mi355["phase_cat"].get(ph, {}).values()), nm)
            print(f"{ph + ' TOTAL':<24} {bt:>8.3f} {mt:>8.3f} {mt/bt if bt>0.001 else 0:>7.2f}x")
            for c in CATS:
                bv = per_step(b200["phase_cat"].get(ph, {}).get(c, 0), nb)
                mv = per_step(mi355["phase_cat"].get(ph, {}).get(c, 0), nm)
                if bv < 0.02 and mv < 0.02:
                    continue
                print(f"  {c:<22} {bv:>8.3f} {mv:>8.3f} {mv/bv if bv>0.001 else 0:>7.2f}x")

    for workload in ["8k", "70k"]:
        for ph in ["target_verify", "draft_extend"]:
            print()
            print(f"{workload.upper()} conc=4 — Top kernels per-step SLOW `{ph}`")
            print(f"{'#':<3} {'B200':>8} {'MI355':>8} {'MI/B200':>8}  kernel")
            print("-" * 86)
            b200 = results.get((workload, "b200"), {})
            mi355 = results.get((workload, "mi355"), {})
            nb = b200.get("phase_marker_n", {}).get(ph, 0)
            nm = mi355.get("phase_marker_n", {}).get(ph, 0)
            bk = b200.get("phase_kernel", {}).get(ph, {})
            mk = mi355.get("phase_kernel", {}).get(ph, {})
            names = set(bk) | set(mk)
            ranked = sorted(names, key=lambda n: bk.get(n, 0) / max(nb, 1) + mk.get(n, 0) / max(nm, 1), reverse=True)
            for i, name in enumerate(ranked[:12], 1):
                b = bk.get(name, 0) / nb if nb else 0
                m = mk.get(name, 0) / nm if nm else 0
                ratio = f"{m/b:.2f}x" if b > 0.001 else "n/a"
                print(f"{i:<3} {b:>8.3f} {m:>8.3f} {ratio:>8}  {shorten_kernel(name)}")

    print()
    print("=" * 88)
    print("Cross-workload summary — SLOW step decode path (phase marker ms/step)")
    print("=" * 88)
    print(f"{'wl':<4} {'GPU':<6} {'draft':>8} {'t_verify':>10} {'d_extend':>10} {'sum':>8}")
    print("-" * 52)
    for wl in ["8k", "70k"]:
        for gpu in ["b200", "mi355"]:
            r = results.get((wl, gpu))
            if not r:
                continue
            vals = [per_step(r["phase_marker_ms"].get(p, 0), r["phase_marker_n"].get(p, 0)) for p in PHASES]
            print(f"{wl:<4} {gpu:<6} {vals[0]:>8.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {sum(vals):>8.3f}")


if __name__ == "__main__":
    main()
