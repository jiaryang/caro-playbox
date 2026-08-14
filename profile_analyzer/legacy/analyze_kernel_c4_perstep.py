import gzip
import json
import os
import re
import statistics
from collections import defaultdict

OUTLIER_MS = 100
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


def shorten_kernel(name, max_len=70):
    name = re.sub(r"\s+", " ", name.strip())
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def analyze(path):
    with gzip.open(path, "rt") as fh:
        events = json.load(fh)["traceEvents"]

    phases = []
    phase_marker_ms = defaultdict(float)
    phase_marker_n = defaultdict(int)
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        pname = phase_name(e.get("name", ""))
        if not pname:
            continue
        dur_ms = e.get("dur", 0) / 1000
        if dur_ms > OUTLIER_MS:
            continue
        phases.append({"phase": pname, "start": e["ts"], "end": e["ts"] + e.get("dur", 0)})
        phase_marker_ms[pname] += dur_ms
        phase_marker_n[pname] += 1

    kernels = []
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0)
        if dur <= 0:
            continue
        kernels.append(
            {
                "name": e.get("name", "unknown"),
                "start": e["ts"],
                "end": e["ts"] + dur,
                "dur_ms": dur / 1000,
            }
        )

    phase_kernel = defaultdict(lambda: defaultdict(float))
    phase_cat = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        kdur = k["dur_ms"]
        kstart, kend = k["start"], k["end"]
        span = kend - kstart
        if span <= 0:
            continue
        for p in phases:
            overlap_start = max(kstart, p["start"])
            overlap_end = min(kend, p["end"])
            if overlap_end > overlap_start:
                frac = (overlap_end - overlap_start) / span
                part = kdur * frac
                phase_kernel[p["phase"]][k["name"]] += part
                phase_cat[p["phase"]][categorize_kernel(k["name"])] += part

    return {
        "path": path,
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
        print(f"Loaded {key}: {os.path.basename(path)}")

    cats = ["attention", "gemm", "comm", "norm_rope", "moe_topk", "kv_cache", "quant", "misc_gpu", "other"]
    phases = ["draft", "target_verify", "draft_extend"]

    print(f"\n=== Per-step analysis (outliers > {OUTLIER_MS}ms excluded) ===")
    print("Per-step = total / #phase_steps (clean phase marker count)\n")

    for workload in ["8k", "70k"]:
        print("=" * 92)
        print(f"{workload.upper()} conc=4 — Phase marker per-step (ms/step)")
        print("=" * 92)
        print(f"{'GPU':<6} {'phase':<14} {'#steps':>6} {'total_ms':>10} {'per_step':>10}")
        print("-" * 52)
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            for ph in phases:
                n = r["phase_marker_n"].get(ph, 0)
                total = r["phase_marker_ms"].get(ph, 0)
                print(f"{gpu:<6} {ph:<14} {n:>6} {total:>10.2f} {per_step(total, n):>10.3f}")

        print()
        print(f"{workload.upper()} conc=4 — Kernel per-step by category (ms/step)")
        print(f"{'GPU/phase':<22}" + "".join(f"{c:>9}" for c in cats) + f"{'TOTAL':>9}  {'#steps':>6}")
        print("-" * (22 + 9 * len(cats) + 9 + 8))
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            for ph in phases:
                n = r["phase_marker_n"].get(ph, 0)
                if n == 0:
                    continue
                row = f"{gpu}/{ph:<14}"
                total = 0.0
                for c in cats:
                    v = per_step(r["phase_cat"].get(ph, {}).get(c, 0), n)
                    row += f"{v:>9.3f}"
                    total += v
                row += f"{total:>9.3f}  {n:>6}"
                print(row)

        print()
        print(f"{workload.upper()} conc=4 — B200 vs MI355 kernel per-step ratio")
        print(f"{'phase/category':<24} {'B200':>8} {'MI355':>8} {'MI/B200':>8}")
        print("-" * 52)
        b200 = results.get((workload, "b200"))
        mi355 = results.get((workload, "mi355"))
        if b200 and mi355:
            for ph in phases:
                nb = b200["phase_marker_n"].get(ph, 0)
                nm = mi355["phase_marker_n"].get(ph, 0)
                if nb == 0 or nm == 0:
                    continue
                bt = per_step(sum(b200["phase_cat"].get(ph, {}).values()), nb)
                mt = per_step(sum(mi355["phase_cat"].get(ph, {}).values()), nm)
                ratio = mt / bt if bt > 0.001 else float("inf")
                print(f"{ph + ' TOTAL':<24} {bt:>8.3f} {mt:>8.3f} {ratio:>7.2f}x")
                for c in cats:
                    bv = per_step(b200["phase_cat"].get(ph, {}).get(c, 0), nb)
                    mv = per_step(mi355["phase_cat"].get(ph, {}).get(c, 0), nm)
                    if bv < 0.01 and mv < 0.01:
                        continue
                    ratio = mv / bv if bv > 0.001 else float("inf")
                    print(f"  {c:<22} {bv:>8.3f} {mv:>8.3f} {ratio:>7.2f}x")

        for ph in phases:
            print()
            print(f"{workload.upper()} conc=4 — Top kernels per-step in `{ph}` (ms/step)")
            print(f"{'#':<3} {'B200':>8} {'MI355':>8} {'MI/B200':>8}  kernel")
            print("-" * 88)
            b200 = results.get((workload, "b200"), {}).get("phase_kernel", {}).get(ph, {})
            mi355 = results.get((workload, "mi355"), {}).get("phase_kernel", {}).get(ph, {})
            nb = results.get((workload, "b200"), {}).get("phase_marker_n", {}).get(ph, 0)
            nm = results.get((workload, "mi355"), {}).get("phase_marker_n", {}).get(ph, 0)
            all_names = set(b200) | set(mi355)
            ranked = sorted(all_names, key=lambda n: b200.get(n, 0) / max(nb, 1) + mi355.get(n, 0) / max(nm, 1), reverse=True)
            for i, name in enumerate(ranked[:12], 1):
                b = b200.get(name, 0) / nb if nb else 0
                m = mi355.get(name, 0) / nm if nm else 0
                ratio = f"{m/b:.2f}x" if b > 0.001 else "n/a"
                print(f"{i:<3} {b:>8.3f} {m:>8.3f} {ratio:>8}  {shorten_kernel(name)}")

    print()
    print("=" * 92)
    print("Cross-workload summary — decode path per-step (phase marker, ms/step)")
    print("=" * 92)
    print(f"{'workload':<8} {'GPU':<6} {'draft':>8} {'t_verify':>10} {'d_extend':>10} {'sum':>8}")
    print("-" * 56)
    for workload in ["8k", "70k"]:
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            vals = []
            s = 0
            for ph in phases:
                v = per_step(r["phase_marker_ms"].get(ph, 0), r["phase_marker_n"].get(ph, 0))
                vals.append(v)
                s += v
            print(f"{workload:<8} {gpu:<6} {vals[0]:>8.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {s:>8.3f}")


if __name__ == "__main__":
    main()
