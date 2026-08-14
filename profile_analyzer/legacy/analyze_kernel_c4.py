import gzip
import json
import os
import re
from collections import defaultdict

OUTLIER_MS = 100
BASE_DIRS = {
    "8k": os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("2_70k_profile_MTP", "2_8k_profile_MTP"), ""),
    "70k": os.path.dirname(os.path.abspath(__file__)),
}

# Fix path: script may live in either folder; search both.
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOTS = [
    os.path.join(os.path.dirname(ROOT), "2_8k_profile_MTP"),
    os.path.join(os.path.dirname(ROOT), "2_70k_profile_MTP"),
]

# 8k decode path markers live in DECODE traces (EXTEND traces are prefill-only).
# mi355 8k DECODE trace is currently corrupt; fall back to EXTEND (partial coverage).
FILES = {
    ("8k", "b200"): [
        "glm_mtp_glm_b200_i8192_c4-*-TP-0-DECODE.trace.json.gz",
        "glm_mtp_glm_b200_i8192_c4-*-TP-0-EXTEND.trace.json.gz",
    ],
    ("8k", "mi355"): [
        "glm_mtp_glm_mi355_i8192_c4-*-TP-0-DECODE.trace.json.gz",
        "glm_mtp_glm_mi355_i8192_c4-*-TP-0-EXTEND.trace.json.gz",
    ],
    ("70k", "b200"): ["glm_mtp_glm_b200_i70000_c4-*-TP-0-EXTEND.trace.json.gz"],
    ("70k", "mi355"): ["glm_mtp_glm_mi355_i70000_c4-*-TP-0-EXTEND.trace.json.gz"],
}


def find_file(patterns):
    import glob

    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        for root in ROOTS:
            matches = glob.glob(os.path.join(root, pattern))
            if not matches:
                continue
            for path in matches:
                try:
                    with gzip.open(path, "rt") as fh:
                        json.load(fh)
                    return path, pattern
                except (EOFError, OSError, json.JSONDecodeError):
                    continue
    return None, None


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
    if any(k in n for k in ["allreduce", "reduce_scatter", "nccl", "mnnvl_allreduce"]):
        return "comm"
    if any(
        k in n for k in [
            "gemm", "bmm", "nvjet", "hgemm", "deep_gemm", "cijk_", "moe_gemm",
            "a4w4", "a8w8", "batched_gemm",
        ]
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


def shorten_kernel(name, max_len=72):
    name = re.sub(r"\s+", " ", name.strip())
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def analyze(path):
    with gzip.open(path, "rt") as fh:
        events = json.load(fh)["traceEvents"]

    phases = []
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        pname = phase_name(e.get("name", ""))
        if not pname:
            continue
        dur_ms = e.get("dur", 0) / 1000
        if dur_ms > OUTLIER_MS:
            continue
        phases.append(
            {
                "phase": pname,
                "start": e["ts"],
                "end": e["ts"] + e.get("dur", 0),
            }
        )

    kernels = []
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if e.get("cat") != "kernel":
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

    # Assign kernel time to phases by overlap fraction.
    phase_kernel = defaultdict(lambda: defaultdict(float))
    phase_cat = defaultdict(lambda: defaultdict(float))
    unassigned = 0.0

    for k in kernels:
        kdur = k["dur_ms"]
        kstart, kend = k["start"], k["end"]
        assigned = 0.0
        for p in phases:
            overlap_start = max(kstart, p["start"])
            overlap_end = min(kend, p["end"])
            if overlap_end > overlap_start:
                frac = (overlap_end - overlap_start) / (kend - kstart)
                part = kdur * frac
                phase_kernel[p["phase"]][k["name"]] += part
                phase_cat[p["phase"]][categorize_kernel(k["name"])] += part
                assigned += part
        unassigned += max(0.0, kdur - assigned)

    phase_total = {p: sum(phase_kernel[p].values()) for p in phase_kernel}
    return {
        "phases": phases,
        "phase_kernel": phase_kernel,
        "phase_cat": phase_cat,
        "phase_total": phase_total,
        "unassigned_ms": unassigned,
    }


def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def main():
    results = {}
    meta = {}
    for key, patterns in FILES.items():
        path, used = find_file(patterns)
        if not path:
            print(f"MISSING {key}")
            continue
        results[key] = analyze(path)
        meta[key] = used
        print(f"Loaded {key}: {os.path.basename(path)}  [{used}]")

    print_header("Trace file notes")
    for key in [("8k", "b200"), ("8k", "mi355"), ("70k", "b200"), ("70k", "mi355")]:
        if key in meta:
            note = ""
            if key[0] == "8k" and "DECODE" not in meta[key]:
                note = "  WARNING: decode trace unavailable, only partial phases"
            print(f"  {key[0]} {key[1]}: {meta[key]}{note}")

    for workload in ["8k", "70k"]:
        print_header(f"{workload.upper()} conc=4 — Phase kernel totals (ms, outliers>{OUTLIER_MS}ms excluded)")
        print(f"{'GPU':<6} {'phase':<14} {'kernel_ms':>10} {'#phase_steps':>12}")
        print("-" * 50)
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            step_counts = defaultdict(int)
            for p in r["phases"]:
                step_counts[p["phase"]] += 1
            for phase in ["draft", "target_verify", "draft_extend"]:
                print(
                    f"{gpu:<6} {phase:<14} {r['phase_total'].get(phase,0):>10.2f} "
                    f"{step_counts.get(phase,0):>12}"
                )

    for workload in ["8k", "70k"]:
        print_header(f"{workload.upper()} conc=4 — Kernel category breakdown by phase (ms)")
        phases = ["draft", "target_verify", "draft_extend"]
        cats = ["attention", "gemm", "comm", "norm_rope", "moe_topk", "kv_cache", "quant", "misc_gpu", "other"]
        header = f"{'GPU/phase':<22}" + "".join(f"{c:>10}" for c in cats) + f"{'TOTAL':>10}"
        print(header)
        print("-" * len(header))
        for gpu in ["b200", "mi355"]:
            r = results.get((workload, gpu))
            if not r:
                continue
            for phase in phases:
                row = f"{gpu}/{phase:<14}"
                total = 0.0
                vals = []
                for c in cats:
                    v = r["phase_cat"].get(phase, {}).get(c, 0.0)
                    vals.append(v)
                    total += v
                row += "".join(f"{v:>10.1f}" for v in vals)
                row += f"{total:>10.1f}"
                print(row)

    for workload in ["8k", "70k"]:
        for phase in ["draft", "draft_extend", "target_verify"]:
            print_header(f"{workload.upper()} conc=4 — Top kernels in `{phase}` (ms)")
            print(f"{'rank':<4} {'B200 ms':>10} {'MI355 ms':>10} {'MI/B200':>8}  kernel")
            print("-" * 90)
            b200 = results.get((workload, "b200"), {}).get("phase_kernel", {}).get(phase, {})
            mi355 = results.get((workload, "mi355"), {}).get("phase_kernel", {}).get(phase, {})
            all_names = set(b200) | set(mi355)
            ranked = sorted(
                all_names,
                key=lambda n: b200.get(n, 0) + mi355.get(n, 0),
                reverse=True,
            )
            for i, name in enumerate(ranked[:15], 1):
                b = b200.get(name, 0.0)
                m = mi355.get(name, 0.0)
                ratio = m / b if b > 0.01 else float("inf")
                ratio_s = f"{ratio:.1f}x" if b > 0.01 else "n/a"
                print(f"{i:<4} {b:>10.2f} {m:>10.2f} {ratio_s:>8}  {shorten_kernel(name)}")

    print_header("B200 vs MI355 category totals across all decode phases (ms)")
    for workload in ["8k", "70k"]:
        print(f"\n--- {workload.upper()} ---")
        cats = ["attention", "gemm", "comm", "norm_rope", "moe_topk", "kv_cache", "quant", "misc_gpu", "other"]
        print(f"{'category':<12} {'B200':>10} {'MI355':>10} {'MI/B200':>10}")
        print("-" * 46)
        b200 = results.get((workload, "b200"))
        mi355 = results.get((workload, "mi355"))
        if not b200 or not mi355:
            continue
        for c in cats:
            b = sum(b200["phase_cat"].get(p, {}).get(c, 0.0) for p in b200["phase_cat"])
            m = sum(mi355["phase_cat"].get(p, {}).get(c, 0.0) for p in mi355["phase_cat"])
            ratio = m / b if b > 0.01 else float("inf")
            ratio_s = f"{ratio:.2f}x" if b > 0.01 else "n/a"
            print(f"{c:<12} {b:>10.1f} {m:>10.1f} {ratio_s:>10}")


if __name__ == "__main__":
    main()
