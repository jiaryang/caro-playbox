import gzip
import json
import glob
import os
import re
import statistics
from collections import defaultdict

OUTLIER_MS = 100
ROOTS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("2_70k_profile_MTP", "2_8k_profile_MTP"), ""),
    os.path.dirname(os.path.abspath(__file__)),
]
# fix roots
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOTS = [
    os.path.join(os.path.dirname(ROOT), "2_8k_profile_MTP"),
    os.path.join(os.path.dirname(ROOT), "2_70k_profile_MTP"),
]

CONFIG = {
    ("8k", "mi355"): ("glm_mtp_glm_mi355_i8192_c4-*-TP-0-DECODE.trace.json.gz", 8),
    ("8k", "b200"): ("glm_mtp_glm_b200_i8192_c4-*-TP-0-DECODE.trace.json.gz", 132),
    ("70k", "mi355"): ("glm_mtp_glm_mi355_i70000_c4-*-TP-0-EXTEND.trace.json.gz", 8),
    ("70k", "b200"): ("glm_mtp_glm_b200_i70000_c4-*-TP-0-EXTEND.trace.json.gz", 132),
}
PHASES = ["draft", "target_verify", "draft_extend"]
CATS = ["attention", "gemm", "comm", "norm_rope", "moe_topk", "kv_cache", "quant", "misc_gpu", "other"]


def find_file(pattern):
    best = None
    for root in ROOTS:
        for path in glob.glob(os.path.join(root, pattern)):
            try:
                with gzip.open(path, "rt") as fh:
                    json.load(fh)
                if best is None or os.path.getmtime(path) > os.path.getmtime(best):
                    best = path
            except (EOFError, OSError, json.JSONDecodeError):
                pass
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


def shorten(name, n=68):
    name = re.sub(r"\s+", " ", name.strip())
    return name if len(name) <= n else name[: n - 3] + "..."


def analyze(path, stream_tid):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]

    phases = []
    phase_durs = defaultdict(list)
    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if e.get("cat") != "gpu_user_annotation" or e.get("tid") != stream_tid:
            continue
        pname = phase_name(e.get("name", ""))
        if not pname:
            continue
        dur = e.get("dur", 0) / 1000
        if dur > OUTLIER_MS:
            continue
        phases.append({"phase": pname, "start": e["ts"], "end": e["ts"] + e.get("dur", 0), "dur_ms": dur})
        phase_durs[pname].append(dur)

    kernels = []
    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if e.get("cat") != "kernel" or e.get("tid") != stream_tid:
            continue
        dur = e.get("dur", 0)
        if dur <= 0:
            continue
        kernels.append({"name": e.get("name", ""), "start": e["ts"], "end": e["ts"] + dur, "dur_ms": dur / 1000})

    phase_kernel = defaultdict(lambda: defaultdict(float))
    phase_cat = defaultdict(lambda: defaultdict(float))
    for p in phases:
        for k in kernels:
            os_ = max(k["start"], p["start"])
            oe = min(k["end"], p["end"])
            if oe <= os_:
                continue
            frac = (oe - os_) / (k["end"] - k["start"])
            part = k["dur_ms"] * frac
            phase_kernel[p["phase"]][k["name"]] += part
            phase_cat[p["phase"]][categorize_kernel(k["name"])] += part

    return {
        "path": os.path.basename(path),
        "stream_tid": stream_tid,
        "phase_durs": dict(phase_durs),
        "phase_kernel": phase_kernel,
        "phase_cat": phase_cat,
    }


def med(durs):
    return statistics.median(durs) if durs else 0.0


def per_step(total, n):
    return total / n if n else 0.0


def main():
    results = {}
    for key, (pattern, stream_tid) in CONFIG.items():
        path = find_file(pattern)
        if not path:
            print(f"MISSING {key}")
            continue
        results[key] = analyze(path, stream_tid)
        print(f"Loaded {key}: {os.path.basename(path)}  GPU stream tid={stream_tid}")

    print("\n=== GPU stream focus: MI355 tid=8, B200 tid=132 ===")
    print("Phase markers: gpu_user_annotation on target stream only")
    print("Kernels: cat=kernel on same stream tid only\n")

    for wl in ["8k", "70k"]:
        print("=" * 78)
        print(f"{wl.upper()} conc=4 — Phase marker per-step (ms/step, median)")
        print("=" * 78)
        print(f"{'GPU':<6} {'stream':>6} {'draft':>8} {'t_verify':>10} {'d_extend':>10} {'sum':>8}  {'n_steps':>8}")
        print("-" * 72)
        for gpu in ["b200", "mi355"]:
            r = results.get((wl, gpu))
            if not r:
                continue
            d = {p: med(r["phase_durs"].get(p, [])) for p in PHASES}
            n = max(len(r["phase_durs"].get(p, [])) for p in PHASES)
            ns = {p: len(r["phase_durs"].get(p, [])) for p in PHASES}
            print(
                f"{gpu:<6} {r['stream_tid']:>6} {d['draft']:>8.3f} {d['target_verify']:>10.3f} "
                f"{d['draft_extend']:>10.3f} {sum(d.values()):>8.3f}  tv_n={ns.get('target_verify',0)}"
            )

    for wl in ["8k", "70k"]:
        print()
        print(f"{wl.upper()} — Kernel per-step by category (median phase window count as n)")
        hdr = f"{'GPU/phase':<22}" + "".join(f"{c:>9}" for c in CATS) + f"{'TOTAL':>9}"
        print(hdr)
        print("-" * len(hdr))
        for gpu in ["b200", "mi355"]:
            r = results.get((wl, gpu))
            if not r:
                continue
            for ph in PHASES:
                n = len(r["phase_durs"].get(ph, []))
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

    for wl in ["8k", "70k"]:
        print()
        print(f"{wl.upper()} — B200 vs MI355 per-step ratio (stream-focused)")
        b200, mi = results.get((wl, "b200")), results.get((wl, "mi355"))
        if not b200 or not mi:
            continue
        print(f"{'phase':<16} {'B200':>8} {'MI355':>8} {'MI/B200':>8}")
        for ph in PHASES:
            nb, nm = len(b200["phase_durs"].get(ph, [])), len(mi["phase_durs"].get(ph, []))
            if not nb or not nm:
                continue
            bv = med(b200["phase_durs"].get(ph, []))
            mv = med(mi["phase_durs"].get(ph, []))
            print(f"{ph+' (phase)':<16} {bv:>8.3f} {mv:>8.3f} {mv/bv if bv else 0:>7.2f}x")
            bk = per_step(sum(b200["phase_cat"].get(ph, {}).values()), nb)
            mk = per_step(sum(mi["phase_cat"].get(ph, {}).values()), nm)
            print(f"{'  kernel':<16} {bk:>8.3f} {mk:>8.3f} {mk/bk if bk else 0:>7.2f}x")

    for wl in ["8k", "70k"]:
        for ph in ["target_verify", "draft_extend"]:
            print()
            print(f"{wl.upper()} — Top kernels/step stream-focused `{ph}`")
            print(f"{'#':<3} {'B200':>8} {'MI355':>8} {'MI/B200':>8}  kernel")
            print("-" * 86)
            b200, mi = results.get((wl, "b200")), results.get((wl, "mi355"))
            if not b200 or not mi:
                continue
            nb = len(b200["phase_durs"].get(ph, []))
            nm = len(mi["phase_durs"].get(ph, []))
            bk = b200["phase_kernel"].get(ph, {})
            mk = mi["phase_kernel"].get(ph, {})
            names = set(bk) | set(mk)
            ranked = sorted(names, key=lambda x: bk.get(x, 0) / max(nb, 1) + mk.get(x, 0) / max(nm, 1), reverse=True)
            for i, name in enumerate(ranked[:10], 1):
                b = bk.get(name, 0) / nb if nb else 0
                m = mk.get(name, 0) / nm if nm else 0
                ratio = f"{m/b:.2f}x" if b > 0.001 else "n/a"
                print(f"{i:<3} {b:>8.3f} {m:>8.3f} {ratio:>8}  {shorten(name)}")


if __name__ == "__main__":
    main()
