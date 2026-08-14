import gzip
import glob
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

CONFIG = {
    ("8k", "mi355"): ("glm_mtp_glm_mi355_i8192_c4-*-TP-0-DECODE.trace.json.gz", 8),
    ("8k", "b200"): ("glm_mtp_glm_b200_i8192_c4-*-TP-0-DECODE.trace.json.gz", 132),
    ("70k", "mi355"): ("glm_mtp_glm_mi355_i70000_c4-*-TP-0-EXTEND.trace.json.gz", 8),
    ("70k", "b200"): ("glm_mtp_glm_b200_i70000_c4-*-TP-0-EXTEND.trace.json.gz", 132),
}

PHASES = ["draft", "target_verify", "draft_extend"]

CATEGORIES = {
    "attention": ["fmha", "mqa_logits", "sparse_mla", "flash_attn", "deep_gemm::sm100_mqa"],
    "moe_gemm": ["gemm1_a4w4", "gemm2_a4w4", "kernel_moe_gemm", "moe_gemm", "cijk_"],
    "dense_gemm": ["bmm_", "nvjet", "hgemm", "fused_a_gemm", "cublaslt", "batched_gemm"],
    "comm": ["allreduce", "reduce_scatter", "nccl", "mnnvl_allreduce", "quickreduce", "allgather"],
    "moe_route": ["topk_transform", "routing", "moe::dev::routing", "sort_quant", "grouped_topk"],
    "norm_rope": ["rmsnorm", "norm", "rope", "hadamard", "fused_qk"],
    "quant": ["quant", "fp4", "fp8", "mxfp", "e2m1", "e4m3", "ropequantize"],
    "kv_cache": ["cache", "paged", "kv", "indexer", "cache_locs"],
    "verify_misc": ["verify_tree", "softmax", "elementwise", "vectorized", "reduce_kernel"],
    "other": [],
}


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


def categorize(name):
    n = name.lower()
    for cat, keys in CATEGORIES.items():
        if cat == "other":
            continue
        if any(k in n for k in keys):
            return cat
    return "other"


def shorten(name, n=60):
    name = re.sub(r"\s+", " ", name.strip())
    return name if len(name) <= n else name[: n - 3] + "..."


def kernel_group(name):
    n = name.lower()
    rules = [
        ("sparse_mla", "_sparse_mla_fwd"),
        ("gemm1_a4w4", "gemm1_a4w4"),
        ("gemm2_a4w4", "gemm2_a4w4"),
        ("nvfp4_bmm", "bmm_e2m1"),
        ("nvjet", "nvjet"),
        ("fmha", "fmhasm100f"),
        ("nccl_ar", "nccldevkernel_allreduce"),
        ("flashinfer_ar", "mnnvl_allreduce"),
        ("aiter_rs", "reduce_scatter_cross_device"),
        ("aiter_ar", "cross_device_reduce"),
        ("topk_decode", "topk_transform_decode"),
        ("topk_prefill", "topk_transform_prefill"),
        ("ck_moe_gemm", "kernel_moe_gemm"),
        ("hgemm", "hgemm_bf16"),
        ("deep_gemm_mqa", "deep_gemm::sm100_mqa"),
        ("moe_routing", "routingindicesdynblock"),
    ]
    for g, k in rules:
        if k in n:
            return g
    return shorten(name, 40)


def analyze(path, stream_tid):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    phases = []
    phase_durs = defaultdict(list)
    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if e.get("cat") != "gpu_user_annotation" or e.get("tid") != stream_tid:
            continue
        raw = e.get("name", "")
        if raw == "draft":
            pname = "draft"
        elif raw == "draft_extend":
            pname = "draft_extend"
        elif "TARGET_VERIFY" in raw:
            pname = "target_verify"
        else:
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
    phase_group = defaultdict(lambda: defaultdict(float))
    for p in phases:
        ph = p["phase"]
        for k in kernels:
            os_ = max(k["start"], p["start"])
            oe = min(k["end"], p["end"])
            if oe <= os_:
                continue
            frac = (oe - os_) / (k["end"] - k["start"])
            part = k["dur_ms"] * frac
            phase_kernel[ph][k["name"]] += part
            phase_cat[ph][categorize(k["name"])] += part
            phase_group[ph][kernel_group(k["name"])] += part

    return {
        "stream_tid": stream_tid,
        "phase_durs": dict(phase_durs),
        "phase_kernel": phase_kernel,
        "phase_cat": phase_cat,
        "phase_group": phase_group,
    }


def n_steps(r, ph):
    return len(r["phase_durs"].get(ph, []))


def k_per_step(r, ph, bucket, key):
    n = n_steps(r, ph)
    return r[bucket].get(ph, {}).get(key, 0) / n if n else 0.0


def phase_kernel_total(r, ph):
    return sum(r["phase_kernel"].get(ph, {}).values())


def main():
    results = {}
    for key, (pat, tid) in CONFIG.items():
        path = find_file(pat)
        if not path:
            continue
        results[key] = analyze(path, tid)
        print(f"Loaded {key[0]} {key[1]} stream={tid} steps tv={n_steps(results[key],'target_verify')}")

    print("\n" + "=" * 90)
    print("GPU stream kernel composition (MI355 tid=8, B200 tid=132)")
    print("gpu_user_annotation phases + kernel on same stream")
    print("=" * 90)

    for wl in ["8k", "70k"]:
        print(f"\n{'#' * 90}")
        print(f"# {wl.upper()} conc=4")
        print(f"{'#' * 90}")

        for ph in PHASES:
            print(f"\n--- {ph} ---")
            print(f"{'category':<14} {'B200 k/step':>11} {'MI355 k/step':>12} {'B200 %':>8} {'MI355 %':>8} {'MI/B200':>8}")
            print("-" * 68)
            b200, mi = results[(wl, "b200")], results[(wl, "mi355")]
            nb, nm = n_steps(b200, ph), n_steps(mi, ph)
            bt = phase_kernel_total(b200, ph) / nb if nb else 0
            mt = phase_kernel_total(mi, ph) / nm if nm else 0
            cats = sorted(set(b200["phase_cat"].get(ph, {})) | set(mi["phase_cat"].get(ph, {})))
            for c in cats:
                bv = k_per_step(b200, ph, "phase_cat", c)
                mv = k_per_step(mi, ph, "phase_cat", c)
                if bv < 0.005 and mv < 0.005:
                    continue
                bp = bv / bt * 100 if bt else 0
                mp = mv / mt * 100 if mt else 0
                ratio = mv / bv if bv > 0.001 else float("inf")
                print(f"{c:<14} {bv:>11.3f} {mv:>12.3f} {bp:>7.1f}% {mp:>7.1f}% {ratio:>7.2f}x")
            print(f"{'TOTAL kernel':<14} {bt:>11.3f} {mt:>12.3f}")
            bp = med_phase(b200, ph)
            mp = med_phase(mi, ph)
            print(f"{'phase marker':<14} {bp:>11.3f} {mp:>12.3f}  (non-kernel gap: {bp-bt:.2f} / {mp-mt:.2f} ms)")

        print(f"\n--- target_verify functional groups (ms/step) ---")
        print(f"{'group':<18} {'8k B200':>9} {'8k MI355':>9} {'70k B200':>9} {'70k MI355':>9}")
        print("-" * 58)
        groups = set()
        for wl in ["8k", "70k"]:
            for gpu in ["b200", "mi355"]:
                groups |= set(results[(wl, gpu)]["phase_group"].get("target_verify", {}))
        ranked = sorted(
            groups,
            key=lambda g: sum(k_per_step(results[(wl, gpu)], "target_verify", "phase_group", g) for wl in ["8k", "70k"] for gpu in ["b200", "mi355"]),
            reverse=True,
        )
        for g in ranked[:18]:
            vals = [k_per_step(results[(wl, gpu)], "target_verify", "phase_group", g) for wl in ["8k", "70k"] for gpu in ["b200", "mi355"]]
            if max(vals) < 0.02:
                continue
            print(f"{g:<18} {vals[0]:>9.3f} {vals[1]:>9.3f} {vals[2]:>9.3f} {vals[3]:>9.3f}")

        print(f"\n--- draft_extend functional groups (ms/step) ---")
        print(f"{'group':<18} {'8k B200':>9} {'8k MI355':>9} {'70k B200':>9} {'70k MI355':>9}")
        print("-" * 58)
        groups = set()
        for wl in ["8k", "70k"]:
            for gpu in ["b200", "mi355"]:
                groups |= set(results[(wl, gpu)]["phase_group"].get("draft_extend", {}))
        ranked = sorted(
            groups,
            key=lambda g: sum(k_per_step(results[(wl, gpu)], "draft_extend", "phase_group", g) for wl in ["8k", "70k"] for gpu in ["b200", "mi355"]),
            reverse=True,
        )
        for g in ranked[:15]:
            vals = [k_per_step(results[(wl, gpu)], "draft_extend", "phase_group", g) for wl in ["8k", "70k"] for gpu in ["b200", "mi355"]]
            if max(vals) < 0.02:
                continue
            print(f"{g:<18} {vals[0]:>9.3f} {vals[1]:>9.3f} {vals[2]:>9.3f} {vals[3]:>9.3f}")

    print("\n" + "=" * 90)
    print("Full decode step kernel budget (sum of 3 phases, ms/step)")
    print("=" * 90)
    print(f"{'wl':<4} {'GPU':<6} {'draft':>7} {'tv':>7} {'de':>7} {'kernel_sum':>10} {'phase_sum':>10} {'MI355/B200':>10}")
    print("-" * 68)
    for wl in ["8k", "70k"]:
        b200, mi = results[(wl, "b200")], results[(wl, "mi355")]
        bk = sum(phase_kernel_total(b200, p) / max(n_steps(b200, p), 1) for p in PHASES)
        mk = sum(phase_kernel_total(mi, p) / max(n_steps(mi, p), 1) for p in PHASES)
        bp = sum(statistics.median(b200["phase_durs"].get(p, [0])) for p in PHASES)
        mp = sum(statistics.median(mi["phase_durs"].get(p, [0])) for p in PHASES)
        bd = {p: phase_kernel_total(b200, p) / max(n_steps(b200, p), 1) for p in PHASES}
        md = {p: phase_kernel_total(mi, p) / max(n_steps(mi, p), 1) for p in PHASES}
        print(
            f"{wl:<4} {'b200':<6} {bd['draft']:>7.3f} {bd['target_verify']:>7.3f} {bd['draft_extend']:>7.3f} {bk:>10.3f} {bp:>10.3f}"
        )
        print(
            f"{'':4} {'mi355':<6} {md['draft']:>7.3f} {md['target_verify']:>7.3f} {md['draft_extend']:>7.3f} {mk:>10.3f} {mp:>10.3f} {mk/bk if bk else 0:>9.2f}x"
        )


def med_phase(r, ph):
    d = r["phase_durs"].get(ph, [])
    return statistics.median(d) if d else 0.0


if __name__ == "__main__":
    main()
