"""B200 multistream decode analysis — dry run: 8k conc=4 DECODE trace.

Explains missing moe_gemm on single-stream view and maps kernels across all GPU streams.
"""
import gzip
import glob
import json
import os
import re
import statistics
from collections import defaultdict

OUTLIER_MS = 100
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT8 = os.path.join(os.path.dirname(ROOT), "2_8k_profile_MTP")
OUT_REPORT = os.path.join(ROOT, "b200_8k_c4_multistream_report.txt")
OUT_TXT = os.path.join(ROOT, "b200_8k_c4_multistream_by_stream.txt")

PATTERN = "glm_mtp_glm_b200_i8192_c4-*-TP-0-DECODE.trace.json.gz"
PHASE_STREAM = 132  # stream used in prior single-stream analysis

CATEGORIES = {
    "attention": ["fmha", "mqa_logits", "sparse_mla", "flash_attn", "deep_gemm::sm100_mqa"],
    "moe_gemm": ["gemm1_a4w4", "gemm2_a4w4", "kernel_moe_gemm", "moe_gemm", "cijk_"],
    "moe_gemm_nv": ["bmm_e2m1", "nvjet", "nvfp4", "cutlass.*moe", "grouped_gemm"],
    "dense_gemm": ["bmm_", "hgemm", "fused_a_gemm", "cublaslt", "batched_gemm"],
    "comm": ["allreduce", "reduce_scatter", "nccl", "mnnvl_allreduce", "quickreduce", "allgather"],
    "moe_route": ["topk_transform", "routing", "moe::dev::routing", "sort_quant", "grouped_topk"],
    "norm_rope": ["rmsnorm", "norm", "rope", "hadamard", "fused_qk"],
    "quant": ["quant", "fp4", "fp8", "mxfp", "e2m1", "e4m3", "ropequantize"],
    "kv_cache": ["cache", "paged", "kv", "indexer", "cache_locs"],
    "verify_misc": ["verify_tree", "softmax", "elementwise", "vectorized", "reduce_kernel"],
}


def find_file():
    best = None
    for path in glob.glob(os.path.join(ROOT8, "**", PATTERN), recursive=True):
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
        if cat in ("other",):
            continue
        if any(re.search(k, n) if ".*" in k else k in n for k in keys):
            return cat
    return "other"


def shorten(name, n=55):
    name = re.sub(r"\s+", " ", name.strip())
    return name if len(name) <= n else name[: n - 3] + "..."


def phase_name(raw):
    if raw == "draft":
        return "draft"
    if raw == "draft_extend":
        return "draft_extend"
    if "TARGET_VERIFY" in raw:
        return "target_verify"
    return None


def overlap_frac(k, p):
    os_ = max(k["start"], p["start"])
    oe = min(k["end"], p["end"])
    if oe <= os_:
        return 0.0
    return (oe - os_) / (k["end"] - k["start"])


def load_trace(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    phases = []
    ann_streams = defaultdict(int)
    kernel_streams = defaultdict(int)

    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        tid = e.get("tid")
        if cat == "gpu_user_annotation":
            pname = phase_name(e.get("name", ""))
            if pname is None:
                continue
            dur = e.get("dur", 0) / 1000
            if dur > OUTLIER_MS:
                continue
            ann_streams[tid] += 1
            phases.append(
                {
                    "phase": pname,
                    "tid": tid,
                    "start": e["ts"],
                    "end": e["ts"] + e.get("dur", 0),
                    "dur_ms": dur,
                }
            )

    kernels = []
    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0)
        if dur <= 0:
            continue
        tid = e.get("tid")
        kernel_streams[tid] += 1
        kernels.append(
            {
                "name": e.get("name", ""),
                "tid": tid,
                "start": e["ts"],
                "end": e["ts"] + dur,
                "dur_ms": dur / 1000,
            }
        )

    return ev, phases, kernels, ann_streams, kernel_streams


def analyze_multistream(phases, kernels, phase_filter=None):
    """Attribute kernel time to (phase, stream) using phase windows on PHASE_STREAM."""
    phase_windows = [p for p in phases if p["tid"] == PHASE_STREAM]
    if phase_filter:
        phase_windows = [p for p in phase_windows if p["phase"] == phase_filter]

    # per (phase, stream_tid): category -> ms total
    bucket = defaultdict(lambda: defaultdict(float))
    kernel_bucket = defaultdict(lambda: defaultdict(float))

    for p in phase_windows:
        ph = p["phase"]
        for k in kernels:
            frac = overlap_frac(k, p)
            if frac <= 0:
                continue
            part = k["dur_ms"] * frac
            cat = categorize(k["name"])
            bucket[(ph, k["tid"])][cat] += part
            kernel_bucket[(ph, k["tid"])][k["name"]] += part

    return bucket, kernel_bucket, phase_windows


def n_steps(phases, ph, tid=PHASE_STREAM):
    return sum(1 for p in phases if p["phase"] == ph and p["tid"] == tid)


def med_phase(phases, ph, tid=PHASE_STREAM):
    d = [p["dur_ms"] for p in phases if p["phase"] == ph and p["tid"] == tid]
    return statistics.median(d) if d else 0.0


def main():
    path = find_file()
    if not path:
        raise SystemExit("B200 8k c4 DECODE trace not found")

    print(f"Loading {path} ...")
    _, phases, kernels, ann_streams, kernel_streams = load_trace(path)
    n_kern = len(kernels)
    print(f"Kernels: {n_kern:,}, phase markers on streams: {dict(sorted(ann_streams.items()))}")

    lines = []
    w = lines.append
    w("=" * 90)
    w("B200 8k conc=4 DECODE — Multistream Analysis (dry run)")
    w(f"Trace: {os.path.basename(path)}")
    w(f"Phase markers read from gpu_user_annotation stream tid={PHASE_STREAM}")
    w("Kernels attributed to phase windows by time overlap, grouped by executing stream tid")
    w("=" * 90)

    # --- Why no moe_gemm on single stream ---
    w("\n## Q: Why no moe_gemm on stream 132 target_verify?")
    w("")
    w("1) Category naming: B200 MoE uses NVFP4 BMM / NVJet kernels (bmm_e2m1, nvjet_*),")
    w("   which match dense_gemm / moe_gemm_nv rules, NOT gemm1_a4w4/gemm2_a4w4 (MI355).")
    w("   Prior Excel 'moe_gemm' column only keys on MI355-style kernel names.")
    w("")
    w("2) Multistream: MoE/comm kernels may execute on streams != 132 while phase marker")
    w("   stays on stream 132. Single-stream view misses off-stream work.")

    # stream inventory during target_verify
    bucket, kernel_bucket, tv_windows = analyze_multistream(phases, kernels, "target_verify")
    steps = len(tv_windows)
    w(f"\n## target_verify: {steps} steps on stream {PHASE_STREAM}")

    # total kernel ms per stream during TV
    stream_totals = defaultdict(float)
    for (ph, tid), cats in bucket.items():
        stream_totals[tid] += sum(cats.values())

    w("\n--- Kernel time by stream during target_verify (total ms, all steps) ---")
    w(f"{'stream':>8} {'total_ms':>12} {'ms/step':>10} {'share%':>8}")
    grand = sum(stream_totals.values())
    for tid in sorted(stream_totals, key=stream_totals.get, reverse=True):
        tot = stream_totals[tid]
        w(f"{tid:>8} {tot:>12.1f} {tot/steps:>10.3f} {tot/grand*100:>7.1f}%")

    # category by stream (ms/step)
    w("\n--- Category ms/step by stream (target_verify) ---")
    streams = sorted(stream_totals, key=stream_totals.get, reverse=True)
    cats_seen = sorted({c for (ph, tid), cs in bucket.items() for c in cs})
    hdr = f"{'category':<14}" + "".join(f"{('s'+str(t))[-6:]:>10}" for t in streams[:8])
    w(hdr)
    w("-" * len(hdr))
    for cat in cats_seen:
        row = f"{cat:<14}"
        for tid in streams[:8]:
            v = bucket[("target_verify", tid)].get(cat, 0) / steps
            row += f"{v:>10.3f}"
        w(row)

    # NV MoE-like vs legacy moe_gemm on stream 132 vs all streams
    w("\n--- MoE-related kernel time (target_verify, ms/step) ---")
    def sum_cat_ms(step_tid=None, cat_keys=None):
        total = 0
        for (ph, tid), cats in bucket.items():
            if ph != "target_verify":
                continue
            if step_tid is not None and tid != step_tid:
                continue
            for c, v in cats.items():
                if cat_keys and c not in cat_keys:
                    continue
                total += v
        return total / steps

    w(f"  stream {PHASE_STREAM}  moe_gemm (MI355 keys):     {sum_cat_ms(PHASE_STREAM, ['moe_gemm']):.3f}")
    w(f"  stream {PHASE_STREAM}  moe_gemm_nv (B200 MoE):    {sum_cat_ms(PHASE_STREAM, ['moe_gemm_nv']):.3f}")
    w(f"  stream {PHASE_STREAM}  dense_gemm (rest):         {sum_cat_ms(PHASE_STREAM, ['dense_gemm']):.3f}")
    w(f"  ALL streams          moe_gemm (MI355 keys):     {sum_cat_ms(None, ['moe_gemm']):.3f}")
    w(f"  ALL streams          moe_gemm_nv (B200 MoE):    {sum_cat_ms(None, ['moe_gemm_nv']):.3f}")
    w(f"  ALL streams          dense_gemm:                  {sum_cat_ms(None, ['dense_gemm']):.3f}")

    # top kernels per top streams
    stream_lines = []
    sw = stream_lines.append
    sw("=" * 90)
    sw("Top kernels per stream — target_verify (ms/step)")
    sw("=" * 90)
    for tid in streams[:6]:
        ktot = kernel_bucket[("target_verify", tid)]
        if not ktot:
            continue
        sw(f"\n### stream tid={tid}  total={stream_totals[tid]/steps:.3f} ms/step")
        ranked = sorted(ktot.items(), key=lambda x: x[1], reverse=True)[:12]
        for name, ms in ranked:
            sw(f"  {ms/steps:7.3f} ms/step  [{categorize(name)}]  {shorten(name)}")

    # all phases aggregated view (all streams)
    bucket_all, _, _ = analyze_multistream(phases, kernels)
    w("\n## All phases — all streams kernel sum (ms/step)")
    for ph in ["draft", "target_verify", "draft_extend"]:
        ns = n_steps(phases, ph)
        if not ns:
            continue
        w(f"\n### {ph} ({ns} steps)")
        by_stream = defaultdict(float)
        by_cat_all = defaultdict(float)
        for (pname, tid), cats in bucket_all.items():
            if pname != ph:
                continue
            by_stream[tid] += sum(cats.values())
            for c, v in cats.items():
                by_cat_all[c] += v
        tot = sum(by_stream.values())
        w(f"  phase median (stream {PHASE_STREAM}): {med_phase(phases, ph):.3f} ms")
        w(f"  kernel sum all streams: {tot/ns:.3f} ms/step")
        w(f"  kernel sum stream {PHASE_STREAM} only: {by_stream.get(PHASE_STREAM,0)/ns:.3f} ms/step")
        w(f"  off-stream kernel: {(tot-by_stream.get(PHASE_STREAM,0))/ns:.3f} ms/step")
        w("  category (all streams):")
        for c in sorted(by_cat_all, key=by_cat_all.get, reverse=True):
            w(f"    {c:<14} {by_cat_all[c]/ns:.3f} ms/step")

    report = "\n".join(lines)
    stream_report = "\n".join(stream_lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write("\n\n")
        fh.write(stream_report)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(stream_report)

    print(report)
    print(f"\nWrote {OUT_REPORT}")
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
