"""Export 8k conc=4 DECODE trace pair (B200 + MI355) to Excel (EN + ZH).

Includes single-stream view and B200 all-stream (multistream) fair comparison.
"""
import gzip
import json
import os
import re
import statistics
from collections import defaultdict

import pandas as pd

OUTLIER_MS = 100
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRACES = {
    "b200": os.path.join(
        OUT_DIR,
        "glm_mtp_glm_b200_i8192_c4-1785920750.2769158-TP-0-DECODE.trace.json.gz",
    ),
    "mi355": os.path.join(
        OUT_DIR,
        "glm_mtp_glm_mi355_i8192_c4-1786058894.8985496-TP-0-DECODE.trace.json.gz",
    ),
}
PHASE_STREAM = {"b200": 132, "mi355": 8}
PHASES = ["draft", "target_verify", "draft_extend"]

# B200 nvjet/bmm checked before generic bmm_ / dense_gemm
CATEGORIES = {
    "attention": ["fmha", "mqa_logits", "sparse_mla", "flash_attn", "deep_gemm::sm100_mqa"],
    "moe_gemm": ["gemm1_a4w4", "gemm2_a4w4", "kernel_moe_gemm", "moe_gemm", "cijk_"],
    "moe_gemm_nv": ["bmm_e2m1", "nvjet", "nvfp4"],
    "dense_gemm": ["bmm_", "hgemm", "fused_a_gemm", "cublaslt", "batched_gemm"],
    "comm": ["allreduce", "reduce_scatter", "nccl", "mnnvl_allreduce", "quickreduce", "allgather"],
    "moe_route": ["topk_transform", "routing", "moe::dev::routing", "sort_quant", "grouped_topk"],
    "norm_rope": ["rmsnorm", "norm", "rope", "hadamard", "fused_qk"],
    "quant": ["quant", "fp8", "mxfp", "e2m1", "e4m3", "ropequantize"],
    "kv_cache": ["cache", "paged", "kv", "indexer", "cache_locs"],
    "verify_misc": ["verify_tree", "softmax", "elementwise", "vectorized", "reduce_kernel"],
    "other": [],
}

CAT_ORDER = [
    "attention", "moe_gemm", "moe_gemm_nv", "dense_gemm", "comm",
    "moe_route", "norm_rope", "quant", "kv_cache", "verify_misc", "other",
]

KERNEL_GROUPS = [
    ("sparse_mla", "_sparse_mla_fwd", "Sparse MLA Attention"),
    ("gemm1_a4w4", "gemm1_a4w4", "MoE GEMM1 a4w4 (MI355)"),
    ("gemm2_a4w4", "gemm2_a4w4", "MoE GEMM2 a4w4 (MI355)"),
    ("nvfp4_bmm", "bmm_e2m1", "NVFP4 BMM E2M1 (B200)"),
    ("nvjet", "nvjet", "NVJet GEMM (B200)"),
    ("fmha", "fmhasm100f", "FMHA Attention (B200)"),
    ("flashinfer_ar", "mnnvl_allreduce", "FlashInfer AllReduce"),
    ("aiter_rs", "reduce_scatter_cross_device", "Aiter ReduceScatter"),
    ("hgemm", "hgemm_bf16", "HGEMM BF16"),
    ("deep_gemm_mqa", "deep_gemm", "DeepGEMM MQA/Gluon"),
    ("moe_routing", "routingindicesdynblock", "MoE Routing"),
    ("rmsnorm", "local_device_load_rmsnorm", "RMSNorm"),
    ("fused_a_gemm", "fused_a_gemm_kernel", "Fused A GEMM"),
    ("ck_moe_gemm", "kernel_moe_gemm", "CK MoE GEMM"),
]

META = {
    "en": {
        "file": "GLM_MTP_8k_DECODE_c4_EN.xlsx",
        "readme": "README",
        "summary": "Summary",
        "phase": "Phase_per_step",
        "fair": "Fair_compare",
        "detail": "Category_detail",
        "moe_detail": "MoE_breakdown",
        "tv_kern": "TV_top_kernels",
        "b200_ms": "B200_multistream",
        "ratio": "MI355_vs_B200",
    },
    "zh": {
        "file": "GLM_MTP_8k_DECODE_c4_ZH.xlsx",
        "readme": "说明",
        "summary": "总览",
        "phase": "Phase耗时",
        "fair": "功能对比",
        "detail": "分类明细",
        "moe_detail": "MoE细分",
        "tv_kern": "TV核心Kernel",
        "b200_ms": "B200多Stream",
        "ratio": "MI355对比B200",
    },
}

CAT_LABELS = {
    "en": {
        "attention": "attention", "moe_gemm": "moe_gemm (MI355 detail)",
        "moe_gemm_nv": "moe_gemm_nv (B200 detail)",
        "dense_gemm": "dense_gemm", "comm": "comm", "moe_route": "moe_route",
        "norm_rope": "norm_rope", "quant": "quant", "kv_cache": "kv_cache",
        "verify_misc": "verify_misc", "other": "other",
    },
    "zh": {
        "attention": "Attention", "moe_gemm": "MoE GEMM (MI355 明细)",
        "moe_gemm_nv": "MoE GEMM NV (B200 明细)",
        "dense_gemm": "Dense GEMM", "comm": "通信 Comm", "moe_route": "MoE 路由",
        "norm_rope": "Norm/RoPE", "quant": "量化 Quant", "kv_cache": "KV Cache",
        "verify_misc": "Verify/杂项", "other": "其他",
    },
}

FUNCTIONAL = [
    {
        "key": "moe_gemm_total",
        "en": "MoE GEMM (total)",
        "zh": "MoE GEMM (功能合计)",
        "b200_cats": ["moe_gemm_nv"],
        "mi355_cats": ["moe_gemm"],
        "b200_map": "bmm_e2m1 + nvjet (all-stream)",
        "mi355_map": "gemm1_a4w4 + gemm2_a4w4",
    },
    {
        "key": "attention",
        "en": "Attention (total)",
        "zh": "Attention (功能合计)",
        "b200_cats": ["attention"],
        "mi355_cats": ["attention"],
        "b200_map": "fmha / deep_gemm MQA",
        "mi355_map": "sparse_mla / mqa_logits",
    },
    {
        "key": "dense_gemm",
        "en": "Dense GEMM (non-MoE)",
        "zh": "Dense GEMM (非MoE)",
        "b200_cats": ["dense_gemm"],
        "mi355_cats": ["dense_gemm"],
        "b200_map": "fused_a_gemm, hgemm, ...",
        "mi355_map": "hgemm, batched_gemm, ...",
    },
    {
        "key": "comm",
        "en": "Comm (total)",
        "zh": "通信 (功能合计)",
        "b200_cats": ["comm"],
        "mi355_cats": ["comm"],
        "b200_map": "flashinfer AR, nccl",
        "mi355_map": "aiter RS/AR",
    },
    {
        "key": "moe_route",
        "en": "MoE routing (total)",
        "zh": "MoE 路由 (功能合计)",
        "b200_cats": ["moe_route"],
        "mi355_cats": ["moe_route"],
        "b200_map": "topk, routing",
        "mi355_map": "topk, grouped_topk",
    },
    {
        "key": "norm_rope",
        "en": "Norm/RoPE (total)",
        "zh": "Norm/RoPE (功能合计)",
        "b200_cats": ["norm_rope"],
        "mi355_cats": ["norm_rope"],
        "b200_map": "rmsnorm, rope",
        "mi355_map": "aiter rmsnorm, rope",
    },
    {
        "key": "quant",
        "en": "Quant (total)",
        "zh": "量化 (功能合计)",
        "b200_cats": ["quant"],
        "mi355_cats": ["quant"],
        "b200_map": "fp4 quant kernels",
        "mi355_map": "mxfp quant kernels",
    },
    {
        "key": "kv_cache",
        "en": "KV cache (total)",
        "zh": "KV Cache (功能合计)",
        "b200_cats": ["kv_cache"],
        "mi355_cats": ["kv_cache"],
        "b200_map": "cache/indexer",
        "mi355_map": "cache/indexer",
    },
    {
        "key": "verify_misc",
        "en": "Verify/misc (total)",
        "zh": "Verify/杂项 (功能合计)",
        "b200_cats": ["verify_misc"],
        "mi355_cats": ["verify_misc"],
        "b200_map": "verify, elementwise",
        "mi355_map": "verify, elementwise",
    },
    {
        "key": "other",
        "en": "Other",
        "zh": "其他",
        "b200_cats": ["other"],
        "mi355_cats": ["other"],
        "b200_map": "-",
        "mi355_map": "-",
    },
]


def categorize(name):
    n = name.lower()
    for cat in CAT_ORDER:
        if cat == "other":
            continue
        for k in CATEGORIES[cat]:
            if k in n:
                return cat
    return "other"


def kernel_group(name):
    n = name.lower()
    for gid, key, _ in KERNEL_GROUPS:
        if key in n:
            return gid
    s = re.sub(r"\s+", " ", name.strip())
    return s[:45] + ("..." if len(s) > 45 else "")


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


def round3(x):
    return round(x, 3) if x is not None else None


def round2pct(num, den):
    return round(num / den * 100, 1) if den else None


def load_trace(path):
    print(f"Loading {os.path.basename(path)} ...")
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    phases = []
    phase_durs = defaultdict(list)
    ann_streams = defaultdict(int)

    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if e.get("cat") != "gpu_user_annotation":
            continue
        pname = phase_name(e.get("name", ""))
        if pname is None:
            continue
        dur = e.get("dur", 0) / 1000
        if dur > OUTLIER_MS:
            continue
        tid = e.get("tid")
        ann_streams[tid] += 1
        rec = {
            "phase": pname,
            "tid": tid,
            "start": e["ts"],
            "end": e["ts"] + e.get("dur", 0),
            "dur_ms": dur,
        }
        phases.append(rec)
        if tid == PHASE_STREAM.get("b200") or tid == PHASE_STREAM.get("mi355"):
            phase_durs[pname].append(dur)

    kernels = []
    kernel_streams = defaultdict(int)
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

    return {
        "path": os.path.basename(path),
        "phases": phases,
        "phase_durs": dict(phase_durs),
        "kernels": kernels,
        "ann_streams": dict(ann_streams),
        "kernel_streams": dict(kernel_streams),
    }


def attribute_kernels(phases, kernels, phase_stream, all_streams=False):
    """Attribute kernel ms to phases. all_streams=True uses kernels from any tid."""
    windows = [p for p in phases if p["tid"] == phase_stream]
    phase_kernel = defaultdict(lambda: defaultdict(float))
    phase_cat = defaultdict(lambda: defaultdict(float))
    phase_group = defaultdict(lambda: defaultdict(float))
    stream_cat = defaultdict(lambda: defaultdict(float))  # (phase, stream) -> cat

    for p in windows:
        ph = p["phase"]
        for k in kernels:
            if not all_streams and k["tid"] != phase_stream:
                continue
            frac = overlap_frac(k, p)
            if frac <= 0:
                continue
            part = k["dur_ms"] * frac
            phase_kernel[ph][k["name"]] += part
            cat = categorize(k["name"])
            phase_cat[ph][cat] += part
            phase_group[ph][kernel_group(k["name"])] += part
            if all_streams:
                stream_cat[(ph, k["tid"])][cat] += part

    return {
        "phase_kernel": phase_kernel,
        "phase_cat": phase_cat,
        "phase_group": phase_group,
        "stream_cat": stream_cat,
        "phase_durs": defaultdict(list, {ph: [p["dur_ms"] for p in windows if p["phase"] == ph] for ph in PHASES}),
    }


def n_steps(r, ph):
    return len(r["phase_durs"].get(ph, []))


def k_per_step(r, ph, bucket, key):
    n = n_steps(r, ph)
    return r[bucket].get(ph, {}).get(key, 0) / n if n else 0.0


def phase_kernel_total(r, ph):
    return sum(r["phase_kernel"].get(ph, {}).values())


def med_phase(r, ph):
    d = r["phase_durs"].get(ph, [])
    return statistics.median(d) if d else 0.0


def kernels_in_category(r, ph, cat, min_ms=0.005):
    return kernels_in_categories(r, ph, [cat], min_ms)


def kernels_in_categories(r, ph, cats, min_ms=0.005):
    n = n_steps(r, ph)
    if not n:
        return ""
    items = []
    for kname, total_ms in r["phase_kernel"].get(ph, {}).items():
        if categorize(kname) not in cats:
            continue
        per_step = total_ms / n
        if per_step < min_ms:
            continue
        items.append((per_step, kname))
    items.sort(key=lambda x: x[0], reverse=True)
    return "; ".join(f"{name} ({round3(ms)})" for ms, name in items)


def sum_cats_per_step(r, ph, cats):
    return sum(k_per_step(r, ph, "phase_cat", c) for c in cats)


def analyze_gpu(gpu, raw):
    ps = PHASE_STREAM[gpu]
    single = attribute_kernels(raw["phases"], raw["kernels"], ps, all_streams=False)
    single["stream_tid"] = ps
    single["path"] = raw["path"]
    single["ann_streams"] = raw["ann_streams"]
    if gpu == "b200":
        multi = attribute_kernels(raw["phases"], raw["kernels"], ps, all_streams=True)
        single["stream_cat"] = multi["stream_cat"]
    return single


def build_readme(lang):
    m = META[lang]
    lines = [
        "GLM MTP 8k DECODE kernel analysis (8192/1024, conc=4)" if lang == "en"
        else "GLM MTP 8k DECODE kernel 分析 (8192/1024, conc=4)",
        f"B200: {os.path.basename(TRACES['b200'])}",
        f"MI355: {os.path.basename(TRACES['mi355'])}",
        "Phase markers: gpu_user_annotation on MI355 tid=8, B200 tid=132",
        "Fair_compare: one functional row per category; B200 all-stream vs MI355 single-stream",
        "  MoE GEMM (total): B200 bmm_e2m1+nvjet | MI355 gemm1+gemm2 a4w4",
        "Category_detail / MoE_breakdown: platform-specific names for kernel optimization",
        "Unit: ms/step | MI355: MXFP4 | B200: NVFP4 | TP=4",
    ]
    col = "Item" if lang == "en" else "项目"
    return pd.DataFrame([{col: x} for x in lines])


def build_summary(b200, mi355, b200_all, lang):
    rows = []
    for label, b, note in [
        ("B200 single-stream", b200, f"stream {b200['stream_tid']}"),
        ("B200 all-stream", b200_all, "multistream fair"),
        ("MI355 single-stream", mi355, f"stream {mi355['stream_tid']}"),
    ]:
        kd = {p: round3(phase_kernel_total(b, p) / max(n_steps(b, p), 1)) for p in PHASES}
        pd_ = {p: round3(med_phase(b, p)) for p in PHASES}
        rows.append({
            "View" if lang == "en" else "视图": label,
            "Note" if lang == "en" else "说明": note,
            "draft kernel": kd["draft"],
            "target_verify kernel": kd["target_verify"],
            "draft_extend kernel": kd["draft_extend"],
            "Kernel sum ms/step": round3(sum(kd.values())),
            "draft phase": pd_["draft"],
            "target_verify phase": pd_["target_verify"],
            "draft_extend phase": pd_["draft_extend"],
            "Phase sum ms/step": round3(sum(pd_.values())),
        })

    b_s = rows[0]
    b_a = rows[1]
    mi = rows[2]
    rows.append({
        "View" if lang == "en" else "视图": "MI355/B200 (single-stream)",
        "Note" if lang == "en" else "说明": "ratio",
        "draft kernel": round3(mi["draft kernel"] / b_s["draft kernel"]) if b_s["draft kernel"] else None,
        "target_verify kernel": round3(mi["target_verify kernel"] / b_s["target_verify kernel"]),
        "draft_extend kernel": round3(mi["draft_extend kernel"] / b_s["draft_extend kernel"]) if b_s["draft_extend kernel"] else None,
        "Kernel sum ms/step": round3(mi["Kernel sum ms/step"] / b_s["Kernel sum ms/step"]),
        "draft phase": round3(mi["draft phase"] / b_s["draft phase"]) if b_s["draft phase"] else None,
        "target_verify phase": round3(mi["target_verify phase"] / b_s["target_verify phase"]),
        "draft_extend phase": round3(mi["draft_extend phase"] / b_s["draft_extend phase"]) if b_s["draft_extend phase"] else None,
        "Phase sum ms/step": round3(mi["Phase sum ms/step"] / b_s["Phase sum ms/step"]),
    })
    rows.append({
        "View" if lang == "en" else "视图": "MI355/B200 (B200 all-stream)",
        "Note" if lang == "en" else "说明": "fair ratio",
        "draft kernel": round3(mi["draft kernel"] / b_a["draft kernel"]) if b_a["draft kernel"] else None,
        "target_verify kernel": round3(mi["target_verify kernel"] / b_a["target_verify kernel"]),
        "draft_extend kernel": round3(mi["draft_extend kernel"] / b_a["draft_extend kernel"]) if b_a["draft_extend kernel"] else None,
        "Kernel sum ms/step": round3(mi["Kernel sum ms/step"] / b_a["Kernel sum ms/step"]),
        "draft phase": round3(mi["draft phase"] / b_a["draft phase"]) if b_a["draft phase"] else None,
        "target_verify phase": round3(mi["target_verify phase"] / b_a["target_verify phase"]),
        "draft_extend phase": round3(mi["draft_extend phase"] / b_a["draft_extend phase"]) if b_a["draft_extend phase"] else None,
        "Phase sum ms/step": round3(mi["Phase sum ms/step"] / b_a["Phase sum ms/step"]),
    })
    return pd.DataFrame(rows)


def build_phase(b200, mi355, b200_all, lang):
    rows = []
    for ph in PHASES:
        bk_s = round3(phase_kernel_total(b200, ph) / max(n_steps(b200, ph), 1))
        bk_a = round3(phase_kernel_total(b200_all, ph) / max(n_steps(b200_all, ph), 1))
        mk = round3(phase_kernel_total(mi355, ph) / max(n_steps(mi355, ph), 1))
        bp = round3(med_phase(b200, ph))
        mp = round3(med_phase(mi355, ph))
        rows.append({
            "Phase": ph,
            "Steps": n_steps(b200, ph),
            "B200 kernel (single)": bk_s,
            "B200 kernel (all-stream)": bk_a,
            "MI355 kernel": mk,
            "B200 phase": bp,
            "MI355 phase": mp,
            "MI355/B200 single": round3(mk / bk_s) if bk_s else None,
            "MI355/B200 all-stream": round3(mk / bk_a) if bk_a else None,
            "B200 off-stream kernel": round3(bk_a - bk_s) if bk_a and bk_s else None,
        })
    return pd.DataFrame(rows)


def build_fair_compare(b200_all, mi355, lang):
    rows = []
    for ph in PHASES:
        nb, nm = n_steps(b200_all, ph), n_steps(mi355, ph)
        bt = phase_kernel_total(b200_all, ph) / nb if nb else 0
        mt = phase_kernel_total(mi355, ph) / nm if nm else 0
        rows.append({
            "Phase": ph,
            "Category": "TOTAL kernel" if lang == "en" else "Kernel 合计",
            "B200 ms/step (all-stream)": round3(bt),
            "MI355 ms/step": round3(mt),
            "MI355/B200": round3(mt / bt) if bt else None,
            "B200 %": round2pct(bt, bt),
            "MI355 %": round2pct(mt, mt),
            "B200 maps to": "all GPU streams" if lang == "en" else "所有 GPU stream",
            "MI355 maps to": f"stream {mi355['stream_tid']}",
            "B200 top kernels": "-",
            "MI355 top kernels": "-",
        })
        for fn in FUNCTIONAL:
            bv = sum_cats_per_step(b200_all, ph, fn["b200_cats"])
            mv = sum_cats_per_step(mi355, ph, fn["mi355_cats"])
            if max(bv, mv) < 0.005:
                continue
            rows.append({
                "Phase": ph,
                "Category": fn[lang],
                "B200 ms/step (all-stream)": round3(bv),
                "MI355 ms/step": round3(mv),
                "MI355/B200": round3(mv / bv) if bv > 0.001 else None,
                "B200 %": round2pct(bv, bt),
                "MI355 %": round2pct(mv, mt),
                "B200 maps to": fn["b200_map"],
                "MI355 maps to": fn["mi355_map"],
                "B200 top kernels": kernels_in_categories(b200_all, ph, fn["b200_cats"]),
                "MI355 top kernels": kernels_in_categories(mi355, ph, fn["mi355_cats"]),
            })
    return pd.DataFrame(rows)


def build_category_detail(b200, mi355, b200_all, lang):
    cl = CAT_LABELS[lang]
    rows = []
    for ph in PHASES:
        nb, nm = n_steps(b200, ph), n_steps(mi355, ph)
        bt_s = phase_kernel_total(b200, ph) / nb if nb else 0
        bt_a = phase_kernel_total(b200_all, ph) / nb if nb else 0
        mt = phase_kernel_total(mi355, ph) / nm if nm else 0
        cats = sorted(
            set(b200["phase_cat"].get(ph, {}))
            | set(b200_all["phase_cat"].get(ph, {}))
            | set(mi355["phase_cat"].get(ph, {})),
            key=lambda c: CAT_ORDER.index(c) if c in CAT_ORDER else 99,
        )
        for c in cats:
            bv_s = k_per_step(b200, ph, "phase_cat", c)
            bv_a = k_per_step(b200_all, ph, "phase_cat", c)
            mv = k_per_step(mi355, ph, "phase_cat", c)
            if max(bv_s, bv_a, mv) < 0.005:
                continue
            rows.append({
                "Phase": ph,
                "Category": cl.get(c, c),
                "Platform scope": "platform-specific detail",
                "B200 ms/step (single)": round3(bv_s),
                "B200 ms/step (all-stream)": round3(bv_a),
                "MI355 ms/step": round3(mv),
                "B200 single %": round2pct(bv_s, bt_s),
                "B200 all-stream %": round2pct(bv_a, bt_a),
                "MI355 %": round2pct(mv, mt),
                "Note": "See Fair_compare for cross-platform ratio" if lang == "en"
                else "跨平台对比请看「功能对比」",
                "B200 kernels (all-stream)": kernels_in_category(b200_all, ph, c),
                "MI355 kernels": kernels_in_category(mi355, ph, c),
            })
    return pd.DataFrame(rows)


def build_moe_breakdown(b200, mi355, b200_all, lang):
    group_desc = {gid: desc for gid, _, desc in KERNEL_GROUPS}
    moe_groups = ["gemm1_a4w4", "gemm2_a4w4", "nvfp4_bmm", "nvjet", "ck_moe_gemm"]
    rows = []
    for ph in PHASES:
        for g in moe_groups:
            bv_s = k_per_step(b200, ph, "phase_group", g)
            bv_a = k_per_step(b200_all, ph, "phase_group", g)
            mv = k_per_step(mi355, ph, "phase_group", g)
            if max(bv_s, bv_a, mv) < 0.005:
                continue
            platform = "MI355" if g in ("gemm1_a4w4", "gemm2_a4w4", "ck_moe_gemm") else "B200"
            rows.append({
                "Phase": ph,
                "Platform": platform,
                "Kernel group": g,
                "Description": group_desc.get(g, g),
                "B200 single ms/step": round3(bv_s),
                "B200 all-stream ms/step": round3(bv_a),
                "MI355 ms/step": round3(mv),
                "Rolls up to": "MoE GEMM (total)" if lang == "en" else "MoE GEMM (功能合计)",
            })
    return pd.DataFrame(rows)


def build_tv_kernels(b200, mi355, b200_all, lang):
    group_desc = {gid: desc for gid, _, desc in KERNEL_GROUPS}
    ph = "target_verify"
    groups = (
        set(b200["phase_group"].get(ph, {}))
        | set(b200_all["phase_group"].get(ph, {}))
        | set(mi355["phase_group"].get(ph, {}))
    )
    ranked = sorted(
        groups,
        key=lambda g: k_per_step(mi355, ph, "phase_group", g) + k_per_step(b200_all, ph, "phase_group", g),
        reverse=True,
    )
    rows = []
    for g in ranked:
        bv_s = k_per_step(b200, ph, "phase_group", g)
        bv_a = k_per_step(b200_all, ph, "phase_group", g)
        mv = k_per_step(mi355, ph, "phase_group", g)
        if max(bv_s, bv_a, mv) < 0.02:
            continue
        rows.append({
            "Kernel group": g,
            "Description": group_desc.get(g, g),
            "B200 single ms/step": round3(bv_s),
            "B200 all-stream ms/step": round3(bv_a),
            "MI355 ms/step": round3(mv),
            "MI355/B200 all-stream": round3(mv / bv_a) if bv_a > 0.001 else None,
        })
    return pd.DataFrame(rows)


def build_b200_multistream(b200_all, lang):
    rows = []
    ph = "target_verify"
    steps = n_steps(b200_all, ph)
    stream_totals = defaultdict(float)
    for (pname, tid), cats in b200_all.get("stream_cat", {}).items():
        if pname != ph:
            continue
        stream_totals[tid] += sum(cats.values())

    grand = sum(stream_totals.values())
    cl = CAT_LABELS[lang]
    for tid in sorted(stream_totals, key=stream_totals.get, reverse=True):
        tot = stream_totals[tid]
        row = {
            "Phase": ph,
            "Stream tid": tid,
            "Total ms/step": round3(tot / steps),
            "Share %": round2pct(tot, grand),
        }
        for c in CAT_ORDER:
            v = b200_all["stream_cat"].get((ph, tid), {}).get(c, 0) / steps
            if v >= 0.005:
                row[cl.get(c, c)] = round3(v)
        rows.append(row)
    return pd.DataFrame(rows)


def build_ratio(b200, mi355, b200_all, lang):
    rows = []
    for ph in PHASES:
        bk_a = phase_kernel_total(b200_all, ph) / max(n_steps(b200_all, ph), 1)
        mk = phase_kernel_total(mi355, ph) / max(n_steps(mi355, ph), 1)
        rows.append({
            "Phase": ph,
            "Category": "TOTAL kernel" if lang == "en" else "Kernel 合计",
            "B200 all-stream": round3(bk_a),
            "MI355": round3(mk),
            "MI355/B200": round3(mk / bk_a) if bk_a else None,
        })
        for fn in FUNCTIONAL:
            bv = sum_cats_per_step(b200_all, ph, fn["b200_cats"])
            mv = sum_cats_per_step(mi355, ph, fn["mi355_cats"])
            if max(bv, mv) < 0.005:
                continue
            rows.append({
                "Phase": ph,
                "Category": fn[lang],
                "B200 all-stream": round3(bv),
                "MI355": round3(mv),
                "MI355/B200": round3(mv / bv) if bv > 0.001 else None,
            })
    return pd.DataFrame(rows)


def export(lang):
    m = META[lang]
    raw_b200 = load_trace(TRACES["b200"])
    raw_mi355 = load_trace(TRACES["mi355"])

    b200 = analyze_gpu("b200", raw_b200)
    mi355 = analyze_gpu("mi355", raw_mi355)
    b200_all = attribute_kernels(raw_b200["phases"], raw_b200["kernels"], PHASE_STREAM["b200"], all_streams=True)
    b200_all["phase_durs"] = b200["phase_durs"]
    b200_all["phase_kernel"] = defaultdict(lambda: defaultdict(float))
    b200_all["phase_cat"] = defaultdict(lambda: defaultdict(float))
    b200_all["phase_group"] = defaultdict(lambda: defaultdict(float))
    for p in [x for x in raw_b200["phases"] if x["tid"] == PHASE_STREAM["b200"]]:
        ph = p["phase"]
        for k in raw_b200["kernels"]:
            frac = overlap_frac(k, p)
            if frac <= 0:
                continue
            part = k["dur_ms"] * frac
            b200_all["phase_kernel"][ph][k["name"]] += part
            b200_all["phase_cat"][ph][categorize(k["name"])] += part
            b200_all["phase_group"][ph][kernel_group(k["name"])] += part

    out = os.path.join(OUT_DIR, m["file"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        build_readme(lang).to_excel(writer, sheet_name=m["readme"], index=False)
        build_summary(b200, mi355, b200_all, lang).to_excel(writer, sheet_name=m["summary"], index=False)
        build_phase(b200, mi355, b200_all, lang).to_excel(writer, sheet_name=m["phase"], index=False)
        build_fair_compare(b200_all, mi355, lang).to_excel(writer, sheet_name=m["fair"], index=False)
        build_category_detail(b200, mi355, b200_all, lang).to_excel(writer, sheet_name=m["detail"], index=False)
        build_moe_breakdown(b200, mi355, b200_all, lang).to_excel(writer, sheet_name=m["moe_detail"], index=False)
        build_tv_kernels(b200, mi355, b200_all, lang).to_excel(writer, sheet_name=m["tv_kern"], index=False)
        build_b200_multistream(b200_all, lang).to_excel(writer, sheet_name=m["b200_ms"], index=False)
        build_ratio(b200, mi355, b200_all, lang).to_excel(writer, sheet_name=m["ratio"], index=False)

    print(f"Wrote {out}")
    print(f"  B200 phase streams: {raw_b200['ann_streams']}")
    print(f"  MI355 phase streams: {raw_mi355['ann_streams']}")
    print(f"  B200 kernel streams (top): {sorted(raw_b200['kernel_streams'], key=raw_b200['kernel_streams'].get, reverse=True)[:6]}")
    return out


if __name__ == "__main__":
    for lang in ("en", "zh"):
        export(lang)
