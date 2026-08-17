"""Export kernel composition analysis to Excel (Chinese + English)."""
import gzip
import glob
import json
import os
import re
import statistics
from collections import defaultdict

import pandas as pd

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

KERNEL_GROUPS = [
    ("sparse_mla", "_sparse_mla_fwd", "Sparse MLA Attention"),
    ("gemm1_a4w4", "gemm1_a4w4", "MoE GEMM1 (a4w4)"),
    ("gemm2_a4w4", "gemm2_a4w4", "MoE GEMM2 (a4w4)"),
    ("nvfp4_bmm", "bmm_e2m1", "NVFP4 BMM (E2M1)"),
    ("nvjet", "nvjet", "NVJet GEMM"),
    ("fmha", "fmhasm100f", "FMHA Attention"),
    ("nccl_ar", "nccldevkernel_allreduce", "NCCL AllReduce"),
    ("flashinfer_ar", "mnnvl_allreduce", "FlashInfer AllReduce"),
    ("aiter_rs", "reduce_scatter_cross_device", "Aiter ReduceScatter"),
    ("aiter_ar", "cross_device_reduce", "Aiter AllReduce"),
    ("topk_decode", "topk_transform_decode", "TopK Decode"),
    ("topk_prefill", "topk_transform_prefill", "TopK Prefill"),
    ("ck_moe_gemm", "kernel_moe_gemm", "CK MoE GEMM"),
    ("hgemm", "hgemm_bf16", "HGEMM BF16"),
    ("deep_gemm_mqa", "deep_gemm::sm100_mqa", "DeepGEMM MQA"),
    ("moe_routing", "routingindicesdynblock", "MoE Routing"),
    ("rmsnorm", "local_device_load_rmsnorm", "RMSNorm"),
    ("fused_a_gemm", "fused_a_gemm_kernel", "Fused A GEMM"),
]

# i18n labels
PHASE_LABELS = {
    "en": {"draft": "draft", "target_verify": "target_verify", "draft_extend": "draft_extend"},
    "zh": {"draft": "draft", "target_verify": "target_verify", "draft_extend": "draft_extend"},
}

CAT_LABELS = {
    "en": {
        "attention": "attention",
        "moe_gemm": "moe_gemm",
        "dense_gemm": "dense_gemm",
        "comm": "comm",
        "moe_route": "moe_route",
        "norm_rope": "norm_rope",
        "quant": "quant",
        "kv_cache": "kv_cache",
        "verify_misc": "verify_misc",
        "other": "other",
    },
    "zh": {
        "attention": "Attention",
        "moe_gemm": "MoE GEMM",
        "dense_gemm": "Dense GEMM",
        "comm": "通信 Comm",
        "moe_route": "MoE 路由 TopK",
        "norm_rope": "Norm/RoPE",
        "quant": "量化 Quant",
        "kv_cache": "KV Cache",
        "verify_misc": "Verify/杂项",
        "other": "其他",
    },
}

META = {
    "en": {
        "file": "GLM_MTP_kernel_analysis_EN.xlsx",
        "sheet_readme": "README",
        "sheet_summary": "Summary",
        "sheet_phase": "Phase_per_step",
        "sheet_cat": "Kernel_by_category",
        "sheet_tv_kern": "TV_top_kernels",
        "sheet_de_kern": "DE_top_kernels",
        "sheet_ratio": "MI355_vs_B200",
        "title": "GLM MTP Decode Kernel Analysis (conc=4)",
        "method": [
            "Workload: 8k (8192/1024 DECODE trace) and 70k (70000/300 EXTEND trace), conc=4",
            "GPU stream: MI355 cuda stream tid=8, B200 main compute stream tid=132",
            "Phase markers: gpu_user_annotation on the same stream (not CPU thread annotations)",
            "Kernels: cat=kernel events on the same stream tid",
            "Unit: ms/step (total kernel or phase time / number of phase steps)",
            "Model: MI355 amd/GLM-5.2-MXFP4, B200 nvidia/GLM-5.2-NVFP4, TP=4",
        ],
        "col_workload": "Workload",
        "col_gpu": "GPU",
        "col_phase": "Phase",
        "col_category": "Category",
        "col_group": "Kernel Group",
        "col_steps": "Steps",
        "col_b200_k": "B200 kernel ms/step",
        "col_mi355_k": "MI355 kernel ms/step",
        "col_b200_pct": "B200 %",
        "col_mi355_pct": "MI355 %",
        "col_b200_phase": "B200 phase ms/step",
        "col_mi355_phase": "MI355 phase ms/step",
        "col_ratio": "MI355/B200",
        "col_kernel_sum": "Kernel sum ms/step",
        "col_phase_sum": "Phase sum ms/step",
        "col_gap": "Non-kernel gap ms",
        "col_b200_kernels": "B200 kernels",
        "col_mi355_kernels": "MI355 kernels",
    },
    "zh": {
        "file": "GLM_MTP_kernel_analysis_ZH.xlsx",
        "sheet_readme": "说明",
        "sheet_summary": "总览",
        "sheet_phase": "Phase耗时",
        "sheet_cat": "Kernel分类",
        "sheet_tv_kern": "TV核心Kernel",
        "sheet_de_kern": "DE核心Kernel",
        "sheet_ratio": "MI355对比B200",
        "title": "GLM MTP Decode Kernel 分析 (conc=4)",
        "method": [
            "测试场景：8k (8192/1024 DECODE trace) 与 70k (70000/300 EXTEND trace)，conc=4",
            "GPU Stream：MI355 cuda stream tid=8，B200 主计算 stream tid=132",
            "Phase 标记：gpu_user_annotation（同一 stream，非 CPU thread 标注）",
            "Kernel：同一 stream tid 上的 cat=kernel 事件",
            "单位：ms/step（kernel 或 phase 总时间 / phase 步数）",
            "模型：MI355 amd/GLM-5.2-MXFP4，B200 nvidia/GLM-5.2-NVFP4，TP=4",
        ],
        "col_workload": "场景",
        "col_gpu": "GPU",
        "col_phase": "阶段",
        "col_category": "分类",
        "col_group": "Kernel 组",
        "col_steps": "步数",
        "col_b200_k": "B200 kernel ms/step",
        "col_mi355_k": "MI355 kernel ms/step",
        "col_b200_pct": "B200 占比%",
        "col_mi355_pct": "MI355 占比%",
        "col_b200_phase": "B200 phase ms/step",
        "col_mi355_phase": "MI355 phase ms/step",
        "col_ratio": "MI355/B200",
        "col_kernel_sum": "Kernel合计 ms/step",
        "col_phase_sum": "Phase合计 ms/step",
        "col_gap": "非Kernel开销 ms",
        "col_b200_kernels": "B200 Kernel 列表",
        "col_mi355_kernels": "MI355 Kernel 列表",
    },
}


def find_file(pattern):
    best = None
    for root in ROOTS:
        search = os.path.join(root, "**", pattern) if "**" not in pattern else os.path.join(root, pattern)
        for path in glob.glob(search, recursive=True):
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


def kernel_group(name):
    n = name.lower()
    for gid, key, _ in KERNEL_GROUPS:
        if key in n:
            return gid
    s = re.sub(r"\s+", " ", name.strip())
    return s[:40] + ("..." if len(s) > 40 else "")


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
        "path": os.path.basename(path),
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


def med_phase(r, ph):
    d = r["phase_durs"].get(ph, [])
    return statistics.median(d) if d else 0.0


def round3(x):
    return round(x, 3) if x is not None else None


def round2pct(num, den):
    return round(num / den * 100, 1) if den else None


def kernels_in_category(r, ph, cat, min_ms=0.005):
    n = n_steps(r, ph)
    if not n:
        return ""
    items = []
    for kname, total_ms in r["phase_kernel"].get(ph, {}).items():
        if categorize(kname) != cat:
            continue
        per_step = total_ms / n
        if per_step < min_ms:
            continue
        items.append((per_step, kname))
    items.sort(key=lambda x: x[0], reverse=True)
    return "; ".join(f"{name} ({round3(ms)} ms/step)" for ms, name in items)


def load_all():
    results = {}
    for key, (pat, tid) in CONFIG.items():
        path = find_file(pat)
        if path:
            results[key] = analyze(path, tid)
    return results


def build_readme(lang):
    m = META[lang]
    rows = [{"Item" if lang == "en" else "项目": m["title"]}] + [
        {"Item" if lang == "en" else "项目": line} for line in m["method"]
    ]
    return pd.DataFrame(rows)


def build_summary(results, lang):
    m = META[lang]
    rows = []
    for wl in ["8k", "70k"]:
        for gpu in ["b200", "mi355"]:
            r = results[(wl, gpu)]
            kd = {p: round3(phase_kernel_total(r, p) / max(n_steps(r, p), 1)) for p in PHASES}
            pd_ = {p: round3(med_phase(r, p)) for p in PHASES}
            rows.append(
                {
                    m["col_workload"]: wl,
                    m["col_gpu"]: gpu.upper(),
                    "Stream tid": r["stream_tid"],
                    f"draft kernel": kd["draft"],
                    f"target_verify kernel": kd["target_verify"],
                    f"draft_extend kernel": kd["draft_extend"],
                    m["col_kernel_sum"]: round3(sum(kd.values())),
                    f"draft phase": pd_["draft"],
                    f"target_verify phase": pd_["target_verify"],
                    f"draft_extend phase": pd_["draft_extend"],
                    m["col_phase_sum"]: round3(sum(pd_.values())),
                }
            )
    df = pd.DataFrame(rows)
    # add ratio row per workload
    extra = []
    for wl in ["8k", "70k"]:
        b = [x for x in rows if x[m["col_workload"]] == wl and x[m["col_gpu"]] == "B200"][0]
        mi = [x for x in rows if x[m["col_workload"]] == wl and x[m["col_gpu"]] == "MI355"][0]
        extra.append(
            {
                m["col_workload"]: wl,
                m["col_gpu"]: "MI355/B200",
                "Stream tid": "-",
                f"draft kernel": round3(mi[f"draft kernel"] / b[f"draft kernel"]) if b[f"draft kernel"] else None,
                f"target_verify kernel": round3(mi[f"target_verify kernel"] / b[f"target_verify kernel"]),
                f"draft_extend kernel": round3(mi[f"draft_extend kernel"] / b[f"draft_extend kernel"]) if b[f"draft_extend kernel"] else None,
                m["col_kernel_sum"]: round3(mi[m["col_kernel_sum"]] / b[m["col_kernel_sum"]]),
                f"draft phase": round3(mi[f"draft phase"] / b[f"draft phase"]) if b[f"draft phase"] else None,
                f"target_verify phase": round3(mi[f"target_verify phase"] / b[f"target_verify phase"]),
                f"draft_extend phase": round3(mi[f"draft_extend phase"] / b[f"draft_extend phase"]) if b[f"draft_extend phase"] else None,
                m["col_phase_sum"]: round3(mi[m["col_phase_sum"]] / b[m["col_phase_sum"]]),
            }
        )
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def build_phase(results, lang):
    m = META[lang]
    rows = []
    for wl in ["8k", "70k"]:
        for ph in PHASES:
            b200, mi = results[(wl, "b200")], results[(wl, "mi355")]
            bk = round3(phase_kernel_total(b200, ph) / max(n_steps(b200, ph), 1))
            mk = round3(phase_kernel_total(mi, ph) / max(n_steps(mi, ph), 1))
            bp = round3(med_phase(b200, ph))
            mp = round3(med_phase(mi, ph))
            rows.append(
                {
                    m["col_workload"]: wl,
                    m["col_phase"]: ph,
                    "B200 steps": n_steps(b200, ph),
                    "MI355 steps": n_steps(mi, ph),
                    m["col_b200_k"]: bk,
                    m["col_mi355_k"]: mk,
                    m["col_b200_phase"]: bp,
                    m["col_mi355_phase"]: mp,
                    "B200 non-kernel gap": round3(bp - bk) if bp and bk else None,
                    "MI355 non-kernel gap": round3(mp - mk) if mp and mk else None,
                    m["col_ratio"]: round3(mk / bk) if bk else None,
                }
            )
    return pd.DataFrame(rows)


def build_category(results, lang):
    m = META[lang]
    cl = CAT_LABELS[lang]
    rows = []
    for wl in ["8k", "70k"]:
        for ph in PHASES:
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
                rows.append(
                    {
                        m["col_workload"]: wl,
                        m["col_phase"]: ph,
                        m["col_category"]: cl.get(c, c),
                        m["col_b200_k"]: round3(bv),
                        m["col_mi355_k"]: round3(mv),
                        m["col_b200_pct"]: round2pct(bv, bt),
                        m["col_mi355_pct"]: round2pct(mv, mt),
                        m["col_ratio"]: round3(mv / bv) if bv > 0.001 else None,
                        m["col_b200_kernels"]: kernels_in_category(b200, ph, c),
                        m["col_mi355_kernels"]: kernels_in_category(mi, ph, c),
                    }
                )
    return pd.DataFrame(rows)


def build_top_kernels(results, phase, lang):
    m = META[lang]
    group_desc = {gid: desc for gid, _, desc in KERNEL_GROUPS}
    rows = []
    for wl in ["8k", "70k"]:
        b200, mi = results[(wl, "b200")], results[(wl, "mi355")]
        nb, nm = n_steps(b200, phase), n_steps(mi, phase)
        groups = set(b200["phase_group"].get(phase, {})) | set(mi["phase_group"].get(phase, {}))
        ranked = sorted(
            groups,
            key=lambda g: k_per_step(b200, phase, "phase_group", g) + k_per_step(mi, phase, "phase_group", g),
            reverse=True,
        )
        for g in ranked:
            bv = k_per_step(b200, phase, "phase_group", g)
            mv = k_per_step(mi, phase, "phase_group", g)
            if max(bv, mv) < 0.02:
                continue
            rows.append(
                {
                    m["col_workload"]: wl,
                    m["col_group"]: g,
                    "Description": group_desc.get(g, g),
                    m["col_b200_k"]: round3(bv),
                    m["col_mi355_k"]: round3(mv),
                    m["col_ratio"]: round3(mv / bv) if bv > 0.001 else None,
                }
            )
    return pd.DataFrame(rows)


def build_ratio(results, lang):
    m = META[lang]
    cl = CAT_LABELS[lang]
    rows = []
    for wl in ["8k", "70k"]:
        for ph in PHASES:
            b200, mi = results[(wl, "b200")], results[(wl, "mi355")]
            nb, nm = n_steps(b200, ph), n_steps(mi, ph)
            # total line
            bk = phase_kernel_total(b200, ph) / nb if nb else 0
            mk = phase_kernel_total(mi, ph) / nm if nm else 0
            bp, mp = med_phase(b200, ph), med_phase(mi, ph)
            rows.append(
                {
                    m["col_workload"]: wl,
                    m["col_phase"]: ph,
                    m["col_category"]: "TOTAL kernel",
                    m["col_b200_k"]: round3(bk),
                    m["col_mi355_k"]: round3(mk),
                    m["col_ratio"]: round3(mk / bk) if bk else None,
                    m["col_b200_phase"]: round3(bp),
                    m["col_mi355_phase"]: round3(mp),
                }
            )
            for c in sorted(set(b200["phase_cat"].get(ph, {})) | set(mi["phase_cat"].get(ph, {}))):
                bv = k_per_step(b200, ph, "phase_cat", c)
                mv = k_per_step(mi, ph, "phase_cat", c)
                if bv < 0.005 and mv < 0.005:
                    continue
                rows.append(
                    {
                        m["col_workload"]: wl,
                        m["col_phase"]: ph,
                        m["col_category"]: cl.get(c, c),
                        m["col_b200_k"]: round3(bv),
                        m["col_mi355_k"]: round3(mv),
                        m["col_ratio"]: round3(mv / bv) if bv > 0.001 else None,
                        m["col_b200_phase"]: None,
                        m["col_mi355_phase"]: None,
                    }
                )
    return pd.DataFrame(rows)


def export(lang):
    m = META[lang]
    results = load_all()
    out = os.path.join(ROOT, m["file"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        build_readme(lang).to_excel(writer, sheet_name=m["sheet_readme"], index=False)
        build_summary(results, lang).to_excel(writer, sheet_name=m["sheet_summary"], index=False)
        build_phase(results, lang).to_excel(writer, sheet_name=m["sheet_phase"], index=False)
        build_category(results, lang).to_excel(writer, sheet_name=m["sheet_cat"], index=False)
        build_top_kernels(results, "target_verify", lang).to_excel(writer, sheet_name=m["sheet_tv_kern"], index=False)
        build_top_kernels(results, "draft_extend", lang).to_excel(writer, sheet_name=m["sheet_de_kern"], index=False)
        build_ratio(results, lang).to_excel(writer, sheet_name=m["sheet_ratio"], index=False)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    export("en")
    export("zh")
