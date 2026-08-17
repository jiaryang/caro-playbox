import json, collections
p = r"C:\Workspace\qwen\Atom_Qwen3.5-397B-A17B-MXFP4_ts_20260728_101711_991.pt.trace.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
evs = d["traceEvents"]

# ---- phase windows from gpu_user_annotation ----
pf = []; dc = []
pf_tok = 0; pf_steps = 0; dc_steps = 0
for e in evs:
    if e.get("cat") == "gpu_user_annotation":
        n = e.get("name", ""); ts = e.get("ts"); dur = e.get("dur", 0)
        if ts is None: continue
        if n.startswith("prefill["): pf.append((ts, ts+dur))
        elif n.startswith("decode["): dc.append((ts, ts+dur))
# step stats from CPU user_annotation (authoritative step count/tokens)
import re
pf_durs=[]; dc_durs=[]; dc_bs=collections.Counter()
for e in evs:
    if e.get("cat") == "user_annotation":
        n = e.get("name", "")
        if n.startswith("prefill["):
            pf_steps += 1; pf_durs.append(e.get("dur",0))
            m = re.search(r"tok=(\d+)", n)
            if m: pf_tok += int(m.group(1))
        elif n.startswith("decode["):
            dc_steps += 1; dc_durs.append(e.get("dur",0))
            m = re.search(r"bs=(\d+)", n)
            if m: dc_bs[int(m.group(1))] += 1

pf_s, pf_e = min(x[0] for x in pf), max(x[1] for x in pf)
dc_s, dc_e = min(x[0] for x in dc), max(x[1] for x in dc)
print(f"PREFILL window: {pf_s/1e6:.3f}s .. {pf_e/1e6:.3f}s  ({(pf_e-pf_s)/1000:.1f} ms)  steps={pf_steps} tokens={pf_tok:,}")
print(f"  step wall: avg={sum(pf_durs)/len(pf_durs)/1000:.2f}ms total={sum(pf_durs)/1000:.1f}ms")
print(f"DECODE  window: {dc_s/1e6:.3f}s .. {dc_e/1e6:.3f}s  ({(dc_e-dc_s)/1000:.1f} ms)  steps={dc_steps}")
print(f"  step wall: avg={sum(dc_durs)/len(dc_durs)/1000:.3f}ms total={sum(dc_durs)/1000:.1f}ms")
print(f"  decode steps @bs=128: {dc_bs.get(128,0)}  (bs distribution top: {dc_bs.most_common(3)})")

# ---- kernel categorizer (heuristic) ----
def cat_of(name):
    n = name.lower()
    is_tri = n.startswith("triton") or "fused" in n
    # communication (true comm kernels, not triton fusions that merely mention all_reduce)
    if ("cross_device_reduce" in n or "nccl" in n or "rccl" in n or "all_gather" in n
        or "reduce_scatter" in n or (("all_reduce" in n or "allreduce" in n) and not is_tri)):
        return "communication"
    # MoE
    if "moe" in n or "topkgating" in n or "expert" in n:
        return "moe"
    # GatedDeltaNet linear-attn token mixer
    if any(k in n for k in ["gated_delta","gdn","causal_conv","recurrent_gated","chunk_gated","split_chunk"]):
        return "gdn(linear-attn)"
    # full attention
    if any(k in n for k in ["fmha","flash_attn","_attn_","paged_attn","mha","attention"]):
        return "attn(full)"
    # gemm (dense matmul / projections) -- Tensile Cijk, aiter gemm, triton gemm fusions
    if n.startswith("cijk") or "gemm" in n or "_mm_" in n or "matmul" in n or "hgemm" in n:
        return "gemm"
    # normalization (rmsnorm / qk_norm)
    if "rsqrt" in n or "rmsnorm" in n or "layernorm" in n or "qk_norm" in n:
        return "normalization"
    # quantization
    if "quant" in n or "fp4" in n or "fp8" in n or "preshuffle" in n:
        return "quantization"
    # activation / elementwise
    if any(k in n for k in ["act_and_mul","silu","gelu","sigmoid","softmax"]):
        return "activation"
    if any(k in n for k in ["copybuffer","memcpy","memset","copy_","__amd_rocclr"]):
        return "memcpy"
    if any(k in n for k in ["add","mul","sigmoid","slice","index","cat","fill","elementwise","_to_copy","embedding"]):
        return "elementwise"
    return "other"

KCAT = {"kernel","gpu_memcpy","gpu_memset"}
def collect(lo, hi):
    catt = collections.Counter(); catc = collections.Counter()
    topk = collections.Counter()
    busy = 0.0
    for e in evs:
        if e.get("cat") in KCAT:
            ts = e.get("ts")
            if ts is None or ts < lo or ts > hi: continue
            dur = e.get("dur", 0); nm = e.get("name","")
            c = cat_of(nm)
            catt[c] += dur; catc[c] += 1; topk[nm] += dur; busy += dur
    return catt, catc, topk, busy

for label, lo, hi in [("PREFILL", pf_s, pf_e), ("DECODE", dc_s, dc_e)]:
    catt, catc, topk, busy = collect(lo, hi)
    print(f"\n========== {label}  kernel composition (GPU busy sum = {busy/1000:.1f} ms across streams) ==========")
    print(f"{'category':18}{'time(ms)':>12}{'share':>9}{'count':>10}")
    for c, t in catt.most_common():
        print(f"{c:18}{t/1000:12.1f}{t/busy*100:8.1f}%{catc[c]:10d}")
    print(f"  -- top 12 kernels by time --")
    for nm, t in topk.most_common(12):
        print(f"   {t/1000:9.1f}ms {t/busy*100:5.1f}%  {nm[:78]}")
