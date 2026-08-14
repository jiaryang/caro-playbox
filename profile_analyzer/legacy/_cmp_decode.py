import json, collections

ATOM = r"C:\Workspace\qwen\Atom_Qwen3.5-397B-A17B-MXFP4_ts_20260728_101711_991.pt.trace.json"
SGL  = r"C:\Workspace\qwen\profile_qwen_mi355_8k1k_c128_1784806436.739748\qwen_mi355_8k1k_c128-1784806436.7420862-TP-0-DECODE.trace.json"

def cat_of(name):
    n = name.lower()
    is_tri = n.startswith("triton") or "fused" in n
    if ("cross_device_reduce" in n or "nccl" in n or "rccl" in n or "all_gather" in n or "allgather" in n
        or "reduce_scatter" in n or "quickreduce" in n or "custom_ar" in n
        or (("all_reduce" in n or "allreduce" in n) and not is_tri)):
        return "communication"
    if "moe" in n or "topkgating" in n or "expert" in n or "grouped_gemm" in n or "group_gemm" in n:
        return "moe"
    if any(k in n for k in ["gated_delta","gdn","causal_conv","recurrent_gated","chunk_gated","split_chunk","chunk_fwd","recompute_w_u"]):
        return "mixer:linear(gdn)"
    if any(k in n for k in ["fmha","flash_attn","_attn_","paged_attn","paged_attention","mha","attention","_attn"]):
        return "mixer:full-attn"
    if n.startswith("cijk") or "gemm" in n or "_mm_" in n or "matmul" in n or "hgemm" in n or "nvjet" in n or "cutlass" in n:
        return "gemm"
    if "rsqrt" in n or "rmsnorm" in n or "layernorm" in n or "qk_norm" in n or "layer_norm" in n:
        return "normalization"
    if "quant" in n or "fp4" in n or "fp8" in n or "preshuffle" in n or "scaled" in n:
        return "quantization"
    if any(k in n for k in ["act_and_mul","silu","gelu","sigmoid","softmax"]):
        return "activation"
    if any(k in n for k in ["copybuffer","memcpy","memset","__amd_rocclr","dtod","htod"]):
        return "memcpy"
    if any(k in n for k in ["elementwise","embedding","rope","rotary","index","cat","slice","add","mul","fill","_to_copy"]):
        return "elementwise"
    return "other"

KCAT = {"kernel","gpu_memcpy","gpu_memset"}

def analyze(path, decode_window=False):
    with open(path,"r",encoding="utf-8") as f:
        d=json.load(f)
    evs=d["traceEvents"]
    lo,hi=-1e30,1e30
    if decode_window:
        dc=[(e["ts"],e["ts"]+e.get("dur",0)) for e in evs
            if e.get("cat")=="gpu_user_annotation" and (e.get("name","")).startswith("decode[")]
        lo,hi=min(a for a,b in dc),max(b for a,b in dc)
    catt=collections.Counter(); catc=collections.Counter()
    knames=collections.defaultdict(lambda: collections.Counter())  # cat -> {name: time}
    busy=0.0
    for e in evs:
        if e.get("cat") in KCAT:
            ts=e.get("ts")
            if ts is None or ts<lo or ts>hi: continue
            dur=e.get("dur",0); nm=e.get("name","")
            c=cat_of(nm); catt[c]+=dur; catc[c]+=1; knames[c][nm]+=dur; busy+=dur
    return catt,catc,knames,busy

print("loading Atom (decode window)...",flush=True)
aC,aN,aK,aB=analyze(ATOM,decode_window=True)
print("loading sglang MI355 c128 decode...",flush=True)
sC,sN,sK,sB=analyze(SGL,decode_window=False)

order=["communication","moe","gemm","mixer:linear(gdn)","mixer:full-attn","normalization","activation","quantization","elementwise","memcpy","other"]
print("\n================ DECODE kernel-category comparison (share of GPU busy) ================")
print(f"{'category':22}{'Atom %':>10}{'sglang %':>10}   {'Atom ms':>10}{'sgl ms':>10}")
for c in order:
    if aC.get(c) or sC.get(c):
        print(f"{c:22}{ (aC.get(c,0)/aB*100):9.1f}%{ (sC.get(c,0)/sB*100):9.1f}%   {aC.get(c,0)/1000:10.1f}{sC.get(c,0)/1000:10.1f}")
print(f"{'TOTAL busy ms':22}{'':>10}{'':>10}   {aB/1000:10.1f}{sB/1000:10.1f}")

# combined mixer
am=(aC.get('mixer:linear(gdn)',0)+aC.get('mixer:full-attn',0))/aB*100
sm=(sC.get('mixer:linear(gdn)',0)+sC.get('mixer:full-attn',0))/sB*100
print(f"\n  token-mixer total: Atom {am:.1f}%  |  sglang {sm:.1f}%")

for label,K in [("ATOM decode",aK),("SGLANG MI355 c128 decode",sK)]:
    print(f"\n================ {label}: actual kernels by category (top by time) ================")
    tot=sum(sum(v.values()) for v in K.values())
    for c in order:
        if c not in K: continue
        print(f"\n  [{c}]  {sum(K[c].values())/1000:.1f} ms")
        for nm,t in K[c].most_common(6):
            print(f"     {t/1000:8.1f}ms  {nm[:82]}")
