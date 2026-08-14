import json, collections
P=r"C:\Workspace\qwen\profile_qwen_b200_8k1k_c128_1784800637.471206\qwen_b200_8k1k_c128-1784800637.475314-TP-0-DECODE.trace.json"
with open(P,"r",encoding="utf-8") as f: d=json.load(f)
evs=d["traceEvents"]
KCAT={"kernel","gpu_memcpy","gpu_memset"}
dur=collections.Counter(); cnt=collections.Counter()
ann=collections.Counter()
for e in evs:
    c=e.get("cat")
    if c in KCAT:
        dur[e.get("name","")]+=e.get("dur",0); cnt[e.get("name","")]+=1
    if c=="gpu_user_annotation" or c=="user_annotation":
        ann[e.get("name","")[:30]]+=1
print("=== annotations (name prefix -> count) ===")
for k,v in ann.most_common(15): print(f"{v:6d}  {k}")
print("\n=== top kernels by total dur (us) ===")
for k,v in dur.most_common(35): print(f"{v/1000:9.2f}ms  x{cnt[k]:<6d} {k[:90]}")
print("\n=== candidate step/layer counters ===")
for key in ["gating","topk","moe","router","attention","attn","rmsnorm","norm","sample","embed"]:
    tot=sum(cnt[k] for k in cnt if key in k.lower())
    print(f"{key:12s} total_count={tot}")
