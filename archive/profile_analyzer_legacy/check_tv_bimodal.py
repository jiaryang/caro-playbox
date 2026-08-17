import gzip
import json
import statistics

path = r"C:\Users\jiaryang\OneDrive - Advanced Micro Devices Inc\2_task\61_sglang_glm\2_8k_profile_MTP\glm_mtp_glm_mi355_i8192_c4-1786058894.8985496-TP-0-DECODE.trace.json.gz"
ev = json.load(gzip.open(path, "rt"))["traceEvents"]

tvs = []
for e in ev:
    if isinstance(e, dict) and e.get("ph") == "X" and "TARGET_VERIFY" in e.get("name", ""):
        tvs.append({"start": e["ts"], "end": e["ts"] + e["dur"], "dur": e["dur"] / 1000})
tvs.sort(key=lambda x: x["start"])

kernels = []
for e in ev:
    if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel":
        kernels.append({"start": e["ts"], "end": e["ts"] + e["dur"], "dur": e["dur"] / 1000, "name": e.get("name", "")})

fast_durs, slow_durs = [], []
fast_k, slow_k = [], []

print("idx  tv_ms  kernel_ms  n_kern  bucket")
for i, tv in enumerate(tvs):
    ks = []
    for k in kernels:
        os_ = max(k["start"], tv["start"])
        oe = min(k["end"], tv["end"])
        if oe > os_:
            frac = (oe - os_) / (k["end"] - k["start"]) if k["end"] > k["start"] else 1
            ks.append((k["dur"] * frac, k["name"]))
    total = sum(x[0] for x in ks)
    bucket = "SLOW" if tv["dur"] > 5 else "FAST"
    if bucket == "FAST":
        fast_durs.append(tv["dur"])
        fast_k.append(total)
    else:
        slow_durs.append(tv["dur"])
        slow_k.append(total)
    print(f"{i+1:2d}  {tv['dur']:6.3f}  {total:8.3f}  {len(ks):4d}  {bucket}")

print()
print("FAST steps:", len(fast_durs), "tv mean", statistics.mean(fast_durs), "kernel mean", statistics.mean(fast_k))
print("SLOW steps:", len(slow_durs), "tv mean", statistics.mean(slow_durs), "kernel mean", statistics.mean(slow_k))

# top kernels in slow steps only
from collections import defaultdict

slow_kernel = defaultdict(float)
for i, tv in enumerate(tvs):
    if tv["dur"] <= 5:
        continue
    for k in kernels:
        os_ = max(k["start"], tv["start"])
        oe = min(k["end"], tv["end"])
        if oe > os_:
            frac = (oe - os_) / (k["end"] - k["start"]) if k["end"] > k["start"] else 1
            slow_kernel[k["name"]] += k["dur"] * frac

print("\nTop kernels in SLOW target_verify steps (total ms, /10 steps):")
for name, ms in sorted(slow_kernel.items(), key=lambda x: -x[1])[:10]:
    print(f"  {ms/10:6.3f} ms/step  {name[:70]}")
