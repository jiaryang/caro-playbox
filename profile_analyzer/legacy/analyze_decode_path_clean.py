import gzip
import json
import os
import re
import statistics
from collections import defaultdict

base = os.path.dirname(os.path.abspath(__file__))
OUTLIER_MS = 100


def load_events(path):
    try:
        with gzip.open(path, "rt") as fh:
            return json.load(fh)["traceEvents"], None
    except (EOFError, OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def bucket_events(events):
    buckets = {"draft": [], "draft_extend": [], "target_verify": []}
    tv_by_bs = defaultdict(list)
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0) / 1000
        if name == "draft":
            buckets["draft"].append(dur)
        elif name == "draft_extend":
            buckets["draft_extend"].append(dur)
        elif "TARGET_VERIFY" in name:
            buckets["target_verify"].append(dur)
            m = re.search(r"bs=(\d+)", name)
            tv_by_bs[m.group(1) if m else "unknown"].append(dur)
    return buckets, dict(tv_by_bs)


def clean_stats(vals):
    clean = [v for v in vals if v <= OUTLIER_MS]
    outliers = [v for v in vals if v > OUTLIER_MS]
    return {
        "n_raw": len(vals),
        "n_clean": len(clean),
        "n_outlier": len(outliers),
        "outlier_vals": sorted(outliers, reverse=True),
        "sum_clean": sum(clean),
        "med_clean": statistics.median(clean) if clean else 0,
        "avg_clean": sum(clean) / len(clean) if clean else 0,
    }


rows = []
for fn in sorted(os.listdir(base)):
    if not fn.endswith(".trace.json.gz"):
        continue
    m = re.match(r"glm_mtp_glm_(b200|mi355)_i70000_c(\d+)", fn)
    if not m:
        continue
    gpu, conc = m.group(1), int(m.group(2))
    events, err = load_events(os.path.join(base, fn))
    if err:
        rows.append({"gpu": gpu, "conc": conc, "error": err})
        continue
    buckets, tv_by_bs = bucket_events(events)
    ds = clean_stats(buckets["draft"])
    des = clean_stats(buckets["draft_extend"])
    tvs = clean_stats(buckets["target_verify"])
    total = ds["sum_clean"] + des["sum_clean"] + tvs["sum_clean"]
    rows.append(
        {
            "gpu": gpu,
            "conc": conc,
            "draft": ds,
            "de": des,
            "tv": tvs,
            "total": total,
            "tv_by_bs": {
                bs: clean_stats(vals) for bs, vals in tv_by_bs.items()
            },
        }
    )

print(f"=== Decode path analysis (outliers > {OUTLIER_MS}ms excluded) ===")
print("Events: draft | draft_extend | step[TARGET_VERIFY bs=N]")
print()

print("--- 1. Total GPU time (ms) ---")
print(
    f"{'GPU':<6} {'conc':>4}  {'draft':>8} {'d_extend':>9} {'t_verify':>10} "
    f"{'total':>8}  | {'draft%':>6} {'de%':>6} {'tv%':>6}  {'outliers':>8}"
)
print("-" * 88)
for r in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in r:
        print(f"{r['gpu']:<6} {r['conc']:>4}  SKIP ({r['error'][:50]})")
        continue
    ds, de, tv = r["draft"], r["de"], r["tv"]
    total = r["total"]
    outliers = ds["n_outlier"] + de["n_outlier"] + tv["n_outlier"]
    print(
        f"{r['gpu']:<6} {r['conc']:>4}  "
        f"{ds['sum_clean']:>8.2f} {de['sum_clean']:>9.2f} {tv['sum_clean']:>10.2f} "
        f"{total:>8.2f}  | "
        f"{ds['sum_clean']/total*100:>5.1f}% {de['sum_clean']/total*100:>5.1f}% "
        f"{tv['sum_clean']/total*100:>5.1f}%  {outliers:>8}"
    )

print()
print("--- 2. Per-event median (ms, clean only) ---")
print(
    f"{'GPU':<6} {'conc':>4}  {'draft':>8} {'d_extend':>9} {'t_verify':>10}  "
    f"| {'#draft':>6} {'#d_ext':>6} {'#tv':>5}"
)
print("-" * 72)
for r in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in r:
        continue
    ds, de, tv = r["draft"], r["de"], r["tv"]
    print(
        f"{r['gpu']:<6} {r['conc']:>4}  "
        f"{ds['med_clean']:>8.3f} {de['med_clean']:>9.3f} {tv['med_clean']:>10.3f}  "
        f"| {ds['n_clean']:>6} {de['n_clean']:>6} {tv['n_clean']:>5}"
    )

print()
print("--- 3. Outlier details (removed events) ---")
for r in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in r:
        continue
    parts = []
    for label, key in [("draft", "draft"), ("draft_extend", "de"), ("target_verify", "tv")]:
        vals = r[key]["outlier_vals"]
        if vals:
            parts.append(f"{label}: {[round(v,1) for v in vals]}")
    if parts:
        print(f"{r['gpu']} c{r['conc']}: " + "; ".join(parts))

print()
print("--- 4. target_verify median by batch size (ms, clean) ---")
for r in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in r:
        continue
    parts = []
    for bs, st in sorted(r["tv_by_bs"].items(), key=lambda x: int(x[0])):
        parts.append(f"bs{bs}: med={st['med_clean']:.2f} n={st['n_clean']}")
    print(f"{r['gpu']} c{r['conc']}: {', '.join(parts)}")

print()
print("--- 5. B200 vs MI355 ratio (clean total ms) ---")
for conc in [4, 8, 16, 32, 64]:
    b200 = next((r for r in rows if r.get("gpu") == "b200" and r.get("conc") == conc and "error" not in r), None)
    mi = next((r for r in rows if r.get("gpu") == "mi355" and r.get("conc") == conc and "error" not in r), None)
    if not b200 or not mi:
        continue
    print(f"conc {conc}:")
    for label, key in [("draft", "draft"), ("draft_extend", "de"), ("target_verify", "tv"), ("TOTAL", None)]:
        if key:
            b, m = b200[key]["sum_clean"], mi[key]["sum_clean"]
        else:
            b, m = b200["total"], mi["total"]
        ratio = m / b if b else float("inf")
        print(f"  {label:14}  B200={b:7.1f}ms  MI355={m:7.1f}ms  MI355/B200={ratio:.2f}x")
