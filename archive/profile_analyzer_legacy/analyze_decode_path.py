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


def stats(vals):
    clean = [v for v in vals if v <= OUTLIER_MS]
    return {
        "n": len(vals),
        "sum": sum(vals),
        "sum_clean": sum(clean),
        "med": statistics.median(vals) if vals else 0,
        "med_clean": statistics.median(clean) if clean else 0,
        "outliers": len(vals) - len(clean),
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
    ds = stats(buckets["draft"])
    des = stats(buckets["draft_extend"])
    tvs = stats(buckets["target_verify"])
    total_sum = ds["sum"] + des["sum"] + tvs["sum"]
    total_clean = ds["sum_clean"] + des["sum_clean"] + tvs["sum_clean"]
    rows.append(
        {
            "gpu": gpu,
            "conc": conc,
            "draft": ds,
            "de": des,
            "tv": tvs,
            "total_sum": total_sum,
            "total_clean": total_clean,
            "tv_by_bs": tv_by_bs,
        }
    )

print("Profile source: sglang torch profiler trace (TP-0, decode window)")
print("Event mapping: draft, draft_extend, step[TARGET_VERIFY bs=N]")
print()

print("=== RAW TOTAL GPU TIME (ms) ===")
print(
    f"{'GPU':<6} {'conc':>4} {'draft':>8} {'d_ext':>8} {'t_verify':>10} "
    f"{'total':>8} | {'draft%':>6} {'d_ext%':>6} {'tv%':>6}"
)
print("-" * 78)
for row in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in row:
        print(f"{row['gpu']:<6} {row['conc']:>4}  SKIP (corrupt trace)")
        continue
    ds, des, tvs = row["draft"], row["de"], row["tv"]
    total = row["total_sum"]
    print(
        f"{row['gpu']:<6} {row['conc']:>4} "
        f"{ds['sum']:>8.2f} {des['sum']:>8.2f} {tvs['sum']:>10.2f} "
        f"{total:>8.2f} | "
        f"{ds['sum']/total*100:>5.1f}% {des['sum']/total*100:>5.1f}% "
        f"{tvs['sum']/total*100:>5.1f}%"
    )

print()
print("=== CLEAN TOTAL (exclude events >100ms outliers) ===")
print(
    f"{'GPU':<6} {'conc':>4} {'draft':>8} {'d_ext':>8} {'t_verify':>10} "
    f"{'total':>8} | {'draft%':>6} {'d_ext%':>6} {'tv%':>6} {'outliers':>8}"
)
print("-" * 88)
for row in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in row:
        continue
    ds, des, tvs = row["draft"], row["de"], row["tv"]
    total = row["total_clean"]
    outliers = ds["outliers"] + des["outliers"] + tvs["outliers"]
    print(
        f"{row['gpu']:<6} {row['conc']:>4} "
        f"{ds['sum_clean']:>8.2f} {des['sum_clean']:>8.2f} {tvs['sum_clean']:>10.2f} "
        f"{total:>8.2f} | "
        f"{ds['sum_clean']/total*100:>5.1f}% {des['sum_clean']/total*100:>5.1f}% "
        f"{tvs['sum_clean']/total*100:>5.1f}% {outliers:>8}"
    )

print()
print("=== PER-EVENT MEDIAN (ms) ===")
print(
    f"{'GPU':<6} {'conc':>4} {'draft':>8} {'d_ext':>8} {'t_verify':>10} | "
    f"{'#draft':>6} {'#d_ext':>6} {'#tv':>5}"
)
print("-" * 72)
for row in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in row:
        continue
    ds, des, tvs = row["draft"], row["de"], row["tv"]
    print(
        f"{row['gpu']:<6} {row['conc']:>4} "
        f"{ds['med']:>8.3f} {des['med']:>8.3f} {tvs['med']:>10.3f} | "
        f"{ds['n']:>6} {des['n']:>6} {tvs['n']:>5}"
    )

print()
print("=== TARGET_VERIFY median by batch size (ms) ===")
for row in sorted(rows, key=lambda x: (x["gpu"], x["conc"])):
    if "error" in row:
        print(f"{row['gpu']} c{row['conc']}: SKIP")
        continue
    parts = []
    for bs, vals in sorted(row["tv_by_bs"].items(), key=lambda x: int(x[0])):
        parts.append(f"bs{bs}: med={statistics.median(vals):.2f}ms n={len(vals)}")
    print(f"{row['gpu']} c{row['conc']}: {', '.join(parts)}")

print()
print("=== B200 vs MI355 (clean total ms, typical conc 8/16/32) ===")
for conc in [8, 16, 32]:
    b200 = next((r for r in rows if r.get("gpu") == "b200" and r.get("conc") == conc and "error" not in r), None)
    mi = next((r for r in rows if r.get("gpu") == "mi355" and r.get("conc") == conc and "error" not in r), None)
    if not b200 or not mi:
        continue
    print(f"conc {conc}:")
    for label, key in [("draft", "draft"), ("draft_extend", "de"), ("target_verify", "tv")]:
        b = b200[key]["sum_clean"]
        m = mi[key]["sum_clean"]
        ratio = m / b if b else float("inf")
        print(f"  {label:14} B200={b:7.1f}ms  MI355={m:7.1f}ms  MI355/B200={ratio:.1f}x")
