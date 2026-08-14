import gzip
import json
import os
import re

base = os.path.dirname(os.path.abspath(__file__))

for fn in sorted(os.listdir(base)):
    if not fn.endswith(".trace.json.gz"):
        continue
    m = re.match(r"glm_mtp_glm_(b200|mi355)_i70000_c(\d+)", fn)
    if not m:
        continue
    try:
        ev = json.load(gzip.open(os.path.join(base, fn), "rt"))["traceEvents"]
    except (EOFError, OSError, json.JSONDecodeError):
        print(m.group(1), "c" + m.group(2), "SKIP")
        continue
    ts = []
    for e in ev:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        n = e.get("name", "")
        if n in ("draft", "draft_extend") or "TARGET_VERIFY" in n:
            ts.append(e["ts"])
            ts.append(e["ts"] + e["dur"])
    if ts:
        wall = (max(ts) - min(ts)) / 1000
        print(f"{m.group(1)} c{m.group(2)}: profile wall-clock span = {wall:.1f} ms")
