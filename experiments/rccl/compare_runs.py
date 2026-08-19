#!/usr/bin/env python3
"""Compare two experiments/rccl result directories (info + all_reduce_perf logs).

Prefer the slim bundles written by run_allreduce_bench.sh:

    python experiments/rccl/compare_runs.py ~/rccl_m11/compare ~/rccl_n10/compare

Full OUT_DIR also works if it still has info/ + bench/sweep.log.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# rccl-tests all_reduce_perf table rows typically start with size in bytes.
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_sweep(log: Path) -> dict[int, dict]:
    """-> {size_bytes: {time_us, algbw, busbw, ...}} from out-of-place columns when present."""
    out: dict[int, dict] = {}
    for line in read_text(log).splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        size = int(m.group(1))
        # Layout varies slightly by rccl-tests version; common:
        # size count type redop root time(us) algbw(GB/s) busbw(GB/s) ...
        # We keep positional fields and also try to find floats near the end.
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            # Prefer last two floats as algbw/busbw when present
            floats = []
            for tok in fields[4:]:
                try:
                    floats.append(float(tok))
                except ValueError:
                    continue
            if len(floats) < 2:
                continue
            time_us, algbw, busbw = floats[0], floats[1], floats[2] if len(floats) > 2 else floats[1]
            out[size] = {
                "time_us": time_us,
                "algbw": algbw,
                "busbw": busbw,
                "raw": line.strip(),
            }
        except (ValueError, IndexError):
            continue
    return out


def snip(path: Path, max_lines: int = 40) -> str:
    lines = read_text(path).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])


def diff_key_lines(a: str, b: str, keys: list[str]) -> list[str]:
    def pick(text: str) -> dict[str, str]:
        found = {}
        for line in text.splitlines():
            low = line.lower()
            for k in keys:
                if k.lower() in low and k not in found:
                    found[k] = line.strip()
        return found

    pa, pb = pick(a), pick(b)
    rows = []
    for k in keys:
        va, vb = pa.get(k, "<missing>"), pb.get(k, "<missing>")
        mark = " " if va == vb else "!"
        rows.append(f"{mark} [{k}]\n    A: {va}\n    B: {vb}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_a", type=Path, help="result dir A (e.g. m11)")
    ap.add_argument("dir_b", type=Path, help="result dir B (e.g. n10)")
    ap.add_argument(
        "--sizes",
        default="1048576,4194304,16777216,67108864",
        help="comma-separated byte sizes to highlight (default 1M,4M,16M,64M)",
    )
    args = ap.parse_args()

    a, b = args.dir_a.resolve(), args.dir_b.resolve()
    for d in (a, b):
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}")

    print(f"A = {a}")
    print(f"B = {b}")
    print()

    print("=== fingerprints (selected lines) ===")
    for rel in ("info/rocm.txt", "info/topo.txt", "info/rccl_libs.txt", "info/host.txt"):
        print(f"\n--- {rel} ---")
        for line in diff_key_lines(
            read_text(a / rel),
            read_text(b / rel),
            [
                "version",
                "ROCk",
                "Driver",
                "firmware",
                "VBIOS",
                "librccl",
                "hostname",
                "numa",
                "XGMI",
                "hops",
                "type",
            ],
        ):
            print(line)

    sa = parse_sweep(a / "bench" / "sweep.log")
    sb = parse_sweep(b / "bench" / "sweep.log")
    want = [int(x) for x in args.sizes.split(",") if x.strip()]

    print("\n=== all_reduce_perf sweep highlight ===")
    print(f"{'size':>12} {'A_busbw':>10} {'B_busbw':>10} {'B/A':>8} {'A_us':>10} {'B_us':>10}")
    sizes = sorted(set(sa) & set(sb))
    if not sizes:
        print("(no overlapping parsed rows — check bench/sweep.log format)")
    for size in sizes:
        if want and size not in want and size not in (1 << 20, 4 << 20, 16 << 20, 64 << 20):
            # still print all overlapping if few; else only highlights + powers of two near band
            if len(sizes) > 20 and size not in want:
                continue
        ra, rb = sa[size], sb[size]
        ratio = rb["busbw"] / ra["busbw"] if ra["busbw"] else float("nan")
        print(
            f"{size:12d} {ra['busbw']:10.3f} {rb['busbw']:10.3f} {ratio:8.3f} "
            f"{ra['time_us']:10.1f} {rb['time_us']:10.1f}"
        )

    print("\n=== fixed-size log tails ===")
    for name in ("fixed_1M.log", "fixed_4M.log", "fixed_16M.log", "fixed_64M.log"):
        print(f"\n-- {name} A --")
        print(snip(a / "bench" / name, 12) or "<missing>")
        print(f"-- {name} B --")
        print(snip(b / "bench" / name, 12) or "<missing>")

    print("\nDone. If B/A busbw << 1 on 1M–64M, suspect fabric/driver/topo on B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
