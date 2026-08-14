"""Op-call cross-check from CUDA-graph-OFF traces.

Graph-ON traces replay a captured graph, so their CPU side says nothing about
which ops ran. The graph-OFF run of the same workload keeps the full
python_function / cpu_op tree, which is the only place to confirm both GPUs
call the same ops.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from .phases import merge_windows
from .trace import Trace


def op_calls_per_step(tr: Trace, min_dur_us: float = 0.0) -> dict:
    """-> {phase: {op_name: calls per decode step}} from CPU-side markers."""
    by_phase = defaultdict(list)
    for span in tr.cpu_spans:
        by_phase[span.phase].append(span)

    starts = [o[0] for o in tr.cpu_ops]
    out = {}
    for phase in tr.phases:
        windows = merge_windows(by_phase.get(phase, []))
        n = len(windows)
        if not n:
            out[phase] = {}
            continue
        counts = defaultdict(int)
        for start, end in windows:
            hi = bisect_right(starts, end)
            i = hi - 1
            while i >= 0 and starts[i] >= start:
                op_start, op_end, name = tr.cpu_ops[i]
                if op_end - op_start >= min_dur_us:
                    counts[name] += 1
                i -= 1
        out[phase] = {k: v / n for k, v in counts.items()}
    return out


def kernel_names_per_phase(tr: Trace) -> dict:
    """-> {phase: set(kernel names)} using GPU markers, for coverage checks."""
    by_phase = defaultdict(list)
    for span in tr.gpu_spans:
        by_phase[span.phase].append(span)

    out = {}
    for phase in tr.phases:
        windows = merge_windows(by_phase.get(phase, []))
        names = set()
        for start, end in windows:
            hi = bisect_right(tr.k_start, end)
            lo_bound = start - tr.k_max_dur
            i = hi - 1
            while i >= 0 and tr.k_start[i] >= lo_bound:
                if min(tr.k_end[i], end) - max(tr.k_start[i], start) > 0:
                    names.add(tr.k_names[i])
                i -= 1
        out[phase] = names
    return out
