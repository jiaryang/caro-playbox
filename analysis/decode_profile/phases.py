"""Phase segmentation and per-step timing.

A phase can be annotated on several GPU streams at once (B200 runs
target_verify on two), so windows are merged across streams before anything is
measured. Kernel time is then summed over every stream inside those windows,
which is what makes the kernel sum exceed the wall clock when streams overlap.

Aggregation is mean over steady-state windows only. Outlier windows (wall >
``OUTLIER_WALL_FACTOR`` × median wall) are dropped first so a single stalled
all-reduce cannot inflate mean kernel while median wall stayed small.
"""

from __future__ import annotations

import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

from .trace import Trace

# Drop windows whose wall exceeds this × median(wall). 3× is enough to keep
# mild jitter (e.g. 47 vs 41 ms) while rejecting stalls (289 ms, 4 s).
OUTLIER_WALL_FACTOR = 3.0


@dataclass
class PhaseStats:
    phase: str
    n_steps: int
    wall_ms: float = 0.0
    kernel_ms: float = 0.0
    n_outliers: int = 0
    per_stream_ms: dict = field(default_factory=dict)
    per_kernel_ms: dict = field(default_factory=dict)
    per_kernel_calls: dict = field(default_factory=dict)
    per_category_ms: dict = field(default_factory=dict)
    windows: list = field(default_factory=list)

    @property
    def overlap_factor(self) -> float:
        return self.kernel_ms / self.wall_ms if self.wall_ms else 0.0

    @property
    def non_kernel_gap_ms(self) -> float:
        return self.wall_ms - self.kernel_ms


def get_phase_stats(stats: dict, phase: str) -> PhaseStats:
    """Safe lookup for A/B compares when phase sets differ (MTP vs nomtp)."""
    st = stats.get(phase)
    if st is None:
        return PhaseStats(phase=phase, n_steps=0)
    return st


def merge_windows(spans) -> list:
    """Union overlapping spans of one phase into one window per decode step."""
    intervals = sorted((s.start, s.end) for s in spans)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _kernels_in_window(tr: Trace, start: float, end: float):
    """Yield (index, overlap_us) for kernels intersecting the window."""
    hi = bisect_right(tr.k_start, end)
    lo_bound = start - tr.k_max_dur
    i = hi - 1
    while i >= 0 and tr.k_start[i] >= lo_bound:
        k_s, k_e = tr.k_start[i], tr.k_end[i]
        overlap = min(k_e, end) - max(k_s, start)
        if overlap > 0:
            yield i, overlap
        i -= 1


def _keep_window_indices(walls_us, factor: float = OUTLIER_WALL_FACTOR) -> list[int]:
    """Indices whose wall is within ``factor`` × median wall (always keep ≥1)."""
    if not walls_us:
        return []
    med = statistics.median(walls_us)
    if med <= 0:
        return list(range(len(walls_us)))
    thresh = med * factor
    kept = [i for i, w in enumerate(walls_us) if w <= thresh]
    return kept if kept else [min(range(len(walls_us)), key=lambda i: walls_us[i])]


def analyze_phases(tr: Trace, classifier) -> dict:
    """-> {phase: PhaseStats} using GPU-side markers from every stream."""
    by_phase = defaultdict(list)
    for span in tr.gpu_spans:
        by_phase[span.phase].append(span)

    results = {}
    for phase in tr.phases:
        windows = merge_windows(by_phase.get(phase, []))
        n_raw = len(windows)
        stats = PhaseStats(phase=phase, n_steps=n_raw, windows=windows)
        if not n_raw:
            results[phase] = stats
            continue

        walls_us = [end - start for start, end in windows]
        kept = _keep_window_indices(walls_us)
        stats.n_outliers = n_raw - len(kept)
        n = len(kept)  # timing divisor; n_steps stays raw for validate

        per_stream = defaultdict(float)
        per_kernel = defaultdict(float)
        per_calls = defaultdict(int)
        per_category = defaultdict(float)
        total_wall_us = 0.0
        total_kern_us = 0.0
        for idx in kept:
            start, end = windows[idx]
            total_wall_us += walls_us[idx]
            for i, overlap in _kernels_in_window(tr, start, end):
                name = tr.k_names[i]
                total_kern_us += overlap
                per_stream[tr.k_tids[i]] += overlap
                per_kernel[name] += overlap
                per_calls[name] += 1
                per_category[classifier.classify(name)] += overlap

        stats.wall_ms = total_wall_us / 1000.0 / n
        stats.kernel_ms = total_kern_us / 1000.0 / n
        stats.per_stream_ms = {t: v / 1000.0 / n for t, v in per_stream.items()}
        stats.per_kernel_ms = {k: v / 1000.0 / n for k, v in per_kernel.items()}
        stats.per_kernel_calls = {k: v / n for k, v in per_calls.items()}
        stats.per_category_ms = {c: v / 1000.0 / n for c, v in per_category.items()}
        results[phase] = stats

    return results


def decode_totals(stats: dict) -> tuple:
    """-> (wall ms/step, kernel ms/step) summed over every phase of the step."""
    wall = sum(st.wall_ms for st in stats.values())
    kernel = sum(st.kernel_ms for st in stats.values())
    return wall, kernel


def phase_order(*stats_dicts) -> list:
    """Ordered union of the phases present, first-seen order wins."""
    order = []
    for stats in stats_dicts:
        for phase in stats:
            if phase not in order:
                order.append(phase)
    return order
