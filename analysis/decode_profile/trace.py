"""Trace loading and GPU stream detection."""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

MTP_PHASES = ("draft", "target_verify", "draft_extend")
PLAIN_PHASES = ("decode",)

# Back-compat alias: callers that predate non-MTP support import PHASES.
PHASES = MTP_PHASES

BS_RE = re.compile(r"bs=(\d+)")


# Outer phase markers only. Inner markers such as ``step[DRAFT_EXTEND_V2 bs=4]``
# nest inside these, so matching them too would count the same GPU time twice.
# ``step[DECODE bs=N]`` is the whole step in a non-MTP run but only an inner
# marker in an MTP run, so it is kept separate and used only when no MTP marker
# is present.
def phase_of(name: str):
    if name == "draft":
        return "draft"
    if name == "draft_extend":
        return "draft_extend"
    if "TARGET_VERIFY" in name:
        return "target_verify"
    if name.startswith("step[DECODE"):
        return "decode"
    return None


@dataclass
class Span:
    phase: str
    tid: int
    start: float
    end: float

    @property
    def dur_ms(self) -> float:
        return (self.end - self.start) / 1000.0


@dataclass
class Trace:
    label: str
    path: str
    # GPU phase markers (cat=gpu_user_annotation), never the CPU-side ones:
    # mixing the two produced a spurious bimodal distribution in earlier runs.
    gpu_spans: list = field(default_factory=list)
    cpu_spans: list = field(default_factory=list)
    # Kernels sorted by start, kept as parallel arrays for cheap window scans.
    k_names: list = field(default_factory=list)
    k_tids: list = field(default_factory=list)
    k_start: list = field(default_factory=list)
    k_end: list = field(default_factory=list)
    k_max_dur: float = 0.0
    kernel_tid_counts: dict = field(default_factory=dict)
    cpu_ops: list = field(default_factory=list)
    event_total: int = 0
    bs_counts: dict = field(default_factory=dict)

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)

    @property
    def batch_size(self):
        """Running batch the profile captured, read from the ``bs=N`` markers."""
        if not self.bs_counts:
            return None
        return max(self.bs_counts, key=self.bs_counts.get)

    @property
    def is_mtp(self) -> bool:
        return any(s.phase in MTP_PHASES for s in self.gpu_spans)

    @property
    def phases(self) -> tuple:
        """Phase names this trace is segmented into, in execution order."""
        return MTP_PHASES if self.is_mtp else PLAIN_PHASES

    @property
    def phase_tids(self) -> list:
        phases = set(self.phases)
        return sorted({s.tid for s in self.gpu_spans if s.phase in phases})

    @property
    def main_tid(self):
        """Stream carrying the most kernels; only used for reporting."""
        if not self.kernel_tid_counts:
            return None
        return max(self.kernel_tid_counts, key=self.kernel_tid_counts.get)


def load_trace(path: str, label: str, want_cpu_ops: bool = False) -> Trace:
    """Read a PyTorch profiler trace, keeping only what the analysis needs."""
    with gzip.open(path, "rt") as fh:
        events = json.load(fh)["traceEvents"]

    tr = Trace(label=label, path=path, event_total=len(events))
    kernels = []
    kernel_tids = Counter()
    bs_counts = Counter()

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat == "gpu_user_annotation":
            name = e.get("name", "")
            m = BS_RE.search(name)
            if m:
                bs_counts[int(m.group(1))] += 1
            ph = phase_of(name)
            if ph:
                ts = e["ts"]
                tr.gpu_spans.append(Span(ph, e.get("tid"), ts, ts + e.get("dur", 0)))
        elif cat == "user_annotation":
            ph = phase_of(e.get("name", ""))
            if ph:
                ts = e["ts"]
                tr.cpu_spans.append(Span(ph, e.get("tid"), ts, ts + e.get("dur", 0)))
        elif cat == "kernel":
            dur = e.get("dur", 0)
            if dur <= 0:
                continue
            ts = e["ts"]
            tid = e.get("tid")
            kernel_tids[tid] += 1
            kernels.append((ts, ts + dur, tid, e.get("name", "")))
        elif want_cpu_ops and cat == "cpu_op":
            ts = e["ts"]
            tr.cpu_ops.append((ts, ts + e.get("dur", 0), e.get("name", "")))

    kernels.sort(key=lambda k: k[0])
    tr.k_start = [k[0] for k in kernels]
    tr.k_end = [k[1] for k in kernels]
    tr.k_tids = [k[2] for k in kernels]
    tr.k_names = [k[3] for k in kernels]
    tr.k_max_dur = max((e - s for s, e in zip(tr.k_start, tr.k_end)), default=0.0)
    tr.kernel_tid_counts = dict(kernel_tids)
    tr.bs_counts = dict(bs_counts)
    tr.cpu_ops.sort(key=lambda o: o[0])
    return tr
