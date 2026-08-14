"""Kernel-level analysis of SGLang decode profiles, MTP and plain.

Timing comes from CUDA-graph-ON traces; graph-OFF traces are only read to
cross-check which ops each side calls.
"""

from .trace import MTP_PHASES, PHASES, PLAIN_PHASES, Trace, load_trace
from .phases import PhaseStats, analyze_phases, phase_order
from .kernels import KernelClassifier

__all__ = [
    "MTP_PHASES",
    "PLAIN_PHASES",
    "PHASES",
    "Trace",
    "load_trace",
    "PhaseStats",
    "analyze_phases",
    "phase_order",
    "KernelClassifier",
]
