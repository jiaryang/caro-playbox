#!/bin/bash

rm rpd_tracer_output_trace.*

python gemm_jax.py

python /dockerx/PerformanceCorrelationDebugging/prof_tools/summarize_kernels_rpd.py rpd_tracer_output_trace.rpd
