#!/bin/bash

rm rpd_tracer_output_trace.*

python tf_gemm.py

python /dockerx/PerformanceCorrelationDebugging/prof_tools/summarize_kernels_rpd.py tf_trace.rpd
