# profile_analyzer

Kernel-level comparison of SGLang decode profiles across GPUs.

`mtp_profile/` is the current tool. `legacy/` holds the earlier one-off scripts
and their outputs; they hardcode stream ids that no longer match newer traces.

Both speculative and plain runs are handled. The phase set comes from the trace
itself: `draft` / `target_verify` / `draft_extend` when the MTP markers are
present, otherwise a single `decode` phase read from `step[DECODE bs=N]`.

## What it measures

Per decode step, for each phase:

| metric | meaning |
|--------|---------|
| wall | median of the phase window, unioned across every GPU stream that carries the marker |
| kernel | kernel time summed over all streams inside those windows |
| overlap factor | kernel / wall; above 1.0 means streams ran concurrently |
| non-kernel gap | wall - kernel; GPU idle inside the phase |

Both numbers are needed. B200 runs `target_verify` on two streams, so its
kernel sum exceeds its wall clock; MI355 spends time in `draft_extend` with no
kernel running at all. Comparing only kernel sums hides both effects.

## Usage

```bash
python -m mtp_profile \
    --a MI355 mi355_graphON_DECODE.trace.json.gz \
    --b B200  b200_graphON_DECODE.trace.json.gz \
    --graph-off-a mi355_graphOFF_DECODE.trace.json.gz \
    --graph-off-b b200_graphOFF_DECODE.trace.json.gz \
    -o report.xlsx
```

Timing always comes from the CUDA-graph-ON traces. The graph-OFF traces are
optional and only feed the `GraphOFF_Ops` sheet: a graph replay says nothing on
the CPU side about which ops ran, so confirming that both GPUs call the same
ops needs the run that kept its `cpu_op` tree.

## Sheets

| sheet | content |
|-------|---------|
| README | trace paths, detected streams, method |
| PerStep_Summary | one row per GPU plus diff and ratio rows |
| Phase_Diff | per phase wall / kernel / overlap / gap for both sides |
| Kernel_Category_Diff | per phase and category, ms/step and diff |
| Top_Kernels | top kernels per phase per GPU with calls/step |
| Streams | per-stream kernel time, showing which streams carry phase markers |
| GraphOFF_Ops | op calls/step per phase from the graph-OFF traces |

All values are ms per decode step, rounded to two decimals.

## Concurrency sweep

`mtp_profile.sweep` runs the same analysis over a whole profile directory and
makes concurrency a dimension of the report. It pairs the traces by the `_c<N>-`
tag in the filename and reads the running batch out of the `bs=N` markers, so a
run whose batch never reached the requested concurrency is visible rather than
silently compared against a different batch.

```bash
python -m mtp_profile.sweep \
    --a MI355 profiles_8k_mi355 \
    --b B200  profiles_8k_b200 \
    -o sweep.xlsx
```

| sheet | content |
|-------|---------|
| BatchSize_Check | requested concurrency vs the batch each side actually ran |
| Decode_Scaling | decode totals per concurrency, plus wall per request and growth vs the smallest concurrency |
| Phase_Wall_ByConc | phase wall time with concurrency across the columns |
| Phase_Scaling | full per-phase breakdown for every concurrency |
| TargetVerify_Categories | category ms/step for `target_verify`, concurrency across the columns |
| Kernel_Category_Scaling | every concurrency, phase and category |
| Top_Kernels / Streams | as above, per concurrency |

## MTP against a plain run

`compare_modes.py` answers whether speculative decoding pays for itself. An MTP
step is longer but emits `accept_length` tokens per request instead of one, so
it pairs the traces of both runs with the client benchmark summaries that carry
that number and reports output tokens per second.

```bash
python compare_modes.py --mtp-root 3_0811_MTP --nonmtp-root 3_0811_nonMTP -o modes.xlsx
```

Profile directories are discovered as `profiles_<context>_<gpu>[_<variant>]` and
matched against `perf_glm_<mode>_<gpu>[_<variant>]_*/summary_<context>.csv`. The
`Mode_Gain` sheet holds the step cost, the accepted length and the resulting
gain both GPU-side and as measured end to end.

## Kernel categories

`kernel_categories_glm52.csv` is the default: GLM-5.2 functional buckets ported
from `SGLang-benchmarks/trace_analysis/compare/glm52_buckets.py`, so MI355 and
B200 kernels with different names land in the same bucket. First matching row
wins, so row order is the priority order.

Two mappings are worth knowing because the names suggest otherwise:

- `nvjet_*` is cuBLAS on Blackwell and sits under attention projections, the
  dense MLP and the MoE shared expert, so it is dense GEMM, not MoE GEMM.
- B200's MoE experts are `bmm_E2m1_*` (up/gate) and `bmm_Bfloat16_E2m1*`
  (down), matching MI355's `gemm1_a4w4` / `gemm2_a4w4`.

`kernel_categories.csv` is the broader non-GLM rule set from
torch-profiler-parser, selectable with `--rules`.
