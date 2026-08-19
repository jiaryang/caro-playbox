# RCCL / fabric probe

Compare **intra-node allreduce** health across machines (e.g. two MI355 nodes
running the same SGLang TP4 docker). Same image does **not** guarantee the same
RCCL performance — host driver, firmware, XGMI topology, and clocks matter.

This tool:

1. Collects host / ROCm / topology / library fingerprints
2. Builds or reuses [ROCm/rccl-tests](https://github.com/ROCm/rccl-tests)
3. Runs `all_reduce_perf` with configs aligned to GLM TP4 workloads

> **Caveat:** SGLang GLM EXTEND on MI355 often uses **quickreduce** AllReduce
> (not plain `rcclAllReduce`). These microbenches are a **fabric / driver proxy**.
> If busbw matches across nodes but SGLang EXTEND AR still differs, dig into
> quickreduce / aiter paths next.

## Quick start

Inside the **same docker** you use for SGLang (ROCm + HIP):

```bash
# default GPUs match GLM suite: 4,5,6,7
bash experiments/rccl/run_allreduce_bench.sh

# explicit GPUs / output dir
GPUS=4,5,6,7 OUT_DIR=/tmp/rccl_m11 \
  bash experiments/rccl/run_allreduce_bench.sh

# info only (no compile / no bench)
bash experiments/rccl/run_allreduce_bench.sh --info-only

# reuse a prebuilt binary
RCCL_TESTS_DIR=/path/to/rccl-tests/build \
  bash experiments/rccl/run_allreduce_bench.sh
```

Compare two machines:

```bash
# on m11-13
OUT_DIR=~/rccl_m11 bash experiments/rccl/run_allreduce_bench.sh

# on n10-17
OUT_DIR=~/rccl_n10 bash experiments/rccl/run_allreduce_bench.sh

# anywhere with both result trees
python experiments/rccl/compare_runs.py ~/rccl_m11 ~/rccl_n10
```

## What gets collected (`info/`)

| File | Contents |
|------|----------|
| `host.txt` | hostname, uname, date, CPU model, meminfo |
| `rocm.txt` | `/opt/rocm/.info/version`, `rocm-smi` product/driver/fw, ROCK module |
| `topo.txt` | `rocm-smi --showtopo` / `amd-smi topology` if present |
| `gpus.txt` | `rocm-smi` snapshot (clocks/temp/power) at start |
| `rccl_libs.txt` | `librccl.so*` paths + checksums (container libs) |
| `env.txt` | `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, key `NCCL_*` / `RCCL_*` |
| `docker.txt` | cgroup / `.dockerenv` / visible devices (best-effort) |
| `host_tuning.txt` | numa_balancing, brief GRUB/iommu notes (best-effort) |

## Bench configs (`bench/`)

Default (override with env):

| Run | Flags | Why |
|-----|-------|-----|
| `sweep` | `-b 8 -e 128M -f 2 -g N -n 50 -w 20` | full bandwidth curve |
| `fixed_1M` / `4M` / `16M` / `64M` | fixed size, `-n 100` | band near ~0.5–1 ms AR latency seen in GLM EXTEND |

`N` = number of GPUs from `GPUS` (default 4).

## Env knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `GPUS` | `4,5,6,7` | visible device list |
| `OUT_DIR` | `./rccl_bench_<host>_<ts>` | result root |
| `RCCL_TESTS_DIR` | (build under `OUT_DIR/rccl-tests`) | dir containing `all_reduce_perf` |
| `RCCL_TESTS_SRC` | clone to `OUT_DIR/src/rccl-tests` | git source override |
| `SKIP_BUILD` | `0` | set `1` if binary already on `PATH` / `RCCL_TESTS_DIR` |
| `SWEEP_ARGS` | see script | override sweep CLI |
| `FIXED_SIZES` | `1M 4M 16M 64M` | space-separated fixed sizes |

## Interpreting results

- **n10 busbw ≪ m11** on 1M–64M → treat as machine/fabric/driver issue
- **busbw ≈ equal** but SGLang EXTEND AR still diverges → not plain RCCL; check quickreduce / thermal under real load
- Always diff `info/rocm.txt` + `info/topo.txt` first — version-identical containers still differ on host ROCK/firmware/topo
