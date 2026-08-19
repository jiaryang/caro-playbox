# RCCL / fabric probe

Compare **intra-node allreduce** health across machines (e.g. two MI355 nodes
running the same SGLang TP4 docker). Same image does **not** guarantee the same
RCCL performance — host driver, firmware, XGMI topology, and clocks matter.

This tool:

1. Collects host / ROCm / topology / library fingerprints
2. Builds or **reuses** [ROCm/rccl-tests](https://github.com/ROCm/rccl-tests) from a shared cache
3. Runs `all_reduce_perf` with configs aligned to GLM TP4 workloads
4. Writes a slim **`OUT_DIR/compare/`** bundle for cross-node compare

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

# force rebuild of rccl-tests (normally skipped if binary exists)
FORCE_BUILD=1 bash experiments/rccl/run_allreduce_bench.sh
```

Compare two machines (share the **compare** folder only):

```bash
# on m11-13
OUT_DIR=~/rccl_m11 bash experiments/rccl/run_allreduce_bench.sh

# on n10-17
OUT_DIR=~/rccl_n10 bash experiments/rccl/run_allreduce_bench.sh

# anywhere with both compare trees
python experiments/rccl/compare_runs.py ~/rccl_m11/compare ~/rccl_n10/compare
```

## Output layout

```
OUT_DIR/
  manifest.txt summary.txt
  info/          # fingerprints
  bench/         # sweep.log, fixed_*.log, optional build.log
  compare/       # <-- share this (~100KB): logs + info only
```

`compare/` excludes `src/`, `all_reduce_perf` binary, and `build.log`.

## Source / binary cache (no rebuild every run)

Clone + build live under **`experiments/rccl/.cache/rccl-tests/`** (gitignored),
not inside each `OUT_DIR`.

| Situation | Behavior |
|-----------|----------|
| Binary already in `.cache/.../build/all_reduce_perf` | reuse, skip clone/make |
| First run / cache empty | clone once + `make MPI=0` |
| Want a clean rebuild | `FORCE_BUILD=1` |
| Bring your own binary | `RCCL_TESTS_DIR=/path/to/dir` (must contain `all_reduce_perf`) |
| Refuse to compile | `SKIP_BUILD=1` (fails if binary missing) |

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
| `OUT_DIR` | `results/rccl_bench_<host>_<ts>` | result root |
| `CACHE_DIR` | `experiments/rccl/.cache/rccl-tests` | shared clone+build tree |
| `RCCL_TESTS_DIR` | (empty) | dir containing prebuilt `all_reduce_perf` |
| `RCCL_TESTS_SRC` | `$CACHE_DIR` | git source override |
| `SKIP_BUILD` | `0` | fail if binary missing (no compile) |
| `FORCE_BUILD` | `0` | rebuild even when cache hit |
| `COPY_BINARY` | `0` | also copy binary into `bench/` (off by default) |
| `SWEEP_ARGS` | see script | override sweep CLI |
| `FIXED_SIZES` | `1M 4M 16M 64M` | space-separated fixed sizes |

## Interpreting results

- **n10 busbw ≪ m11** on 1M–64M → treat as machine/fabric/driver issue
- **busbw ≈ equal** but SGLang EXTEND AR still diverges → not plain RCCL; check quickreduce / thermal under real load
- Always diff `info/rocm.txt` + `info/topo.txt` first — version-identical containers still differ on host ROCK/firmware/topo
