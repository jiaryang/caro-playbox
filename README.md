# caro-playbox

AMD / ROCm sandbox: env setup, SGLang suites, DECODE analysis, and one-off experiments.

```
env/                 host / conda / TheRock / PyTorch
sglang/
  lib/               client helpers: sweep_bench, summarize_perf/acc
  recipes/           server launch args (per model)
  suites/<model>/    orchestrator (phases + IO matrix)
analysis/
  decode_profile/    python -m decode_profile.single | .sweep
  rules/             kernel category CSVs
  tools/             extra CLIs
experiments/         HIP / pyt / jax / tf / hf / openai / scratch
archive/             deprecated scripts & old reports
```

## Run GLM suite

```bash
bash sglang/suites/glm/run_env_suite.sh
bash sglang/suites/glm/run_env_suite.sh --only-nomtp --skip-long-ctx
bash sglang/suites/glm/run_env_suite.sh --suite-dir ... --phases profile,analyze
bash sglang/suites/glm/run_env_suite.sh --help
```

Default: **GPUS=4,5,6,7**, TP=4, model `amd/GLM-5.2-MXFP4`.

### Phases (per mode: nomtp → mtp)

| Phase | What |
|-------|------|
| **acc** | GSM8K |
| **perf** | short IO (`1024:1024`, `8192:1024`) then long IO (`70000:300` with `--max-running-requests 8`) |
| **profile** | 8k DECODE traces |
| **analyze** | hierarchical Excel under `analyze/` |

Select with `--phases` (default `acc,perf,profile,analyze`). Order among selected stages is always acc → perf → profile → analyze.

```bash
# resume profiles + analyze on an existing suite
bash sglang/suites/glm/run_env_suite.sh --suite-dir suite_glm_env_<ts> --phases profile,analyze

# re-analyze only
bash sglang/suites/glm/run_env_suite.sh --suite-dir suite_glm_env_<ts> --phases analyze
```

`--dry-run` prints the plan only (no suite dir / manifest).
### Suite output layout

```
sglang/suites/glm/suite_glm_env_<ts>/
  manifest.txt
  suite.log
  acc/{nomtp,mtp}/
  perf/{nomtp,mtp}/          # all IOs together → one summary
    glm_<mode>_<ilen>_o<olen>_c<conc>.{jsonl,log}
    summary.txt | summary.xlsx | summary_<ilen>.csv
    sweep.log
  profiles/{nomtp,mtp}/
    i8192_o1024/             # cuda-graph ON, conc 4–64
    i8192_o1024_c4_nocg/     # cuda-graph OFF, conc 4
  profile_sweep_logs/...
  analyze/                   # decode_profile Excel
```

Perf report columns (CSV / Excel):

`Interactivity (tok/s/user)` · `Token TPUT per GPU` · `MedianTTFT` · `MedianTPOT` · `MedianITL`  
(+ `accept_length` for MTP)

### Profile

Tag is `i8192_o1024`; actual profile `output_len` is **64 nomtp / 128 mtp**, `num_prompts=conc×2`, `profile-num-steps=20`.

- cuda-graph ON, conc 4,8,16,32,64 → `profiles/{mode}/i8192_o1024/`
- cuda-graph OFF, conc=4 → `profiles/{mode}/i8192_o1024_c4_nocg/` (`--skip-nocg-profile` to skip)

Watchdog polls server health every 10s (`--profile-watchdog-sec 0` disables poll only) and applies a per-conc wall timeout (`max(600, conc*30)` s; **nocg auto 2x**). On trip: kill client, drop that conc's traces, restart server, retry (default retries=2).

| IO | Perf | Trace / analyze |
|----|------|-----------------|
| `1024:1024` | yes | no |
| `8192:1024` | yes | yes (cg ON 4–64 + cg OFF c4) |
| `70000:300` (+ `--max-running-requests`) | yes | no |

## Analyze traces

```bash
cd analysis
pip install -r requirements.txt
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --label nomtp --rules rules/glm52.csv -o out.xlsx
```

Works for plain (`decode`) and MTP (`draft` / `target_verify` / `draft_extend`) traces. See also `analysis/README.md`.

## Extend

| Add | Where |
|-----|--------|
| Model | `sglang/recipes/<model>.sh` + `sglang/suites/<model>/` |
| Kernel rules | `analysis/rules/<model>.csv` |

If another serving engine shows up later, introduce a parent dir then — no need to invent `serving/<engine>/` early.
