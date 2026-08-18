# caro-playbox

AMD / ROCm sandbox: env setup, SGLang suites, DECODE analysis, and one-off experiments.

```
env/                 host / conda / TheRock / PyTorch
sglang/
  lib/               client: perf / acc / profile (talks to a live server)
  recipes/           server launch args (per model)
  suites/<model>/    orchestrator (IO matrix, phases)
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
bash sglang/suites/glm/run_env_suite.sh --help
```

Phases: **acc → perf → trace (8k only) → analyze**. Modes: nomtp then mtp.

| IO | Perf | Trace / analyze |
|----|------|-----------------|
| `1024:1024` | yes | no |
| `8192:1024` | yes | yes |
| `70000:300` (+ `--max-running-requests`) | yes | no |

Outputs under `sglang/suites/glm/suite_glm_env_<ts>/`.

## Analyze traces

```bash
cd analysis
pip install -r requirements.txt
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --label nomtp --rules rules/glm52.csv -o out.xlsx
```

Works for both plain (`decode`) and MTP (`draft` / `target_verify` / `draft_extend`) traces.

## Extend

| Add | Where |
|-----|--------|
| Model | `sglang/recipes/<model>.sh` + `sglang/suites/<model>/` |
| Kernel rules | `analysis/rules/<model>.csv` |

If another serving engine shows up later, introduce a parent dir then — no need to invent `serving/<engine>/` early.
