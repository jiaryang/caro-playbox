# caro-playbox

Personal AMD / ROCm sandbox, laid out for multiple **serving engines** and
**models** without flattening everything at the repo root.

## Layout

| Path | Role |
|------|------|
| [`env/`](env/) | Host / conda / TheRock / PyTorch setup |
| [`serving/`](serving/) | Perf / accuracy / profile orchestration (`<engine>/<model>`) |
| [`analysis/`](analysis/) | DECODE trace analysis (`decode_profile`, rules, tools) |
| [`experiments/`](experiments/) | One-off HIP / PyTorch / JAX / TF / HF / OpenAI toys |
| [`archive/`](archive/) | Deprecated scripts and old reports |

## Active GLM workflow

```bash
bash serving/sglang/suites/glm/run_env_suite.sh

# Stages: acc → perf (1k + 8k + 70k) → DECODE only for 8k → analyze
# Modes: nomtp then mtp
```

Workload matrix (documented in the suite README):

| IO | Perf | Trace / analyze |
|----|------|-----------------|
| 1024:1024 | yes | no |
| 8192:1024 | yes | yes |
| 70000:300 (+ max-running-requests) | yes | no |

```bash
cd analysis && PYTHONPATH=. python -m decode_profile.single \
  --dir ../serving/sglang/suites/glm/suite_glm_env_<ts>/profiles/nomtp/i8192_o1024 \
  --label nomtp --rules rules/glm52.csv -o out.xlsx
```

## Adding a model (same engine)

1. `serving/sglang/recipes/<model>.sh`
2. `serving/sglang/suites/<model>/run_env_suite.sh` (+ own IO/trace matrix)
3. Optional `analysis/rules/<model>.csv`

## Adding an engine

Create `serving/<engine>/{lib,recipes,suites/<model>}/` with the same shape.
Analysis stays shared under `analysis/`.
