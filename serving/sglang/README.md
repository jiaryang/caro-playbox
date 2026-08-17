# serving/sglang

SGLang engine tooling.

| Dir | Purpose |
|-----|---------|
| [`lib/`](lib/) | `sweep_bench.sh` + summarizers (client only) |
| [`recipes/`](recipes/) | Server arg snippets (`glm.sh`, …) |
| [`suites/`](suites/) | Model orchestrators (`glm/run_env_suite.sh`, …) |
| [`misc/`](misc/) | Not wired into suites (e.g. DeepSeek CI copy) |

```bash
# Full GLM env suite
bash suites/glm/run_env_suite.sh --dry-run --only-nomtp

# Standalone perf against an already-running server
bash lib/perf_bench.sh --model-key glm \
  --perf-io-pairs 8192:1024 --server-host 127.0.0.1 --server-port 30000
```
