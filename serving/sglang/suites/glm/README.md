# GLM suite

```bash
bash run_env_suite.sh
bash run_env_suite.sh --skip-acc --only-nomtp
bash run_env_suite.sh --from-phase profile --suite-dir ./suite_glm_env_<ts>
```

## Stages

1. **accuracy** — GSM8K (`lib` acc)
2. **perf** — short then long (see matrix)
3. **trace** — DECODE collect + validate (**8k only**)
4. **analyze** — `analysis/decode_profile.single` → xlsx

Modes: **nomtp** then **mtp** (or `--only-*`).

## IO × responsibility

| IO | Server | Perf | Trace | Analyze |
|----|--------|------|-------|---------|
| `1024:1024` | baseline | yes | no | no |
| `8192:1024` | baseline | yes | yes | yes |
| `70000:300` | `--max-running-requests` (default 8) | yes | no | no |

Override with `--short-io`, `--long-io`, `--trace-io`.

## Outputs

Under `suite_glm_env_<ts>/`: `acc/`, `perf/`, `profiles/` (8k), `analyze/`, `manifest.txt`.
