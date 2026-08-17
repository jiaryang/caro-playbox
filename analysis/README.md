# analysis/

Engine-agnostic DECODE / kernel analysis for **non-MTP and MTP** traces.

| Path | Role |
|------|------|
| `decode_profile/` | Core package (`python -m decode_profile.single` / `.sweep` / compare CLI) |
| `rules/` | Kernel category CSVs (`glm52.csv`, `kernel_categories.csv`) |
| `tools/` | Extra CLIs (`compare_modes`, `category_matrix`, …) |

`decode_profile` names the measurement target (CUDA-graph decode steps), not
“MTP only”. Plain runs expose a `decode` phase; MTP runs expose
`draft` / `target_verify` / `draft_extend`.

```bash
cd /dockerx/caro-playbox/analysis
pip install -r requirements.txt
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --label nomtp --rules rules/glm52.csv -o out.xlsx

# Validate traces (used by GLM suite)
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --validate --mode nomtp \
  --expected-steps 20 --min-steps 15 --max-wall-ms 600
```

Legacy one-offs live in [`../archive/profile_analyzer_legacy/`](../archive/profile_analyzer_legacy/).
