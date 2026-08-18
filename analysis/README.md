# analysis/

DECODE / kernel analysis (non-MTP and MTP). Prefer the root README for suite layout and phases.

```bash
pip install -r requirements.txt

# Hierarchical Excel report
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --label nomtp --rules rules/glm52.csv -o out.xlsx

# Suite validation (optional --conc to check one concurrency only)
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --validate --mode nomtp --conc 4 \
  --expected-steps 20 --min-steps 15 --max-wall-ms 600
```

| Path | Role |
|------|------|
| `decode_profile/` | Core package (`single`, `sweep`, …) |
| `rules/` | Kernel category CSVs (e.g. `glm52.csv`) |
| `tools/` | Extra CLIs |
