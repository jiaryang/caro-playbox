# analysis/

DECODE / kernel analysis (non-MTP and MTP). Prefer the root README for the overall map.

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --label nomtp --rules rules/glm52.csv -o out.xlsx

# Suite validation path
PYTHONPATH=. python -m decode_profile.single \
  --dir /path/to/profiles --validate --mode nomtp \
  --expected-steps 20 --min-steps 15 --max-wall-ms 600
```

| Path | Role |
|------|------|
| `decode_profile/` | Core package |
| `rules/` | Kernel category CSVs |
| `tools/` | Extra CLIs |
