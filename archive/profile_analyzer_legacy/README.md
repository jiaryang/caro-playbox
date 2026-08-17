# legacy

Pre-`decode_profile` one-off scripts and text reports. Hardcoded stream ids /
paths often no longer match current traces.

Prefer:

```bash
PYTHONPATH=. python -m decode_profile.single --dir <profiles> --label run -o out.xlsx
```
