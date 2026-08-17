# serving/

Serving-side benches and env suites, organized by **engine** then **model**.

```
serving/<engine>/
  lib/                 # talk to a live server (perf/acc/profile client)
  recipes/             # how to launch the server (per model)
  suites/<model>/      # what sequence / which IOs to run
  misc/                # stray / not part of formal suites
```

Stage vocabulary used by suites: **acc → perf → trace → analyze**.

See [`sglang/`](sglang/) for the first engine.
