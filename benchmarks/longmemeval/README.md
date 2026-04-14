# LongMemEval Benchmark Harness

Measures **session-level recall@K** for the agentssot `/knowledge/recall`
endpoint against the
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) dataset.

## Why this metric
LongMemEval's canonical scoring uses GPT-4o as an LLM judge on generated answers.
That conflates retrieval quality with generation quality and adds cost.
Session-level recall@K is cheaper and isolates retrieval: *did we surface at
least one of the sessions that actually contains the answer in the top K
results?* This is the metric our system's embedding + reranker + tiering
directly controls.

## Prerequisites
- agentssot API running (default `http://localhost:8088`)
- Writer API key exported as `SSOT_TEST_API_KEY`
- `wget`, `python3 -m pip install httpx pyyaml`

## Run
```bash
cd benchmarks/longmemeval
./download.sh oracle          # smallest variant (~evidence-only)
export SSOT_TEST_API_KEY=...
python runner.py --config vector_only --purge
python runner.py --config with_reranker --purge
python runner.py --config tiered_summary --skip-ingest
```

- `--purge` wipes the `bench_longmemeval` namespace before ingest. Requires the
  `/admin/delete-by-tag` endpoint (added alongside this harness).
- `--skip-ingest` reuses an already-ingested haystack, e.g. when flipping the
  server's reranker flag.
- `--limit N` truncates to N questions for smoke-tests.

## Configurations
See `config.yaml`:

| config            | reranker | layer_preference | notes                               |
|-------------------|----------|------------------|-------------------------------------|
| `vector_only`     | off      | full             | Baseline — flip `RERANKER_PROVIDER=disabled` in API |
| `with_reranker`   | on       | full             | `RERANKER_PROVIDER=ollama`          |
| `tiered_summary`  | —        | summary          | Serves L1 summaries instead of L2   |
| `tiered_abstract` | —        | abstract         | Serves L0 abstracts instead of L2   |

The reranker toggle is a *server-side* setting; the client can't flip it.
Restart the API between the vector-only and with-reranker runs.

## Results
Committed to `results/<config>.json` with per-question outcomes + summary
metrics. See `results/README.md` for the running baseline table.

## Known limitations
1. No LLM-judge QA accuracy yet. Add once retrieval baselines are solid.
2. Oracle variant recommended for first run (fastest). Use `./download.sh s` for
   the full ~115k-token haystack.
3. `--purge` relies on `/admin/delete-by-tag` which doesn't exist yet. Workaround:
   drop + recreate the `bench_longmemeval` namespace manually between runs, or
   use a unique `tag` per run in `config.yaml`.
