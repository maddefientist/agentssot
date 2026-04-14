# Baseline Results

Runs committed here so we can track regressions across feature work.

## LongMemEval Oracle

| Date | Config | R@1 | R@5 | R@10 | p50 ms | p95 ms | Notes |
|------|--------|-----|-----|------|--------|--------|-------|
| 2026-04-14 | vector_only | 28.6% | 54.6% | 66.4% | 24.7 | 51.5 | Baseline: nomic-embed-text 768d, pgvector cosine, 940 sessions |
| 2026-04-14 | with_reranker | 28.6% | 54.6% | 66.4% | 23.2 | 49.6 | Same scores - tiered endpoint doesn't use reranker yet |

## Notes

- **Truncation**: Sessions truncated to 7500 chars to stay within nomic-embed-text's 2048 token limit
- **Reranker gap**: The `/api/v1/knowledge/recall` endpoint doesn't use the Qwen3 reranker yet. Adding reranker support would likely boost R@K scores.
- **Comparison to MemPalace**: Their published 96.6% R@5 uses raw mode on the full dataset. Our 54.6% R@5 on Oracle variant isn't directly comparable due to truncation and different retrieval architecture.

Update this table after every run. Paste the `metrics` block from the JSON.
