-- 006_fts_gin.sql — GIN index for hybrid keyword+vector recall (lift #2).
--
-- The hybrid retrieval track fuses Postgres full-text search with pgvector
-- recall via reciprocal-rank fusion (see _recall_knowledge_weighted). Without
-- an index, `to_tsvector(content) @@ websearch_to_tsquery(...)` is a sequential
-- scan. This functional GIN index makes the keyword track O(log n).
--
-- Language must match RECALL_FTS_LANGUAGE (default: english). If you change that
-- setting, add a matching index for the new language config.
--
-- Idempotent: safe to re-run.

CREATE INDEX IF NOT EXISTS idx_knowledge_items_content_fts
    ON knowledge_items
    USING gin (to_tsvector('english', content));
