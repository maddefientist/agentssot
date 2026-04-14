-- Verbatim mode: opt-out of L0/L1 summarization for truth-critical items.
-- When true, synthesis (immediate or background) MUST NOT derive abstract/summary
-- from the content. The full content is the only allowed surface.

ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS verbatim BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_knowledge_items_verbatim
    ON knowledge_items(verbatim) WHERE verbatim = TRUE;
