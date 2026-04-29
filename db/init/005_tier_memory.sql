-- 005_tier_memory.sql — Hive tier-memory overhaul, Plan 1 Phase 0
-- Additive only. Every column has a non-breaking default.
-- Rollback: drop the columns/indexes/tables added below.
--
-- NOTE: memory_type is stored as TEXT in this schema (not a PG enum type).
-- New tier values (command, rule, entity, episodic) are valid text values
-- in the Python MemoryType enum and require no DDL change here.

BEGIN;

-- 1. KnowledgeItem additions
ALTER TABLE knowledge_items
  ADD COLUMN IF NOT EXISTS expires_at         TIMESTAMPTZ NULL,
  ADD COLUMN IF NOT EXISTS superseded_by      UUID NULL REFERENCES knowledge_items(id),
  ADD COLUMN IF NOT EXISTS confidence         DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  ADD COLUMN IF NOT EXISTS entity_refs        JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS rule_refs          JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS cwd_hints          JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS device_hints       JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS loadout_priority   INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_classified_at TIMESTAMPTZ NULL;

-- abstract and summary already exist from milestone 3 (003_memory_layers.sql); do not recreate.

-- 2. Indexes
CREATE INDEX IF NOT EXISTS ix_ki_expires_at
  ON knowledge_items(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_ki_superseded_by
  ON knowledge_items(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_ki_loadout_priority
  ON knowledge_items(loadout_priority DESC) WHERE confidence >= 0.5;
CREATE INDEX IF NOT EXISTS ix_ki_cwd_hints_gin
  ON knowledge_items USING GIN (cwd_hints jsonb_path_ops);
CREATE INDEX IF NOT EXISTS ix_ki_entity_refs_gin
  ON knowledge_items USING GIN (entity_refs jsonb_path_ops);
CREATE INDEX IF NOT EXISTS ix_ki_last_classified_at
  ON knowledge_items(last_classified_at);

-- 3. Deletion audit log
CREATE TABLE IF NOT EXISTS deletion_log (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  item_id     UUID NOT NULL,
  namespace   VARCHAR NOT NULL,
  reason      TEXT,
  deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_by  VARCHAR,
  payload     JSONB
);
CREATE INDEX IF NOT EXISTS ix_deletion_log_item_id ON deletion_log(item_id);

-- 4. Review queue (low-conf, dup, supersede, contradiction)
CREATE TABLE IF NOT EXISTS review_queue (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  namespace     VARCHAR NOT NULL,
  kind          VARCHAR NOT NULL CHECK (kind IN ('low_conf','dup','supersede','contradiction')),
  priority      INTEGER NOT NULL DEFAULT 0,
  primary_id    UUID NOT NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
  secondary_id  UUID NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
  reason        TEXT,
  status        VARCHAR NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','resolved','dismissed')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at   TIMESTAMPTZ NULL,
  resolved_by   VARCHAR NULL
);
CREATE INDEX IF NOT EXISTS ix_rq_status_priority
  ON review_queue(status, priority DESC) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS ix_rq_namespace_kind
  ON review_queue(namespace, kind);

-- 5. memory_type is TEXT in this schema — no ALTER TYPE needed.
-- The Python MemoryType enum in models.py is the source of truth for valid values.
-- New values: command, rule, entity, episodic (added in Plan 1 Phase 0).

COMMIT;
