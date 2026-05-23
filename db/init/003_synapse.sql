-- Synapse plane: volatile live-session state for cross-TUI working-memory awareness.
-- UNLOGGED for performance (RAM-like semantics, no WAL overhead, drops on crash).
-- Idempotent: safe to run multiple times.

CREATE UNLOGGED TABLE IF NOT EXISTS synapse_session (
  session_id   TEXT PRIMARY KEY,
  host         TEXT NOT NULL,
  cwd          TEXT NOT NULL,
  repo         TEXT,
  agent        TEXT NOT NULL,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
  current_file TEXT,
  current_op   TEXT
);

CREATE INDEX IF NOT EXISTS ix_synapse_session_last_seen ON synapse_session (last_seen);
CREATE INDEX IF NOT EXISTS ix_synapse_session_repo_lastseen ON synapse_session (repo, last_seen);

CREATE UNLOGGED TABLE IF NOT EXISTS synapse_event (
  id          BIGSERIAL PRIMARY KEY,
  session_id  TEXT NOT NULL REFERENCES synapse_session(session_id) ON DELETE CASCADE,
  ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
  kind        TEXT NOT NULL,
  file        TEXT,
  line_start  INT,
  line_end    INT,
  payload     JSONB
);

CREATE INDEX IF NOT EXISTS ix_synapse_event_ts ON synapse_event (ts);
CREATE INDEX IF NOT EXISTS ix_synapse_event_file_ts ON synapse_event (file, ts);
