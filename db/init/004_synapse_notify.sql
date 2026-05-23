-- Phase 2: LISTEN/NOTIFY trigger for synapse_event.
-- Idempotent: safe to run multiple times.

-- Trigger function: fires pg_notify on each INSERT into synapse_event.
-- Joins synapse_session to enrich payload with host/cwd/repo.
-- Payload is kept lean: no payload JSONB column included.
-- cwd truncated to 200 chars to stay well under the 8000-byte NOTIFY limit.
CREATE OR REPLACE FUNCTION synapse_event_notify()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    _host TEXT;
    _cwd  TEXT;
    _repo TEXT;
    _payload TEXT;
BEGIN
    SELECT s.host, LEFT(s.cwd, 200), s.repo
      INTO _host, _cwd, _repo
      FROM synapse_session s
     WHERE s.session_id = NEW.session_id;

    _payload := json_build_object(
        'session_id', NEW.session_id,
        'kind',       NEW.kind,
        'file',       NEW.file,
        'line_start', NEW.line_start,
        'line_end',   NEW.line_end,
        'ts',         to_char(NEW.ts AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'),
        'host',       _host,
        'cwd',        _cwd,
        'repo',       _repo
    )::TEXT;

    PERFORM pg_notify('synapse_events', _payload);
    RETURN NEW;
END;
$$;

-- Drop and recreate trigger for idempotency.
DROP TRIGGER IF EXISTS trg_synapse_event_notify ON synapse_event;

CREATE TRIGGER trg_synapse_event_notify
AFTER INSERT ON synapse_event
FOR EACH ROW EXECUTE FUNCTION synapse_event_notify();
