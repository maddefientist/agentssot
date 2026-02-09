CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'entity_type') THEN
        CREATE TYPE entity_type AS ENUM ('project', 'person', 'agent', 'document', 'integration', 'other');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'requirement_priority') THEN
        CREATE TYPE requirement_priority AS ENUM ('low', 'medium', 'high', 'critical');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'requirement_status') THEN
        CREATE TYPE requirement_status AS ENUM ('draft', 'proposed', 'in_progress', 'blocked', 'done', 'archived');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'event_type') THEN
        CREATE TYPE event_type AS ENUM ('note', 'decision', 'directive', 'action', 'result', 'error');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'api_role') THEN
        CREATE TYPE api_role AS ENUM ('reader', 'writer', 'admin');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS namespaces (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    role api_role NOT NULL,
    namespaces TEXT[] NOT NULL DEFAULT ARRAY['default']::TEXT[],
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    type entity_type NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(namespace, slug)
);

CREATE TABLE IF NOT EXISTS requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
    project_id UUID REFERENCES entities(id) ON DELETE SET NULL,
    owner_entity_id UUID REFERENCES entities(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    body TEXT,
    priority requirement_priority NOT NULL DEFAULT 'medium',
    status requirement_status NOT NULL DEFAULT 'draft',
    context_snippet TEXT,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
    project_id UUID REFERENCES entities(id) ON DELETE SET NULL,
    entity_id UUID REFERENCES entities(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    source TEXT,
    source_ref TEXT,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL REFERENCES namespaces(name) ON DELETE CASCADE,
    project_id UUID REFERENCES entities(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES entities(id) ON DELETE SET NULL,
    type event_type NOT NULL DEFAULT 'note',
    title TEXT NOT NULL,
    body TEXT,
    context_snippet TEXT,
    session_id TEXT,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_namespaces_gin ON api_keys USING GIN(namespaces);

CREATE INDEX IF NOT EXISTS idx_entities_namespace_slug ON entities(namespace, slug);
CREATE INDEX IF NOT EXISTS idx_entities_namespace_type ON entities(namespace, type);

CREATE INDEX IF NOT EXISTS idx_requirements_namespace ON requirements(namespace);
CREATE INDEX IF NOT EXISTS idx_requirements_project ON requirements(project_id);
CREATE INDEX IF NOT EXISTS idx_requirements_owner ON requirements(owner_entity_id);
CREATE INDEX IF NOT EXISTS idx_requirements_status_priority ON requirements(status, priority);
CREATE INDEX IF NOT EXISTS idx_requirements_tags_gin ON requirements USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_knowledge_items_namespace ON knowledge_items(namespace);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_project ON knowledge_items(project_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_entity ON knowledge_items(entity_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_tags_gin ON knowledge_items USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_events_namespace_session ON events(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_archived ON events(is_archived);
CREATE INDEX IF NOT EXISTS idx_events_tags_gin ON events USING GIN(tags);

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_entities_set_updated_at ON entities;
CREATE TRIGGER trg_entities_set_updated_at
BEFORE UPDATE ON entities
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_requirements_set_updated_at ON requirements;
CREATE TRIGGER trg_requirements_set_updated_at
BEFORE UPDATE ON requirements
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

INSERT INTO namespaces(name)
SELECT 'default'
WHERE NOT EXISTS (SELECT 1 FROM namespaces WHERE name = 'default');
