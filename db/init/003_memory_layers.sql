-- Add memory_category enum (OpenViking-style categories)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memory_category') THEN
        CREATE TYPE memory_category AS ENUM (
            'user_profile',      -- who the user is
            'user_preferences',  -- how user likes things
            'user_entities',     -- people, projects user knows
            'user_events',       -- decisions, milestones
            'agent_patterns',    -- learned workflows
            'agent_tools',       -- tool usage knowledge
            'agent_skills',      -- skill execution history
            'agent_cases'        -- problem-solving cases
        );
    END IF;
END $$;

-- Add content_layer enum (L0/L1/L2)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'content_layer') THEN
        CREATE TYPE content_layer AS ENUM ('abstract', 'summary', 'full');
    END IF;
END $$;

-- Add columns to knowledge_items
ALTER TABLE knowledge_items 
    ADD COLUMN IF NOT EXISTS category memory_category,
    ADD COLUMN IF NOT EXISTS layer content_layer DEFAULT 'full',
    ADD COLUMN IF NOT EXISTS abstract TEXT,  -- L0: ~50 tokens
    ADD COLUMN IF NOT EXISTS summary TEXT,     -- L1: ~500 tokens
    ADD COLUMN IF NOT EXISTS source_ki_id UUID REFERENCES knowledge_items(id) ON DELETE SET NULL;

-- Indexes for efficient filtering
CREATE INDEX IF NOT EXISTS idx_knowledge_items_category ON knowledge_items(category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_items_layer ON knowledge_items(layer);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_source_ki ON knowledge_items(source_ki_id) WHERE source_ki_id IS NOT NULL;

-- Trigger to auto-set category from memory_type for backward compat
CREATE OR REPLACE FUNCTION set_memory_category() RETURNS TRIGGER AS $$
BEGIN
    -- Auto-assign category based on memory_type if category not set
    IF NEW.category IS NULL AND NEW.memory_type IS NOT NULL THEN
        CASE NEW.memory_type
            WHEN 'preference' THEN NEW.category := 'user_preferences';
            WHEN 'decision' THEN NEW.category := 'user_events';
            WHEN 'skill' THEN NEW.category := 'agent_skills';
            WHEN 'fact' THEN NEW.category := 'user_entities';
            ELSE NEW.category := 'user_entities';
        END CASE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_knowledge_items_set_category ON knowledge_items;
CREATE TRIGGER trg_knowledge_items_set_category
    BEFORE INSERT ON knowledge_items
    FOR EACH ROW
    EXECUTE FUNCTION set_memory_category();
