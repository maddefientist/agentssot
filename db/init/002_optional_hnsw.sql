CREATE OR REPLACE FUNCTION ssot_try_create_hnsw_indexes() RETURNS void AS $$
BEGIN
    BEGIN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_knowledge_items_embedding_hnsw ON knowledge_items USING hnsw (embedding vector_cosine_ops)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'HNSW index for knowledge_items skipped: %', SQLERRM;
    END;

    BEGIN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_requirements_embedding_hnsw ON requirements USING hnsw (embedding vector_cosine_ops)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'HNSW index for requirements skipped: %', SQLERRM;
    END;

    BEGIN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_events_embedding_hnsw ON events USING hnsw (embedding vector_cosine_ops)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'HNSW index for events skipped: %', SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;
