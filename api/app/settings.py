from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    database_url: str = Field(alias="DATABASE_URL")
    api_port: int = Field(default=8088, alias="API_PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")
    max_snippet_chars: int = Field(default=900, alias="MAX_SNIPPET_CHARS")

    embedding_provider: Literal["none", "openai", "ollama"] = Field(
        default="none", alias="EMBEDDING_PROVIDER"
    )
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embed_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBED_MODEL")
    ollama_base_url: str = Field(default="http://host.docker.internal:11434", alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")
    ollama_embed_cpu_only: bool = Field(default=True, alias="OLLAMA_EMBED_CPU_ONLY")

    llm_provider: Literal["none", "openai", "ollama"] = Field(default="none", alias="LLM_PROVIDER")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    ollama_chat_model: str = Field(default="llama3.1", alias="OLLAMA_CHAT_MODEL")

    # Reranker
    reranker_provider: Literal["none", "ollama"] = Field(default="none", alias="RERANKER_PROVIDER")
    ollama_reranker_base_url: str = Field(
        default="", alias="OLLAMA_RERANKER_BASE_URL"
    )
    ollama_reranker_model: str = Field(
        default="dengcao/Qwen3-Reranker-8B:Q8_0", alias="OLLAMA_RERANKER_MODEL"
    )
    reranker_candidate_multiplier: int = Field(default=3, alias="RERANKER_CANDIDATE_MULTIPLIER")
    # Two-tier reranker: 4B for procedural-only queries, 8B for nuanced
    ollama_reranker_fast_model: str = Field(
        default="dengcao/Qwen3-Reranker-4B:Q4_K_M", alias="OLLAMA_RERANKER_FAST_MODEL"
    )
    ollama_reranker_fast_base_url: str = Field(
        default="", alias="OLLAMA_RERANKER_FAST_BASE_URL"
    )
    procedural_tiers: list[str] = Field(
        default=["command", "rule", "entity"],
        description="Tiers that route to the fast reranker when queried alone",
    )

    compaction_enabled: bool = Field(default=True, alias="COMPACTION_ENABLED")
    compaction_event_threshold: int = Field(default=80, alias="COMPACTION_EVENT_THRESHOLD")
    compaction_char_threshold: int = Field(default=24000, alias="COMPACTION_CHAR_THRESHOLD")
    compaction_interval_seconds: int = Field(default=60, alias="COMPACTION_INTERVAL_SECONDS")

    # Synthesis (conceptual memory)
    synthesis_enabled: bool = Field(default=False, alias="SYNTHESIS_ENABLED")
    synthesis_model: str = Field(default="qwen3.5:cloud", alias="SYNTHESIS_MODEL")
    synthesis_fallback_model: str = Field(default="qwen3:latest", alias="SYNTHESIS_FALLBACK_MODEL")
    synthesis_schedule_hour: int = Field(default=3, alias="SYNTHESIS_SCHEDULE_HOUR")
    synthesis_similarity_threshold: float = Field(default=0.65, alias="SYNTHESIS_SIMILARITY_THRESHOLD")
    synthesis_min_cluster_size: int = Field(default=2, alias="SYNTHESIS_MIN_CLUSTER_SIZE")
    synthesis_confidence_decay: float = Field(default=0.02, alias="SYNTHESIS_CONFIDENCE_DECAY")
    synthesis_decay_grace_days: int = Field(default=90, alias="SYNTHESIS_DECAY_GRACE_DAYS")
    synthesis_decay_floor: float = Field(default=0.15, alias="SYNTHESIS_DECAY_FLOOR")
    synthesis_feedback_protection_days: int = Field(default=180, alias="SYNTHESIS_FEEDBACK_PROTECTION_DAYS")

    enable_hnsw_index: bool = Field(default=False, alias="ENABLE_HNSW_INDEX")

    # Typed memory: when enabled, recall accepts memory_type and staleness filters
    typed_memory_enabled: bool = Field(default=False, alias="TYPED_MEMORY_ENABLED")

    # Secret scanning: reject knowledge items containing likely secrets on ingest
    ingest_secret_scanning: bool = Field(default=True, alias="INGEST_SECRET_SCANNING")

    # Sync tracking: enable per-device sync checkpoints and conflict detection
    sync_tracking_enabled: bool = Field(default=False, alias="SYNC_TRACKING_ENABLED")
    sync_conflict_window_hours: int = Field(default=24, alias="SYNC_CONFLICT_WINDOW_HOURS")

    # Write-ahead log (audit artifact for ingest/delete operations)
    wal_enabled: bool = Field(default=True, alias="WAL_ENABLED")
    wal_dir: str = Field(default="/var/lib/agentssot/wal", alias="WAL_DIR")
    wal_retention_days: int = Field(default=30, alias="WAL_RETENTION_DAYS")

    # Semantic dedup on ingest: if cosine similarity to an existing item in the
    # same namespace meets or exceeds this threshold, skip insert and return the
    # existing item. Verbatim items bypass this check. 0.0 disables dedup.
    semantic_dedup_threshold: float = Field(default=0.0, alias="SEMANTIC_DEDUP_THRESHOLD")

    # Auto-classifier (Plan 1 Phase 2)
    classifier_provider: Literal["none", "ollama"] = Field(
        default="ollama", alias="CLASSIFIER_PROVIDER"
    )
    classifier_model: str = Field(
        default="gemma4:31b-cloud", alias="CLASSIFIER_MODEL"
    )
    classifier_base_url: str = Field(
        default="", alias="CLASSIFIER_BASE_URL"
    )
    classifier_timeout_seconds: int = Field(
        default=20, alias="CLASSIFIER_TIMEOUT_SECONDS"
    )
    classifier_min_confidence: float = Field(
        default=0.6, alias="CLASSIFIER_MIN_CONFIDENCE"
    )

    # Open enrollment passphrase (empty = no passphrase required)
    enrollment_passphrase: str = Field(default="", alias="ENROLLMENT_PASSPHRASE")
    expose_db_port: bool = Field(default=False, alias="EXPOSE_DB_PORT")

    bootstrap_admin_namespaces: str = Field(default="default", alias="BOOTSTRAP_ADMIN_NAMESPACES")

    @property
    def bootstrap_namespace_list(self) -> list[str]:
        items = [ns.strip() for ns in self.bootstrap_admin_namespaces.split(",") if ns.strip()]
        return items or ["default"]

    @property
    def effective_compaction_enabled(self) -> bool:
        # Compaction is hard-disabled when no summarizer provider is configured.
        return self.compaction_enabled and self.llm_provider != "none"

    @property
    def effective_synthesis_enabled(self) -> bool:
        return self.synthesis_enabled and self.llm_provider != "none"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
