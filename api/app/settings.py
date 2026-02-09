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

    llm_provider: Literal["none", "openai", "ollama"] = Field(default="none", alias="LLM_PROVIDER")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    ollama_chat_model: str = Field(default="llama3.1", alias="OLLAMA_CHAT_MODEL")

    compaction_enabled: bool = Field(default=True, alias="COMPACTION_ENABLED")
    compaction_event_threshold: int = Field(default=80, alias="COMPACTION_EVENT_THRESHOLD")
    compaction_char_threshold: int = Field(default=24000, alias="COMPACTION_CHAR_THRESHOLD")
    compaction_interval_seconds: int = Field(default=60, alias="COMPACTION_INTERVAL_SECONDS")

    enable_hnsw_index: bool = Field(default=False, alias="ENABLE_HNSW_INDEX")
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
