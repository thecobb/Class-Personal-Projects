"""Configuration management for Multimodal RAG system."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # Model Configuration
    text_embedding_model: str = "text-embedding-3-small"
    vision_model: str = "gpt-4o"
    generation_model: str = "gpt-4o"
    rerank_model: str = "rerank-english-v3.0"

    # Vector Database Configuration
    vector_db_type: Literal["chromadb", "lancedb", "qdrant", "milvus", "weaviate"] = "chromadb"
    vector_db_path: Path = Path("./data/vector_db")
    collection_name: str = "multimodal_rag"

    # Document Storage
    document_store_type: Literal["redis", "in_memory", "postgres"] = "in_memory"
    redis_host: str = "localhost"
    redis_port: int = 6379
    image_storage_path: Path = Path("./data/images")

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: Literal["recursive", "section", "element"] = "recursive"

    # Retrieval Configuration
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    hybrid_search_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    use_reranking: bool = True
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Generation Configuration
    max_context_length: int = 4000
    temperature: float = 0.1
    max_tokens: int = 1000

    # Evaluation Configuration
    synthetic_questions_per_chunk: int = 5
    target_recall: float = 0.97

    # Caching
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    enable_result_cache: bool = True
    cache_ttl: int = 3600

    # Monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def model_post_init(self, __context: object) -> None:
        """Create necessary directories after initialization."""
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.image_storage_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
