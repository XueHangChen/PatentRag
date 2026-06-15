"""Application settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PATENT_RAG_",
        extra="ignore",
    )

    env: str = "local"
    raw_patent_dir: Path = Path("patant")
    processed_dir: Path = Path("data/processed")
    indexes_dir: Path = Path("data/indexes")
    graph_dir: Path = Path("data/graph")
    log_level: str = "INFO"

    llm_provider: str = "dashscope"
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int | None = None
    openai_api_key: str | None = Field(default=None, repr=False)
    dashscope_api_key: str | None = Field(default=None, repr=False)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_settings() -> Settings:
    """Return application settings."""

    return Settings()
