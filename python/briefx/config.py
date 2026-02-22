"""
Configuration module for BriefX analysis pipeline.

Provides BriefXConfig dataclass and provider enums used by the pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from .providers.factory import LlmProviderEnum, EmbeddingProviderEnum


# Re-export provider enums for backward compatibility
LlmProvider = LlmProviderEnum
EmbeddingProvider = EmbeddingProviderEnum


@dataclass
class BriefXConfig:
    """Configuration for the BriefX analysis pipeline."""

    # LLM provider settings
    llm_provider: LlmProviderEnum = LlmProviderEnum.OPENAI
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: Optional[str] = None
    llm_batch_size: int = 5

    # Embedding provider settings
    embedding_provider: EmbeddingProviderEnum = EmbeddingProviderEnum.OPENAI
    embedding_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 50

    # Convenience API key fields (used by get_clio_pipeline in app.py)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Data processing options
    dedup_data: bool = True
    max_conversations: Optional[int] = None

    # Analysis options
    max_clusters: int = 10
    clustering_method: str = "auto"
    privacy_level: str = "standard"

    def __post_init__(self):
        """Auto-detect API keys from environment variables if not set."""
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Sync llm_api_key with provider-specific keys
        if self.llm_api_key is None:
            if self.llm_provider == LlmProviderEnum.OPENAI:
                self.llm_api_key = self.openai_api_key
            elif self.llm_provider == LlmProviderEnum.ANTHROPIC:
                self.llm_api_key = self.anthropic_api_key

        if self.embedding_api_key is None:
            self.embedding_api_key = self.openai_api_key


def get_default_config() -> BriefXConfig:
    """Create a default configuration based on available environment variables."""
    config = BriefXConfig()

    # Auto-select provider based on available API keys
    if config.openai_api_key:
        config.llm_provider = LlmProviderEnum.OPENAI
        config.llm_api_key = config.openai_api_key
        config.embedding_api_key = config.openai_api_key
    elif config.anthropic_api_key:
        config.llm_provider = LlmProviderEnum.ANTHROPIC
        config.llm_api_key = config.anthropic_api_key
    else:
        # Fall back to Ollama (local, no API key needed)
        config.llm_provider = LlmProviderEnum.OLLAMA
        config.llm_model = "llama3.2"

    return config
