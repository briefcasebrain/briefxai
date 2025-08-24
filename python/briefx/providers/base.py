"""Base classes for LLM and embedding providers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from ..data.models import ConversationData, FacetValue

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    @abstractmethod
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract facets from conversations"""
        pass

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        pass