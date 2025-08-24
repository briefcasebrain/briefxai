"""Provider factory for creating LLM and embedding providers"""

import logging
import time
from typing import Optional, Dict, Any, List
from enum import Enum

from ..data.models import ConversationData
from .base import LLMProvider, EmbeddingProvider
from config import LlmProvider as LlmProviderEnum, EmbeddingProvider as EmbeddingProviderEnum

logger = logging.getLogger(__name__)

class ProviderFactory:
    """Factory for creating and managing providers"""
    
    @staticmethod
    def create_llm_provider(
        provider: LlmProviderEnum,
        api_key: Optional[str] = None,
        model: str = "default",
        base_url: Optional[str] = None,
        **kwargs
    ) -> Optional[LLMProvider]:
        """Create an LLM provider instance"""
        
        try:
            if provider == LlmProviderEnum.OPENAI:
                from .openai import OpenAILLMProvider
                if not api_key:
                    logger.error("OpenAI API key required")
                    return None
                return OpenAILLMProvider(
                    api_key=api_key,
                    model=model if model != "default" else "gpt-4o-mini",
                    base_url=base_url
                )
            
            elif provider == LlmProviderEnum.ANTHROPIC:
                from .anthropic import AnthropicLLMProvider
                if not api_key:
                    logger.error("Anthropic API key required")
                    return None
                return AnthropicLLMProvider(
                    api_key=api_key,
                    model=model if model != "default" else "claude-3-haiku"
                )
            
            elif provider == LlmProviderEnum.OLLAMA:
                from .ollama import OllamaLLMProvider
                return OllamaLLMProvider(
                    model=model if model != "default" else "llama3.2",
                    base_url=base_url or "http://localhost:11434"
                )
            
            elif provider == LlmProviderEnum.GEMINI:
                from .gemini import GeminiLLMProvider
                if not api_key:
                    logger.error("Gemini API key required")
                    return None
                return GeminiLLMProvider(
                    api_key=api_key,
                    model=model if model != "default" else "gemini-1.5-flash"
                )
            
            elif provider == LlmProviderEnum.HUGGINGFACE:
                from .huggingface import HuggingFaceLLMProvider
                return HuggingFaceLLMProvider(
                    model=model if model != "default" else "microsoft/DialoGPT-small"
                )
            
            else:
                logger.error(f"Unsupported LLM provider: {provider}")
                return None
                
        except ImportError as e:
            logger.error(f"Provider {provider.value} not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create {provider.value} provider: {e}")
            return None
    
    @staticmethod
    def create_embedding_provider(
        provider: EmbeddingProviderEnum,
        api_key: Optional[str] = None,
        model: str = "default",
        **kwargs
    ) -> Optional[EmbeddingProvider]:
        """Create an embedding provider instance"""
        
        try:
            if provider == EmbeddingProviderEnum.OPENAI:
                from .openai import OpenAIEmbeddingProvider
                if not api_key:
                    logger.error("OpenAI API key required")
                    return None
                return OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model=model if model != "default" else "text-embedding-3-small"
                )
            
            # Add other embedding providers here as they're implemented
            else:
                logger.error(f"Unsupported embedding provider: {provider}")
                return None
                
        except ImportError as e:
            logger.error(f"Embedding provider {provider.value} not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create {provider.value} embedding provider: {e}")
            return None
    
    @staticmethod
    def get_available_llm_providers() -> Dict[str, Dict[str, Any]]:
        """Get information about available LLM providers"""
        providers = {}
        
        # Check OpenAI
        try:
            from .openai import OpenAILLMProvider
            providers["openai"] = {
                "name": "OpenAI",
                "available": True,
                "requires_api_key": True,
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "capabilities": ["text_generation", "conversation_analysis", "facet_extraction"]
            }
        except ImportError:
            providers["openai"] = {"name": "OpenAI", "available": False, "error": "openai package not installed"}
        
        # Check Anthropic
        try:
            from .anthropic import AnthropicLLMProvider
            providers["anthropic"] = {
                "name": "Anthropic",
                "available": True,
                "requires_api_key": True,
                "models": ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "capabilities": ["text_generation", "conversation_analysis", "facet_extraction"]
            }
        except ImportError:
            providers["anthropic"] = {"name": "Anthropic", "available": False, "error": "anthropic package not installed"}
        
        # Check Ollama
        try:
            from .ollama import OllamaLLMProvider
            providers["ollama"] = {
                "name": "Ollama",
                "available": True,
                "requires_api_key": False,
                "models": ["llama3.2", "llama3.1", "mistral", "codellama", "phi3"],
                "capabilities": ["text_generation", "conversation_analysis", "facet_extraction"],
                "local": True
            }
        except ImportError:
            providers["ollama"] = {"name": "Ollama", "available": False, "error": "aiohttp not available"}
        
        # Check Gemini
        try:
            from .gemini import GeminiLLMProvider
            providers["gemini"] = {
                "name": "Google Gemini",
                "available": True,
                "requires_api_key": True,
                "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
                "capabilities": ["text_generation", "conversation_analysis", "facet_extraction", "multimodal"]
            }
        except ImportError:
            providers["gemini"] = {"name": "Google Gemini", "available": False, "error": "google-generativeai not installed"}
        
        # Check HuggingFace
        try:
            from .huggingface import HuggingFaceLLMProvider
            providers["huggingface"] = {
                "name": "HuggingFace",
                "available": True,
                "requires_api_key": False,
                "models": ["gpt2", "distilgpt2", "dialogpt", "blenderbot", "t5-base"],
                "capabilities": ["text_generation", "conversation_analysis"],
                "local": True,
                "memory_intensive": True
            }
        except ImportError:
            providers["huggingface"] = {"name": "HuggingFace", "available": False, "error": "transformers or torch not installed"}
        
        return providers
    
    @staticmethod
    async def test_provider_connection(provider: LLMProvider) -> Dict[str, Any]:
        """Test if a provider is working correctly"""
        test_prompt = "Hello! Please respond with just the word 'test' to confirm you're working."
        
        try:
            start_time = time.time()
            response = await provider.generate_completion(test_prompt, max_tokens=10)
            end_time = time.time()
            
            return {
                "success": True,
                "response_time": round(end_time - start_time, 2),
                "response": response[:50] + "..." if len(response) > 50 else response,
                "provider_info": provider.get_model_info() if hasattr(provider, 'get_model_info') else {}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

# Convenience functions
def create_llm_provider(provider_name: str, **kwargs) -> Optional[LLMProvider]:
    """Create LLM provider from string name"""
    try:
        provider_enum = LlmProviderEnum(provider_name.lower())
        return ProviderFactory.create_llm_provider(provider_enum, **kwargs)
    except ValueError:
        logger.error(f"Unknown provider: {provider_name}")
        return None

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Get available providers"""
    return ProviderFactory.get_available_llm_providers()