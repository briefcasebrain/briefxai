"""HuggingFace provider for local and remote LLM"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import LLMProvider
from ..data.models import ConversationData, FacetValue

logger = logging.getLogger(__name__)

class HuggingFaceLLMProvider(LLMProvider):
    """HuggingFace Transformers LLM provider"""
    
    def __init__(self, api_key: str = "", model: str = "microsoft/DialoGPT-small", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package not installed. Install with: pip install transformers torch")
        
        self.model_name = model
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Common model mappings
        self.model_mapping = {
            "gpt2": "gpt2",
            "gpt2-small": "gpt2",
            "gpt2-medium": "gpt2-medium", 
            "gpt2-large": "gpt2-large",
            "gpt2-xl": "gpt2-xl",
            "dialogpt": "microsoft/DialoGPT-small",
            "dialogpt-small": "microsoft/DialoGPT-small",
            "dialogpt-medium": "microsoft/DialoGPT-medium",
            "dialogpt-large": "microsoft/DialoGPT-large",
            "distilgpt2": "distilgpt2",
            "blenderbot": "facebook/blenderbot-400M-distill",
            "t5-small": "t5-small",
            "t5-base": "t5-base"
        }
        
        self.hf_model = self.model_mapping.get(model, model)
        self._load_model()
        
        logger.info(f"Initialized HuggingFace provider with model: {self.hf_model} on {self.device}")
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer"""
        try:
            logger.info(f"Loading model {self.hf_model}...")
            
            # For text generation models
            if any(name in self.hf_model.lower() for name in ['gpt', 'dialog', 'distil']):
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Use pipeline for simpler interface
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
            # For T5 models (text-to-text)
            elif 't5' in self.hf_model.lower():
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.hf_model,
                    device=0 if self.device == "cuda" else -1
                )
            
            else:
                # Try text generation as default
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    device=0 if self.device == "cuda" else -1
                )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {self.hf_model}: {e}")
            raise
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using HuggingFace model"""
        if not self.pipeline:
            raise Exception("Model not loaded")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate():
                max_tokens = kwargs.get("max_tokens", 150)
                temperature = kwargs.get("temperature", 0.7)
                
                # Adjust parameters based on model type
                if 't5' in self.hf_model.lower():
                    # T5 expects different parameters
                    result = self.pipeline(
                        prompt,
                        max_length=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        num_return_sequences=1
                    )
                    return result[0]['generated_text']
                else:
                    # Standard text generation
                    result = self.pipeline(
                        prompt,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
                    )
                    
                    generated_text = result[0]['generated_text']
                    # Remove the original prompt from the generated text
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    return generated_text
            
            return await loop.run_in_executor(None, _generate)
                
        except Exception as e:
            logger.error(f"HuggingFace completion failed: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for generate_completion"""
        return asyncio.run(self.generate_completion(prompt, **kwargs))
    
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract facets from conversations using HuggingFace model"""
        if not conversations:
            return []
        
        try:
            facet_results = []
            
            for conversation in conversations:
                text = conversation.get_text()
                if not text.strip():
                    facet_results.append([])
                    continue
                
                # Simplified facet extraction for local models
                prompt = f"""Analyze this conversation and identify key characteristics:

Text: {text[:500]}  # Limit for local processing

Identify: intent, sentiment, topic, urgency, complexity

Format: intent=help_seeking, sentiment=neutral, topic=technical, urgency=medium, complexity=low"""
                
                try:
                    response = await self.generate_completion(
                        prompt,
                        temperature=0.3,
                        max_tokens=100
                    )
                    
                    # Parse the response
                    facets = self._parse_facets_response(response)
                    facet_results.append(facets)
                    
                except Exception as e:
                    logger.warning(f"Facet extraction failed for conversation: {e}")
                    # Fallback to simple heuristics
                    facets = self._fallback_facets(text)
                    facet_results.append(facets)
                
                # Small delay for local processing
                await asyncio.sleep(0.1)
            
            return facet_results
            
        except Exception as e:
            logger.error(f"Batch facet extraction failed: {e}")
            return [[] for _ in conversations]
    
    def _parse_facets_response(self, response: str) -> List[FacetValue]:
        """Parse facets from HuggingFace model response"""
        facets = []
        
        try:
            # Look for key=value patterns
            import re
            
            patterns = {
                'intent': r'intent[=:]\s*([^,\n\s]+)',
                'sentiment': r'sentiment[=:]\s*([^,\n\s]+)', 
                'topic': r'topic[=:]\s*([^,\n\s]+)',
                'urgency': r'urgency[=:]\s*([^,\n\s]+)',
                'complexity': r'complexity[=:]\s*([^,\n\s]+)'
            }
            
            for facet_name, pattern in patterns.items():
                match = re.search(pattern, response.lower())
                if match:
                    value = match.group(1).strip()
                    facets.append(FacetValue(
                        name=facet_name,
                        value=value,
                        confidence=0.6  # Lower confidence for local models
                    ))
            
            # Fallback if no patterns matched
            if not facets:
                facets = [
                    FacetValue(name="analysis_method", value="huggingface_local", confidence=0.5)
                ]
            
        except Exception as e:
            logger.warning(f"Failed to parse facets response: {e}")
            facets = [
                FacetValue(name="analysis_status", value="parse_error", confidence=0.3)
            ]
        
        return facets
    
    def _fallback_facets(self, text: str) -> List[FacetValue]:
        """Generate simple facets when model fails"""
        from ..utils import determine_category, calculate_simple_sentiment
        
        return [
            FacetValue(name="category", value=determine_category(text), confidence=0.7),
            FacetValue(name="sentiment_score", value=str(calculate_simple_sentiment(text)), confidence=0.6),
            FacetValue(name="length", value="long" if len(text) > 200 else "short", confidence=0.9),
            FacetValue(name="analysis_method", value="fallback_heuristics", confidence=0.8)
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "huggingface",
            "model": self.hf_model,
            "device": self.device,
            "capabilities": ["text_generation", "conversation_analysis"],
            "local": True,
            "requires_internet": False,  # After initial download
            "memory_intensive": True
        }