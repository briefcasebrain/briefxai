"""Ollama provider for local LLM"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
import json

from .base import LLMProvider
from ..data.models import ConversationData, FacetValue

logger = logging.getLogger(__name__)

class OllamaLLMProvider(LLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, api_key: str = "", model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        super().__init__(api_key, model, base_url)
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
        
        logger.info(f"Initialized Ollama provider with model: {model} at {base_url}")
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using Ollama"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Check if model is available
                await self._ensure_model_available(session)
                
                # Prepare request
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.1),
                        "num_predict": kwargs.get("max_tokens", 2000),
                        "top_p": kwargs.get("top_p", 0.9)
                    }
                }
                
                # Make request
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    return result.get("response", "")
                    
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            raise Exception("Ollama request timed out")
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            raise
    
    async def _ensure_model_available(self, session: aiohttp.ClientSession) -> bool:
        """Ensure the model is available, pull if necessary"""
        try:
            # Check if model exists
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [model["name"] for model in models_data.get("models", [])]
                    
                    # Check if our model is available
                    model_variations = [
                        self.model,
                        f"{self.model}:latest",
                        f"{self.model.split(':')[0]}:latest"  # Handle model:tag format
                    ]
                    
                    for variant in model_variations:
                        if variant in available_models:
                            return True
                    
                    # Model not found, try to pull it
                    logger.info(f"Model {self.model} not found, attempting to pull...")
                    return await self._pull_model(session)
                else:
                    raise Exception(f"Failed to list models: {response.status}")
                    
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
            return False
    
    async def _pull_model(self, session: aiohttp.ClientSession) -> bool:
        """Pull a model from Ollama"""
        try:
            payload = {"name": self.model}
            
            # This is a long-running operation, extend timeout
            extended_timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
            
            async with session.post(f"{self.base_url}/api/pull", 
                                  json=payload, 
                                  timeout=extended_timeout) as response:
                if response.status == 200:
                    # Stream the response to show progress
                    async for line in response.content:
                        if line:
                            try:
                                progress = json.loads(line.decode())
                                status = progress.get("status", "")
                                if "complete" in status.lower():
                                    logger.info(f"Model {self.model} pulled successfully")
                                    return True
                                elif status:
                                    logger.debug(f"Pulling {self.model}: {status}")
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to pull model: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {self.model}: {e}")
            return False
        
        return False
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for generate_completion"""
        return asyncio.run(self.generate_completion(prompt, **kwargs))
    
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract facets from conversations using Ollama"""
        if not conversations:
            return []
        
        try:
            facet_results = []
            
            for conversation in conversations:
                text = conversation.get_text()
                if not text.strip():
                    facet_results.append([])
                    continue
                
                # Create prompt for facet extraction
                prompt = f"""Analyze this conversation and identify 5 key characteristics. For each, provide:
- Name (category like intent, sentiment, topic, urgency, complexity)
- Value (specific value for that category)
- Confidence (number from 0.0 to 1.0)

Conversation:
{text[:1000]}  # Limit text for local models

Format your response exactly like this:
Name: intent
Value: seeking_help
Confidence: 0.9

Name: sentiment  
Value: frustrated
Confidence: 0.8

Continue for 5 total facets."""
                
                try:
                    response = await self.generate_completion(
                        prompt,
                        temperature=0.2,
                        max_tokens=800
                    )
                    
                    # Parse the response
                    facets = self._parse_facets_response(response)
                    facet_results.append(facets)
                    
                except Exception as e:
                    logger.warning(f"Facet extraction failed for conversation: {e}")
                    facet_results.append([])
                
                # Small delay for local processing
                await asyncio.sleep(0.05)
            
            return facet_results
            
        except Exception as e:
            logger.error(f"Batch facet extraction failed: {e}")
            return [[] for _ in conversations]
    
    def _parse_facets_response(self, response: str) -> List[FacetValue]:
        """Parse facets from Ollama response"""
        facets = []
        
        try:
            lines = response.strip().split('\n')
            current_facet = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('Name:'):
                    if current_facet and 'name' in current_facet and 'value' in current_facet:
                        # Save previous facet
                        facets.append(FacetValue(
                            name=current_facet['name'],
                            value=current_facet['value'],
                            confidence=current_facet.get('confidence', 0.7)
                        ))
                    current_facet = {'name': line.replace('Name:', '').strip()}
                
                elif line.startswith('Value:'):
                    current_facet['value'] = line.replace('Value:', '').strip()
                
                elif line.startswith('Confidence:'):
                    try:
                        confidence_str = line.replace('Confidence:', '').strip()
                        current_facet['confidence'] = float(confidence_str)
                    except ValueError:
                        current_facet['confidence'] = 0.6
            
            # Don't forget the last facet
            if current_facet and 'name' in current_facet and 'value' in current_facet:
                facets.append(FacetValue(
                    name=current_facet['name'],
                    value=current_facet['value'],
                    confidence=current_facet.get('confidence', 0.7)
                ))
            
            # Fallback if no facets parsed
            if not facets:
                facets = [
                    FacetValue(name="analysis_method", value="ollama_local", confidence=0.5)
                ]
            
        except Exception as e:
            logger.warning(f"Failed to parse facets response: {e}")
            facets = [
                FacetValue(name="analysis_status", value="parse_error", confidence=0.3)
            ]
        
        return facets[:5]  # Return max 5 facets
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        return [model["name"] for model in models_data.get("models", [])]
                    else:
                        logger.error(f"Failed to get models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "capabilities": ["text_generation", "conversation_analysis", "facet_extraction"],
            "local": True,
            "requires_internet": False
        }