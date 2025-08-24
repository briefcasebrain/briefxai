"""Anthropic provider for LLM"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import LLMProvider
from ..data.models import ConversationData, FacetValue

logger = logging.getLogger(__name__)

class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Map common model names to Anthropic model identifiers
        self.model_mapping = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229", 
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku": "claude-3-5-haiku-20241022"
        }
        
        # Use mapped model if available
        self.anthropic_model = self.model_mapping.get(model, model)
        logger.info(f"Initialized Anthropic provider with model: {self.anthropic_model}")
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using Anthropic Claude"""
        try:
            # Handle system prompt if provided
            system_prompt = kwargs.get("system_prompt", "")
            
            message = await self.client.messages.create(
                model=self.anthropic_model,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.1),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract text from response
            if message.content and len(message.content) > 0:
                return message.content[0].text
            else:
                logger.warning("Empty response from Anthropic")
                return ""
                
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for generate_completion"""
        return asyncio.run(self.generate_completion(prompt, **kwargs))
    
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract facets from conversations using Claude"""
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
                prompt = f"""Analyze this conversation and extract key facets. For each facet, provide:
1. The facet name (e.g., "intent", "sentiment", "topic", "urgency")
2. The facet value
3. Confidence score (0.0 to 1.0)

Conversation:
{text}

Return exactly 5 facets in this format:
Name: [facet_name]
Value: [facet_value] 
Confidence: [0.0-1.0]

Example:
Name: intent
Value: seeking_support
Confidence: 0.9"""
                
                try:
                    response = await self.generate_completion(
                        prompt,
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    # Parse the response
                    facets = self._parse_facets_response(response)
                    facet_results.append(facets)
                    
                except Exception as e:
                    logger.warning(f"Facet extraction failed for conversation: {e}")
                    facet_results.append([])
                
                # Add small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            return facet_results
            
        except Exception as e:
            logger.error(f"Batch facet extraction failed: {e}")
            return [[] for _ in conversations]
    
    def _parse_facets_response(self, response: str) -> List[FacetValue]:
        """Parse facets from Claude response"""
        facets = []
        
        try:
            lines = response.strip().split('\n')
            current_facet = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('Name:'):
                    if current_facet:
                        # Save previous facet
                        if all(key in current_facet for key in ['name', 'value']):
                            facets.append(FacetValue(
                                name=current_facet['name'],
                                value=current_facet['value'],
                                confidence=current_facet.get('confidence', 0.8)
                            ))
                    current_facet = {'name': line.replace('Name:', '').strip()}
                
                elif line.startswith('Value:'):
                    current_facet['value'] = line.replace('Value:', '').strip()
                
                elif line.startswith('Confidence:'):
                    try:
                        confidence_str = line.replace('Confidence:', '').strip()
                        current_facet['confidence'] = float(confidence_str)
                    except ValueError:
                        current_facet['confidence'] = 0.5
            
            # Don't forget the last facet
            if current_facet and all(key in current_facet for key in ['name', 'value']):
                facets.append(FacetValue(
                    name=current_facet['name'],
                    value=current_facet['value'],
                    confidence=current_facet.get('confidence', 0.8)
                ))
            
            # Ensure we have at least some facets
            if not facets:
                facets = [
                    FacetValue(name="analysis_status", value="failed", confidence=0.1)
                ]
            
        except Exception as e:
            logger.warning(f"Failed to parse facets response: {e}")
            facets = [
                FacetValue(name="analysis_status", value="parse_error", confidence=0.1)
            ]
        
        return facets[:5]  # Return max 5 facets
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "anthropic",
            "model": self.anthropic_model,
            "model_family": "claude-3" if "claude-3" in self.anthropic_model else "claude",
            "capabilities": ["text_generation", "conversation_analysis", "facet_extraction"],
            "max_tokens": 4096,  # Conservative estimate
            "supports_system_prompt": True
        }