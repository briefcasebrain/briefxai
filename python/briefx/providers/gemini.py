"""Google Gemini provider for LLM"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base import LLMProvider
from ..data.models import ConversationData, FacetValue

logger = logging.getLogger(__name__)

class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Map common model names
        self.model_mapping = {
            "gemini-pro": "gemini-1.5-pro",
            "gemini-flash": "gemini-1.5-flash",
            "gemini-1.5-pro": "gemini-1.5-pro-latest",
            "gemini-1.5-flash": "gemini-1.5-flash-latest"
        }
        
        self.gemini_model = self.model_mapping.get(model, model)
        
        # Initialize model
        self.model_instance = genai.GenerativeModel(
            model_name=self.gemini_model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        logger.info(f"Initialized Gemini provider with model: {self.gemini_model}")
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using Gemini"""
        try:
            # Gemini doesn't have async client, run in thread pool
            loop = asyncio.get_event_loop()
            
            def _generate():
                generation_config = genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.1),
                    max_output_tokens=kwargs.get("max_tokens", 4000),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 32)
                )
                
                response = self.model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract text from response
                if response.text:
                    return response.text
                else:
                    logger.warning("Empty response from Gemini")
                    return ""
            
            return await loop.run_in_executor(None, _generate)
                
        except Exception as e:
            logger.error(f"Gemini completion failed: {e}")
            # Check if it's a safety filter issue
            if "safety" in str(e).lower():
                logger.warning("Gemini response blocked by safety filters")
                return "Response blocked by safety filters"
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for generate_completion"""
        return asyncio.run(self.generate_completion(prompt, **kwargs))
    
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract facets from conversations using Gemini"""
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
                prompt = f"""Analyze this conversation and extract exactly 5 key characteristics. 

Conversation:
{text}

For each characteristic, provide:
1. Name: A category like intent, sentiment, topic, urgency, complexity
2. Value: The specific value for that category
3. Confidence: A number from 0.0 to 1.0

Format your response exactly like this:
Name: intent
Value: seeking_support
Confidence: 0.9

Name: sentiment
Value: neutral
Confidence: 0.8

Name: topic
Value: technical_issue
Confidence: 0.95

Name: urgency
Value: medium
Confidence: 0.7

Name: complexity
Value: moderate
Confidence: 0.6

Provide exactly 5 facets in this format."""
                
                try:
                    response = await self.generate_completion(
                        prompt,
                        temperature=0.2,
                        max_tokens=1000
                    )
                    
                    # Parse the response
                    facets = self._parse_facets_response(response)
                    facet_results.append(facets)
                    
                except Exception as e:
                    logger.warning(f"Facet extraction failed for conversation: {e}")
                    facet_results.append([])
                
                # Add delay to respect rate limits
                await asyncio.sleep(0.2)
            
            return facet_results
            
        except Exception as e:
            logger.error(f"Batch facet extraction failed: {e}")
            return [[] for _ in conversations]
    
    def _parse_facets_response(self, response: str) -> List[FacetValue]:
        """Parse facets from Gemini response"""
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
                        current_facet['confidence'] = 0.6
            
            # Don't forget the last facet
            if current_facet and 'name' in current_facet and 'value' in current_facet:
                facets.append(FacetValue(
                    name=current_facet['name'],
                    value=current_facet['value'],
                    confidence=current_facet.get('confidence', 0.8)
                ))
            
            # Ensure we have at least some facets
            if not facets:
                facets = [
                    FacetValue(name="analysis_provider", value="gemini", confidence=0.7)
                ]
            
        except Exception as e:
            logger.warning(f"Failed to parse facets response: {e}")
            facets = [
                FacetValue(name="analysis_status", value="parse_error", confidence=0.4)
            ]
        
        return facets[:5]  # Return max 5 facets
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "gemini",
            "model": self.gemini_model,
            "model_family": "gemini-1.5",
            "capabilities": ["text_generation", "conversation_analysis", "facet_extraction", "multimodal"],
            "max_tokens": 1048576,  # Gemini has very high token limits
            "supports_multimodal": True
        }