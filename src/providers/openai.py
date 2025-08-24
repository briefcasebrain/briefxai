"""OpenAI provider for LLM and embeddings"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
from .base import LLMProvider, EmbeddingProvider
from ..data.models import ConversationData, FacetValue

logger = logging.getLogger(__name__)

class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4000)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise
    
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract facets from conversations using OpenAI"""
        
        # Create the facet extraction prompt
        prompt = self._create_facet_prompt(conversations)
        
        try:
            response = await self.generate_completion(prompt)
            return self._parse_facet_response(response, len(conversations))
            
        except Exception as e:
            logger.error(f"Facet extraction error: {e}")
            # Return empty facets on error
            return [[] for _ in conversations]
    
    def _create_facet_prompt(self, conversations: List[ConversationData]) -> str:
        """Create prompt for facet extraction"""
        
        conv_texts = []
        for i, conv in enumerate(conversations):
            text = conv.get_text()[:1000]  # Limit length
            conv_texts.append(f"Conversation {i+1}: {text}")
        
        conversations_text = "\n\n".join(conv_texts)
        
        return f"""Analyze the following conversations and extract key facets (themes, topics, intents) for each conversation.

For each conversation, identify:
- Primary intent/purpose
- Emotional tone
- Key topics discussed
- User satisfaction level
- Issue category (if applicable)

{conversations_text}

Please respond with a JSON array where each element corresponds to a conversation and contains an array of facets in this format:
[
  [
    {{"name": "intent", "value": "support_request", "confidence": 0.9}},
    {{"name": "emotion", "value": "frustrated", "confidence": 0.7}},
    {{"name": "category", "value": "technical_issue", "confidence": 0.8}}
  ],
  ...
]

Respond with only the JSON array, no additional text."""
    
    def _parse_facet_response(self, response: str, expected_count: int) -> List[List[FacetValue]]:
        """Parse facet extraction response"""
        try:
            import json
            
            # Try to find JSON in response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start == -1 or end == 0:
                logger.warning("No JSON array found in facet response")
                return [[] for _ in range(expected_count)]
            
            json_str = response[start:end]
            facets_data = json.loads(json_str)
            
            result = []
            for conv_facets in facets_data:
                facet_values = []
                for facet in conv_facets:
                    if isinstance(facet, dict) and 'name' in facet and 'value' in facet:
                        facet_values.append(FacetValue(
                            name=facet['name'],
                            value=facet['value'],
                            confidence=facet.get('confidence', 1.0)
                        ))
                result.append(facet_values)
            
            # Pad with empty lists if needed
            while len(result) < expected_count:
                result.append([])
            
            return result[:expected_count]
            
        except Exception as e:
            logger.error(f"Error parsing facet response: {e}")
            return [[] for _ in range(expected_count)]

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        super().__init__(api_key, model)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Model dimensions
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model_dimensions.get(self.model, 1536)