"""Demo provider for free operation without API keys"""

import asyncio
import logging
import random
import re
from typing import List, Dict, Any, Optional

from .base import LLMProvider, EmbeddingProvider
from ..data.models import ConversationData, FacetValue

logger = logging.getLogger(__name__)

class DemoLLMProvider(LLMProvider):
    """Demo LLM provider that works without any API keys or external dependencies"""
    
    def __init__(self, api_key: str = "", model: str = "demo-analyzer", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        self.model_name = model
        
        # Pre-defined responses for different types of analysis
        self.demo_responses = {
            "facet_extraction": [
                "intent=information_seeking, sentiment=neutral, topic=general, urgency=low, complexity=medium",
                "intent=help_request, sentiment=positive, topic=technical, urgency=medium, complexity=high",
                "intent=complaint, sentiment=negative, topic=service, urgency=high, complexity=low",
                "intent=feedback, sentiment=mixed, topic=product, urgency=low, complexity=medium",
                "intent=inquiry, sentiment=curious, topic=features, urgency=medium, complexity=low"
            ],
            "analysis": [
                "This conversation shows typical customer interaction patterns with clear intent and moderate engagement levels.",
                "The discussion demonstrates collaborative problem-solving with multiple participants sharing perspectives.",
                "This exchange reflects standard information-seeking behavior with structured question-answer dynamics.",
                "The conversation exhibits task-oriented communication with specific goals and outcomes.",
                "This interaction displays social communication patterns with relationship-building elements."
            ],
            "summary": [
                "Summary: Participants discussed various topics with moderate engagement and clear communication patterns.",
                "Overview: The conversation involved information exchange with structured dialogue and specific objectives.",
                "Analysis: Discussion featured collaborative elements with diverse viewpoints and constructive interaction.",
                "Synopsis: Participants engaged in goal-oriented communication with measurable outcomes.",
                "Summary: The exchange demonstrated typical conversation dynamics with balanced participation."
            ]
        }
        
        logger.info(f"Initialized Demo provider with model: {self.model_name}")
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate a demo completion based on prompt analysis"""
        # Add a small delay to simulate processing
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        prompt_lower = prompt.lower()
        
        # Determine response type based on prompt content
        if any(word in prompt_lower for word in ['facet', 'extract', 'identify', 'characteristic']):
            response_type = "facet_extraction"
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'examine', 'evaluate']):
            response_type = "analysis"
        elif any(word in prompt_lower for word in ['summary', 'summarize', 'overview', 'synopsis']):
            response_type = "summary"
        else:
            response_type = "analysis"  # Default
        
        # Select a random response from the appropriate category
        response = random.choice(self.demo_responses[response_type])
        
        # Add some variation based on prompt content
        if "conversation" in prompt_lower:
            response += " The conversation patterns indicate structured communication flow."
        elif "customer" in prompt_lower:
            response += " Customer interaction metrics show standard engagement levels."
        elif "technical" in prompt_lower:
            response += " Technical discussion elements suggest solution-oriented dialogue."
        
        return response
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for generate_completion"""
        return asyncio.run(self.generate_completion(prompt, **kwargs))
    
    async def extract_facets(self, conversations: List[ConversationData]) -> List[List[FacetValue]]:
        """Extract demo facets from conversations"""
        if not conversations:
            return []
        
        facet_results = []
        
        for conversation in conversations:
            text = conversation.get_text()
            if not text.strip():
                facet_results.append([])
                continue
            
            # Generate demo facets based on text analysis heuristics
            facets = self._generate_demo_facets(text)
            facet_results.append(facets)
            
            # Small delay for realistic processing
            await asyncio.sleep(0.1)
        
        return facet_results
    
    def _generate_demo_facets(self, text: str) -> List[FacetValue]:
        """Generate realistic demo facets based on text analysis"""
        facets = []
        
        # Analyze text characteristics
        word_count = len(text.split())
        has_questions = '?' in text
        has_exclamations = '!' in text
        has_technical_terms = any(term in text.lower() for term in ['api', 'code', 'system', 'error', 'bug', 'feature'])
        has_emotional_words = any(word in text.lower() for word in ['love', 'hate', 'frustrated', 'happy', 'excited', 'disappointed'])
        
        # Intent classification
        if has_questions:
            intent = "information_seeking"
        elif "help" in text.lower() or "support" in text.lower():
            intent = "help_request"
        elif has_emotional_words and any(word in text.lower() for word in ['problem', 'issue', 'wrong']):
            intent = "complaint"
        elif "feedback" in text.lower() or "suggest" in text.lower():
            intent = "feedback"
        else:
            intent = "general_inquiry"
        
        facets.append(FacetValue(name="intent", value=intent, confidence=0.8))
        
        # Sentiment analysis
        if has_emotional_words:
            if any(word in text.lower() for word in ['love', 'happy', 'excited', 'great', 'excellent']):
                sentiment = "positive"
            elif any(word in text.lower() for word in ['hate', 'frustrated', 'disappointed', 'terrible', 'awful']):
                sentiment = "negative"
            else:
                sentiment = "mixed"
        else:
            sentiment = "neutral"
        
        facets.append(FacetValue(name="sentiment", value=sentiment, confidence=0.75))
        
        # Topic classification
        if has_technical_terms:
            topic = "technical"
        elif any(word in text.lower() for word in ['buy', 'purchase', 'price', 'cost', 'payment']):
            topic = "commercial"
        elif any(word in text.lower() for word in ['account', 'login', 'password', 'profile']):
            topic = "account_management"
        else:
            topic = "general"
        
        facets.append(FacetValue(name="topic", value=topic, confidence=0.7))
        
        # Urgency level
        if has_exclamations or any(word in text.lower() for word in ['urgent', 'asap', 'immediately', 'emergency']):
            urgency = "high"
        elif any(word in text.lower() for word in ['soon', 'quick', 'fast']):
            urgency = "medium"
        else:
            urgency = "low"
        
        facets.append(FacetValue(name="urgency", value=urgency, confidence=0.6))
        
        # Complexity level
        if word_count > 100 or has_technical_terms:
            complexity = "high"
        elif word_count > 30:
            complexity = "medium"
        else:
            complexity = "low"
        
        facets.append(FacetValue(name="complexity", value=complexity, confidence=0.8))
        
        # Length category
        if word_count > 50:
            length = "long"
        elif word_count > 15:
            length = "medium"
        else:
            length = "short"
        
        facets.append(FacetValue(name="length", value=length, confidence=0.9))
        
        # Analysis method
        facets.append(FacetValue(name="analysis_method", value="demo_heuristics", confidence=1.0))
        
        return facets
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the demo model"""
        return {
            "provider": "demo",
            "model": self.model_name,
            "capabilities": ["text_generation", "conversation_analysis", "facet_extraction"],
            "local": True,
            "requires_internet": False,
            "requires_api_key": False,
            "memory_intensive": False,
            "cost": "free",
            "description": "Demonstration provider using rule-based analysis for completely free operation"
        }


class DemoEmbeddingProvider(EmbeddingProvider):
    """Demo embedding provider that generates simple embeddings without API calls"""
    
    def __init__(self, api_key: str = "", model: str = "demo-embeddings", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        self.model_name = model
        self.embedding_dim = 384  # Standard embedding dimension
        
        # Pre-computed word vectors for common terms (simplified)
        self.word_vectors = {
            'help': [0.1, 0.2, 0.3] * 128,
            'problem': [0.2, 0.1, 0.4] * 128,
            'support': [0.3, 0.2, 0.1] * 128,
            'question': [0.1, 0.3, 0.2] * 128,
            'issue': [0.4, 0.1, 0.2] * 128,
            'technical': [0.2, 0.4, 0.1] * 128,
            'account': [0.1, 0.1, 0.4] * 128,
            'purchase': [0.3, 0.1, 0.3] * 128,
            'urgent': [0.5, 0.2, 0.1] * 128,
            'feedback': [0.2, 0.3, 0.3] * 128
        }
        
        logger.info(f"Initialized Demo embedding provider with model: {self.model_name}")
    
    async def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate demo embeddings for texts using simple heuristics"""
        embeddings = []
        
        for text in texts:
            embedding = self._generate_demo_embedding(text)
            embeddings.append(embedding)
            
            # Small delay to simulate processing
            await asyncio.sleep(0.05)
        
        return embeddings
    
    def _generate_demo_embedding(self, text: str) -> List[float]:
        """Generate a demo embedding based on text content"""
        # Initialize with random base vector (seeded by text hash for consistency)
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        random.seed(text_hash % 10000)
        
        embedding = [random.uniform(-0.1, 0.1) for _ in range(self.embedding_dim)]
        
        # Modify embedding based on content
        text_lower = text.lower()
        words = text_lower.split()
        
        # Add influence from known word vectors
        word_influence = [0.0] * self.embedding_dim
        word_count = 0
        
        for word in words:
            if word in self.word_vectors:
                for i, val in enumerate(self.word_vectors[word]):
                    if i < self.embedding_dim:
                        word_influence[i] += val
                word_count += 1
        
        # Blend base embedding with word influence
        if word_count > 0:
            blend_factor = min(0.3, word_count * 0.1)
            for i in range(self.embedding_dim):
                embedding[i] = embedding[i] * (1 - blend_factor) + (word_influence[i] / word_count) * blend_factor
        
        # Add some structure based on text characteristics
        text_length = len(text)
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        # Modify specific dimensions based on characteristics
        if text_length > 100:
            embedding[0] += 0.2  # Long text indicator
        if question_count > 0:
            embedding[1] += 0.3 * question_count  # Question indicator
        if exclamation_count > 0:
            embedding[2] += 0.3 * exclamation_count  # Exclamation indicator
        
        # Normalize to unit vector (roughly)
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the demo embedding model"""
        return {
            "provider": "demo",
            "model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "capabilities": ["text_embedding"],
            "local": True,
            "requires_internet": False,
            "requires_api_key": False,
            "cost": "free",
            "description": "Demonstration embedding provider using text heuristics for completely free operation"
        }