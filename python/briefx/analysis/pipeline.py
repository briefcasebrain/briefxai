"""Main analysis pipeline that orchestrates the entire process"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from ..data.models import ConversationData, AnalysisResults, ConversationAnalysis
from ..providers.base import LLMProvider, EmbeddingProvider
from ..providers.factory import ProviderFactory
from ..utils import dedup_conversations, calculate_simple_sentiment, determine_category, generate_session_id
from .clustering import ConversationClusterer
from .dimensionality import reduce_embeddings_for_clustering, reduce_embeddings_for_visualization
from .session_manager import session_manager
from ..prompts import AdvancedPromptManager
# Configuration will be handled directly

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Main analysis pipeline for processing conversations"""
    
    def __init__(self, config=None):
        # Use simple defaults for free demo
        self.config = config or type('Config', (), {
            'llm_provider': 'demo',
            'llm_model': 'demo-analyzer',
            'llm_api_key': '',
            'llm_base_url': None,
            'embedding_provider': 'openai',
            'embedding_model': 'text-embedding-3-small',
            'embedding_api_key': ''
        })()
        
        self.llm_provider: Optional[LLMProvider] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.clusterer = ConversationClusterer(method="auto", max_clusters=10, llm_provider=None)  # Will be updated after initialization
        self.prompt_manager = AdvancedPromptManager()
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize LLM and embedding providers based on config"""
        
        # Initialize LLM provider using factory
        self.llm_provider = ProviderFactory.create_llm_provider(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url
        )
        
        # Initialize embedding provider using factory (allow failure for free demo)
        try:
            self.embedding_provider = ProviderFactory.create_embedding_provider(
                provider=self.config.embedding_provider,
                api_key=self.config.embedding_api_key,
                model=self.config.embedding_model
            )
        except Exception as e:
            logger.warning(f"Embedding provider failed to initialize: {e}. Using fallback.")
            self.embedding_provider = None
        
        # Update clusterer with LLM provider for enhanced naming
        self.clusterer.llm_provider = self.llm_provider
    
    async def analyze_conversations(
        self, 
        conversations: List[ConversationData],
        session_id: Optional[str] = None
    ) -> AnalysisResults:
        """Run the complete analysis pipeline"""
        
        if not session_id:
            session_id = generate_session_id()
        
        start_time = time.time()
        
        try:
            # Create and start session
            session_manager.create_session(
                session_id, 
                total_steps=100,
                metadata={
                    "conversation_count": len(conversations),
                    "analysis_type": "full_analysis"
                }
            )
            session_manager.start_session(session_id)
            
            # Step 1: Data validation and preprocessing (0-15%)
            await self._update_progress(session_id, 5.0, "Validating conversation data")
            
            if not conversations:
                raise ValueError("No conversations provided")
            
            await self._update_progress(session_id, 10.0, f"Processing {len(conversations)} conversations")
            
            # Deduplication if enabled
            processed_conversations = conversations
            if self.config.dedup_data:
                await self._update_progress(session_id, 12.0, "Removing duplicate conversations")
                original_count = len(conversations)
                processed_conversations = dedup_conversations(conversations)
                removed_count = original_count - len(processed_conversations)
                logger.info(f"Removed {removed_count} duplicate conversations")
            
            await self._update_progress(session_id, 15.0, f"Ready to analyze {len(processed_conversations)} conversations")
            
            # Step 2: Generate embeddings (15-40%)
            embeddings = []
            reduced_embeddings_for_clustering = []
            visualization_embeddings = []
            embedding_metadata = {}
            
            if self.embedding_provider:
                await self._update_progress(session_id, 20.0, "Generating embeddings")
                embeddings = await self._generate_embeddings(processed_conversations, session_id)
                await self._update_progress(session_id, 32.0, "Embeddings generated")
                
                # Step 2.1: Dimensionality reduction (32-40%)
                if embeddings:
                    await self._update_progress(session_id, 35.0, "Reducing dimensionality for clustering")
                    reduced_embeddings_for_clustering, clustering_meta = reduce_embeddings_for_clustering(
                        embeddings, target_dim=min(50, len(embeddings) // 2)
                    )
                    embedding_metadata['clustering_reduction'] = clustering_meta
                    
                    await self._update_progress(session_id, 38.0, "Reducing dimensionality for visualization")
                    visualization_embeddings, viz_meta = reduce_embeddings_for_visualization(embeddings)
                    embedding_metadata['visualization_reduction'] = viz_meta
                    
                    logger.info(f"Reduced embeddings: clustering={reduced_embeddings_for_clustering.shape}, viz={visualization_embeddings.shape}")
                
                await self._update_progress(session_id, 40.0, "Dimensionality reduction completed")
            else:
                await self._update_progress(session_id, 40.0, "Skipping embeddings (no provider configured)")
            
            # Step 3: Extract facets (40-60%)
            facets_data = []
            if self.llm_provider:
                await self._update_progress(session_id, 45.0, "Extracting conversation facets")
                facets_data = await self._extract_facets(processed_conversations, session_id)
                await self._update_progress(session_id, 60.0, "Facet extraction complete")
            else:
                await self._update_progress(session_id, 60.0, "Skipping facet extraction (no LLM provider)")
                facets_data = [[] for _ in processed_conversations]
            
            # Step 4: Perform clustering (60-80%)
            await self._update_progress(session_id, 65.0, "Clustering conversations")
            # Use reduced embeddings for clustering if available, otherwise original embeddings
            clustering_embeddings = reduced_embeddings_for_clustering if len(reduced_embeddings_for_clustering) > 0 else embeddings
            clusters = self.clusterer.cluster_conversations(processed_conversations, clustering_embeddings)
            await self._update_progress(session_id, 80.0, f"Created {len(clusters)} clusters")
            
            # Step 5: Generate final analysis (80-100%)
            await self._update_progress(session_id, 85.0, "Generating conversation analysis")
            conversation_analyses = self._create_conversation_analyses(
                processed_conversations, facets_data, clusters, embeddings, visualization_embeddings
            )
            
            await self._update_progress(session_id, 95.0, "Finalizing results")
            
            # Create results
            end_time = time.time()
            processing_time = end_time - start_time
            
            results = AnalysisResults(
                conversations=conversation_analyses,
                clusters=clusters,
                total_conversations=len(processed_conversations),
                total_messages=sum(len(conv.messages) for conv in processed_conversations),
                processing_time=processing_time,
                session_id=session_id
            )
            
            await self._update_progress(session_id, 100.0, f"Analysis complete in {processing_time:.2f}s")
            session_manager.complete_session(
                session_id, 
                f"Analysis completed successfully in {processing_time:.2f}s",
                {
                    "processing_time": processing_time,
                    "total_conversations": len(processed_conversations),
                    "total_clusters": len(clusters)
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            session_manager.fail_session(session_id, str(e), {"error_type": type(e).__name__})
            raise
    
    async def _generate_embeddings(self, conversations: List[ConversationData], session_id: str) -> List[List[float]]:
        """Generate embeddings for conversations"""
        
        if not self.embedding_provider:
            return []
        
        texts = [conv.get_text() for conv in conversations]
        
        try:
            # Process in batches
            batch_size = self.config.embedding_batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                progress = 20.0 + (i / len(texts)) * 20.0  # 20-40%
                
                await self._update_progress(
                    session_id, 
                    progress, 
                    f"Generating embeddings batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                )
                
                batch_embeddings = await self.embedding_provider.generate_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def _extract_facets(self, conversations: List[ConversationData], session_id: str) -> List[List]:
        """Extract facets from conversations using advanced prompting"""
        
        if not self.llm_provider:
            return [[] for _ in conversations]
        
        try:
            # Process in smaller batches for facet extraction
            batch_size = min(self.config.llm_batch_size, 5)  # Smaller batches for LLM
            all_facets = []
            
            for i in range(0, len(conversations), batch_size):
                batch_conversations = conversations[i:i + batch_size]
                progress = 45.0 + (i / len(conversations)) * 15.0  # 45-60%
                
                await self._update_progress(
                    session_id,
                    progress,
                    f"Extracting facets batch {i//batch_size + 1}/{(len(conversations) + batch_size - 1)//batch_size}"
                )
                
                # Use advanced prompting system for facet extraction
                for conv in batch_conversations:
                    prompt = self.prompt_manager.get_prompt(
                        "facet_extraction",
                        conversation=conv.get_text(),
                        metadata={
                            "message_count": len(conv.messages),
                            "session_id": session_id
                        }
                    )
                    
                    # Execute with the selected prompt template
                    start_time = time.time()
                    facets = await self.llm_provider.extract_facets([conv])
                    execution_time = time.time() - start_time
                    
                    # Record performance metrics
                    self.prompt_manager.record_result(
                        "facet_extraction",
                        prompt.version,
                        execution_time,
                        success=len(facets) > 0
                    )
                    
                    all_facets.extend(facets)
                
                # Delay between batches to respect rate limits
                await asyncio.sleep(1.0)
            
            logger.info(f"Extracted facets for {len(all_facets)} conversations")
            
            # Periodically optimize templates based on performance
            if len(all_facets) % 100 == 0:
                self.prompt_manager.optimize_templates("facet_extraction")
            
            return all_facets
            
        except Exception as e:
            logger.error(f"Facet extraction failed: {e}")
            return [[] for _ in conversations]
    
    def _create_conversation_analyses(
        self,
        conversations: List[ConversationData],
        facets_data: List[List],
        clusters: List,
        embeddings: List[List[float]],
        visualization_embeddings: List[List[float]] = None
    ) -> List[ConversationAnalysis]:
        """Create ConversationAnalysis objects"""
        
        # Create cluster lookup
        cluster_lookup = {}
        for cluster in clusters:
            for conv in cluster.conversations:
                cluster_lookup[id(conv)] = cluster
        
        analyses = []
        for i, conv in enumerate(conversations):
            # Get cluster info
            cluster = cluster_lookup.get(id(conv))
            
            # Get facets
            facets = facets_data[i] if i < len(facets_data) else []
            
            # Get embedding
            embedding = embeddings[i] if i < len(embeddings) else None
            
            # Get visualization embedding
            viz_embedding = None
            if visualization_embeddings and hasattr(visualization_embeddings, 'shape'):
                # Handle numpy array
                viz_embedding = visualization_embeddings[i].tolist() if i < visualization_embeddings.shape[0] else None
            elif visualization_embeddings and isinstance(visualization_embeddings, list):
                # Handle regular list
                viz_embedding = visualization_embeddings[i] if i < len(visualization_embeddings) else None
            
            # Calculate sentiment
            sentiment = calculate_simple_sentiment(conv.get_text())
            
            # Determine category
            category = determine_category(conv.get_text())
            
            analysis = ConversationAnalysis(
                conversation=conv,
                cluster_id=cluster.id if cluster else None,
                cluster_name=cluster.name if cluster else None,
                sentiment=sentiment,
                category=category,
                facets=facets,
                embedding=embedding,
                visualization_embedding=viz_embedding,
                topics=self._extract_topics(conv)
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def _extract_topics(self, conversation: ConversationData) -> List[str]:
        """Extract basic topics from conversation"""
        text = conversation.get_text().lower()
        
        # Simple topic detection based on keywords
        topics = []
        
        topic_keywords = {
            'authentication': ['login', 'password', 'signin', 'signup', 'auth'],
            'billing': ['payment', 'charge', 'invoice', 'billing', 'subscription'],
            'technical': ['api', 'code', 'error', 'bug', 'integration'],
            'feature': ['feature', 'functionality', 'capability', 'tool'],
            'performance': ['slow', 'fast', 'performance', 'speed', 'optimization'],
            'ui_ux': ['interface', 'design', 'user experience', 'dashboard', 'navigation']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics[:5]  # Return top 5 topics
    
    async def _update_progress(self, session_id: str, progress: float, message: str):
        """Update progress for session"""
        session_manager.update_progress(session_id, progress, message)
        logger.debug(f"Progress {progress:.1f}%: {message}")

# Global pipeline instance (will be initialized with config)
pipeline: Optional[AnalysisPipeline] = None

def initialize_pipeline(config=None):
    """Initialize the global pipeline instance"""
    global pipeline
    pipeline = AnalysisPipeline(config)

def get_pipeline() -> AnalysisPipeline:
    """Get the global pipeline instance"""
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
    return pipeline