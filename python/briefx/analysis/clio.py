"""
Clio Methodology Implementation - Privacy-Preserving Conversation Analysis

Based on the Clio research paper: https://arxiv.org/html/2412.13678v1
This module implements the complete Clio pipeline for analyzing conversations
while preserving privacy through k-anonymity and PII protection.
"""

import asyncio
import json
import logging
import re
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import hashlib

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer

import briefcase_ai

from ..data.models import ConversationData
from ..providers.base import LLMProvider, EmbeddingProvider
from .session_manager import SessionManager

logger = logging.getLogger(__name__)

class SummarizationLevel(Enum):
    """Different levels of conversation summarization"""
    BRIEF = "brief"
    MODERATE = "moderate"
    DETAILED = "detailed"

class ClioAnalysisStage(Enum):
    """Analysis pipeline stages"""
    PREPROCESSING = "preprocessing"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"
    HIERARCHY = "hierarchy"
    PATTERN_DISCOVERY = "pattern_discovery"
    PRIVACY_AUDIT = "privacy_audit"
    COMPLETE = "complete"

@dataclass
class PrivacyConfig:
    """Privacy configuration for Clio analysis"""
    k_anonymity: int = 5
    remove_pii: bool = True
    redact_sensitive: bool = True
    min_cluster_size: int = 10
    privacy_level: str = "standard"  # standard, high, maximum
    audit_enabled: bool = True

@dataclass
class ClusteringConfig:
    """Clustering configuration"""
    algorithm: str = "auto"  # auto, dbscan, hierarchical
    min_samples: int = 5
    eps: float = 0.3
    max_clusters: int = 50
    silhouette_threshold: float = 0.3

@dataclass
class ClioPattern:
    """Represents a discovered pattern in conversations"""
    id: str
    name: str
    description: str
    support: float  # Frequency of pattern
    confidence: float  # Strength of pattern
    cluster_ids: List[str]
    keywords: List[str]
    examples: List[str]
    privacy_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClioCluster:
    """Enhanced cluster with Clio-specific metadata"""
    id: str
    name: str
    summary: str
    size: int
    centroid: List[float]
    conversation_ids: List[int]
    keywords: List[str]
    sentiment_score: float
    privacy_level: str
    anonymity_group: Optional[str]
    patterns: List[str]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClioHierarchy:
    """Hierarchical cluster structure"""
    root_id: str
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Tuple[str, str]]
    depth: int
    total_nodes: int

@dataclass
class ClioConversation:
    """Privacy-enhanced conversation data"""
    id: str
    original_id: str
    content: str
    metadata: Dict[str, Any]
    privacy_level: str
    pii_removed: bool
    embedding: Optional[List[float]] = None
    cluster_assignment: Optional[str] = None
    anonymity_group: Optional[str] = None

@dataclass
class ClioVisualizationData:
    """Complete analysis results for visualization"""
    clusters: List[ClioCluster]
    patterns: List[ClioPattern]
    hierarchy: ClioHierarchy
    conversations: List[ClioConversation]
    total_conversations: int
    analysis_cost: float
    privacy_audit: Dict[str, Any]
    stage: ClioAnalysisStage
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClioAnalysisRequest:
    """Request for Clio analysis"""
    conversations: List[ConversationData]
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)
    min_cluster_size: int = 10
    max_hierarchy_depth: int = 5
    pattern_threshold: float = 0.3
    summarization_level: SummarizationLevel = SummarizationLevel.MODERATE

class PIIDetector:
    """Privacy-preserving PII detection and redaction backed by briefcase_ai.Sanitizer"""

    # Map briefcase_ai singular type names → plural keys used throughout the codebase
    _TYPE_MAP = {'email': 'emails', 'phone': 'phones', 'ssn': 'ssns', 'credit_card': 'credit_cards'}
    # Map briefcase_ai redaction placeholders → legacy bracket format
    _PLACEHOLDER_MAP = {
        '[REDACTED_EMAIL]': '[EMAIL]',
        '[REDACTED_PHONE]': '[PHONE]',
        '[REDACTED_SSN]': '[SSN]',
        '[REDACTED_CREDIT_CARD]': '[CREDIT_CARD]',
    }

    def __init__(self):
        self._sanitizer = briefcase_ai.Sanitizer()

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect various types of PII in text"""
        analysis = self._sanitizer.analyze_pii(text)
        detected: Dict[str, List[str]] = {}
        for pii_type in analysis.get('detected_types', []):
            key = self._TYPE_MAP.get(pii_type, pii_type + 's')
            detected.setdefault(key, ['[detected]'])
        return detected

    def redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redact PII from text and return per-type redaction counts"""
        result = self._sanitizer.sanitize(text)
        sanitized = result.sanitized
        for src, dst in self._PLACEHOLDER_MAP.items():
            sanitized = sanitized.replace(src, dst)
        redaction_count: Dict[str, int] = {}
        for redaction in result.redactions:
            pii_type = getattr(redaction, 'pii_type', 'unknown')
            key = self._TYPE_MAP.get(pii_type, pii_type + 's')
            redaction_count[key] = redaction_count.get(key, 0) + 1
        return sanitized, redaction_count

class ClioHierarchyBuilder:
    """Build hierarchical cluster structure"""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
    
    def build_hierarchy(
        self, 
        clusters: List[ClioCluster], 
        embeddings: np.ndarray
    ) -> ClioHierarchy:
        """Build hierarchical structure from clusters"""
        if len(clusters) <= 1:
            root_id = clusters[0].id if clusters else "empty"
            return ClioHierarchy(
                root_id=root_id,
                nodes={root_id: {"cluster_id": root_id, "children": [], "level": 0}},
                edges=[],
                depth=1,
                total_nodes=1
            )
        
        # Create cluster centroids matrix
        centroids = np.array([cluster.centroid for cluster in clusters])
        cluster_ids = [cluster.id for cluster in clusters]
        
        # Build hierarchy using agglomerative clustering
        hierarchy = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage='ward'
        ).fit(centroids)
        
        # Build tree structure
        nodes = {}
        edges = []
        node_counter = 0
        
        # Add leaf nodes (original clusters)
        for i, cluster in enumerate(clusters):
            nodes[cluster.id] = {
                "cluster_id": cluster.id,
                "children": [],
                "level": 0,
                "size": cluster.size,
                "is_leaf": True
            }
        
        # Add internal nodes from hierarchy
        n_clusters = len(clusters)
        for i in range(n_clusters - 1):
            left_child = hierarchy.children_[i][0]
            right_child = hierarchy.children_[i][1]
            
            internal_node_id = f"internal_{i}"
            nodes[internal_node_id] = {
                "cluster_id": internal_node_id,
                "children": [],
                "level": i + 1,
                "is_leaf": False
            }
            
            # Add edges
            left_id = cluster_ids[left_child] if left_child < n_clusters else f"internal_{left_child - n_clusters}"
            right_id = cluster_ids[right_child] if right_child < n_clusters else f"internal_{right_child - n_clusters}"
            
            edges.append((internal_node_id, left_id))
            edges.append((internal_node_id, right_id))
            nodes[internal_node_id]["children"] = [left_id, right_id]
        
        root_id = f"internal_{n_clusters - 2}" if n_clusters > 1 else cluster_ids[0]
        
        return ClioHierarchy(
            root_id=root_id,
            nodes=nodes,
            edges=edges,
            depth=min(self.max_depth, n_clusters),
            total_nodes=len(nodes)
        )

class ClioPatternDiscovery:
    """Discover patterns across conversation clusters"""
    
    def __init__(self, llm_provider: LLMProvider, min_support: float = 0.1):
        self.llm_provider = llm_provider
        self.min_support = min_support
    
    async def discover_patterns(
        self, 
        clusters: List[ClioCluster], 
        conversations: List[ClioConversation]
    ) -> List[ClioPattern]:
        """Discover patterns across clusters using LLM analysis"""
        patterns = []
        
        # Create conversation lookup
        conv_lookup = {conv.id: conv for conv in conversations}
        
        # Analyze each cluster for internal patterns
        for cluster in clusters:
            cluster_patterns = await self._analyze_cluster_patterns(cluster, conv_lookup)
            patterns.extend(cluster_patterns)
        
        # Find cross-cluster patterns
        cross_patterns = await self._find_cross_cluster_patterns(clusters, conv_lookup)
        patterns.extend(cross_patterns)
        
        # Filter patterns by support and confidence
        filtered_patterns = self._filter_patterns(patterns)
        
        return filtered_patterns
    
    async def _analyze_cluster_patterns(
        self, 
        cluster: ClioCluster, 
        conv_lookup: Dict[str, ClioConversation]
    ) -> List[ClioPattern]:
        """Analyze patterns within a single cluster"""
        # Get sample conversations from cluster
        sample_convs = []
        for conv_id in cluster.conversation_ids[:10]:  # Sample first 10
            conv_key = str(conv_id)
            if conv_key in conv_lookup:
                sample_convs.append(conv_lookup[conv_key].content[:500])  # First 500 chars
        
        if not sample_convs:
            return []
        
        # Use LLM to identify patterns
        prompt = f"""Analyze these conversations from a cluster named "{cluster.name}" and identify common patterns:

Conversations:
{chr(10).join([f"{i+1}. {conv}" for i, conv in enumerate(sample_convs)])}

Identify:
1. Common themes or topics (3-5)
2. Recurring keywords or phrases (5-10)
3. Communication patterns or styles
4. Problem types or categories

Format response as JSON with patterns array containing name, description, keywords, and confidence (0-1).
"""
        
        try:
            response = await self.llm_provider.generate_completion(prompt, max_tokens=1000)
            # Parse LLM response to extract patterns
            patterns_data = self._parse_pattern_response(response)
            
            patterns = []
            for pattern_data in patterns_data:
                pattern = ClioPattern(
                    id=str(uuid.uuid4()),
                    name=pattern_data.get('name', 'Unnamed Pattern'),
                    description=pattern_data.get('description', ''),
                    support=cluster.size / 100.0,  # Rough support calculation
                    confidence=pattern_data.get('confidence', 0.5),
                    cluster_ids=[cluster.id],
                    keywords=pattern_data.get('keywords', []),
                    examples=sample_convs[:3],
                    privacy_score=self._calculate_privacy_score(pattern_data.get('keywords', []))
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing cluster patterns: {e}")
            return []
    
    async def _find_cross_cluster_patterns(
        self, 
        clusters: List[ClioCluster], 
        conv_lookup: Dict[str, ClioConversation]
    ) -> List[ClioPattern]:
        """Find patterns that span multiple clusters"""
        # Simple cross-cluster analysis based on keywords
        keyword_clusters = defaultdict(list)
        
        for cluster in clusters:
            for keyword in cluster.keywords:
                keyword_clusters[keyword].append(cluster.id)
        
        cross_patterns = []
        for keyword, cluster_ids in keyword_clusters.items():
            if len(cluster_ids) >= 2:  # Pattern spans at least 2 clusters
                support = sum(cluster.size for cluster in clusters if cluster.id in cluster_ids) / len(conv_lookup)
                if support >= self.min_support:
                    pattern = ClioPattern(
                        id=str(uuid.uuid4()),
                        name=f"Cross-Cluster: {keyword.title()}",
                        description=f"Pattern involving '{keyword}' across multiple clusters",
                        support=support,
                        confidence=0.7,
                        cluster_ids=cluster_ids,
                        keywords=[keyword],
                        examples=[],
                        privacy_score=0.8  # Cross-cluster patterns are generally safer
                    )
                    cross_patterns.append(pattern)
        
        return cross_patterns
    
    def _parse_pattern_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract pattern data"""
        try:
            # Try to find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                return data.get('patterns', [])
        except:
            pass
        
        # Fallback: basic parsing
        patterns = []
        lines = response.split('\n')
        current_pattern = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Name:'):
                if current_pattern:
                    patterns.append(current_pattern)
                current_pattern = {'name': line[5:].strip()}
            elif line.startswith('Description:') and current_pattern:
                current_pattern['description'] = line[12:].strip()
            elif line.startswith('Keywords:') and current_pattern:
                keywords = [kw.strip() for kw in line[9:].split(',')]
                current_pattern['keywords'] = keywords
            elif line.startswith('Confidence:') and current_pattern:
                try:
                    current_pattern['confidence'] = float(line[11:].strip())
                except:
                    current_pattern['confidence'] = 0.5
        
        if current_pattern:
            patterns.append(current_pattern)
        
        return patterns
    
    def _calculate_privacy_score(self, keywords: List[str]) -> float:
        """Calculate privacy score based on keyword sensitivity"""
        sensitive_keywords = {'personal', 'private', 'confidential', 'secret', 'password', 'ssn', 'credit'}
        sensitive_count = sum(1 for kw in keywords if any(sens in kw.lower() for sens in sensitive_keywords))
        return max(0.1, 1.0 - (sensitive_count / max(1, len(keywords))))
    
    def _filter_patterns(self, patterns: List[ClioPattern]) -> List[ClioPattern]:
        """Filter patterns by support and confidence thresholds"""
        return [
            pattern for pattern in patterns 
            if pattern.support >= self.min_support and pattern.confidence >= 0.3
        ]

class ClioPrivacyAuditor:
    """Audit privacy compliance of analysis results"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.pii_detector = PIIDetector()
    
    def audit_analysis(
        self, 
        conversations: List[ClioConversation], 
        clusters: List[ClioCluster]
    ) -> Dict[str, Any]:
        """Perform comprehensive privacy audit"""
        audit_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_conversations': len(conversations),
            'privacy_config': {
                'k_anonymity': self.config.k_anonymity,
                'remove_pii': self.config.remove_pii,
                'min_cluster_size': self.config.min_cluster_size
            },
            'pii_analysis': self._audit_pii(conversations),
            'anonymity_analysis': self._audit_anonymity(conversations, clusters),
            'cluster_analysis': self._audit_clusters(clusters),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        audit_results['overall_score'] = self._calculate_overall_score(audit_results)
        audit_results['recommendations'] = self._generate_recommendations(audit_results)
        
        return audit_results
    
    def _audit_pii(self, conversations: List[ClioConversation]) -> Dict[str, Any]:
        """Audit PII handling"""
        total_pii_instances = 0
        pii_types = defaultdict(int)
        redacted_conversations = 0
        
        for conv in conversations:
            pii_found = self.pii_detector.detect_pii(conv.content)
            for pii_type, instances in pii_found.items():
                if instances:
                    pii_types[pii_type] += len(instances)
                    total_pii_instances += len(instances)
            
            if conv.pii_removed:
                redacted_conversations += 1
        
        return {
            'total_pii_instances': total_pii_instances,
            'pii_types': dict(pii_types),
            'redacted_conversations': redacted_conversations,
            'redaction_rate': redacted_conversations / max(1, len(conversations)),
            'compliance': total_pii_instances == 0 if self.config.remove_pii else True
        }
    
    def _audit_anonymity(
        self, 
        conversations: List[ClioConversation], 
        clusters: List[ClioCluster]
    ) -> Dict[str, Any]:
        """Audit k-anonymity compliance"""
        group_sizes = defaultdict(int)
        for conv in conversations:
            if conv.anonymity_group:
                group_sizes[conv.anonymity_group] += 1
        
        min_group_size = min(group_sizes.values()) if group_sizes else 0
        compliant_groups = sum(1 for size in group_sizes.values() if size >= self.config.k_anonymity)
        total_groups = len(group_sizes)
        
        return {
            'k_anonymity_target': self.config.k_anonymity,
            'min_group_size': min_group_size,
            'total_anonymity_groups': total_groups,
            'compliant_groups': compliant_groups,
            'compliance_rate': compliant_groups / max(1, total_groups),
            'k_anonymity_satisfied': min_group_size >= self.config.k_anonymity
        }
    
    def _audit_clusters(self, clusters: List[ClioCluster]) -> Dict[str, Any]:
        """Audit cluster size compliance"""
        small_clusters = [c for c in clusters if c.size < self.config.min_cluster_size]
        total_in_small = sum(c.size for c in small_clusters)
        
        return {
            'total_clusters': len(clusters),
            'small_clusters': len(small_clusters),
            'conversations_in_small_clusters': total_in_small,
            'min_cluster_size_target': self.config.min_cluster_size,
            'size_compliance': len(small_clusters) == 0
        }
    
    def _calculate_overall_score(self, audit_results: Dict[str, Any]) -> float:
        """Calculate overall privacy compliance score (0-1)"""
        scores = []
        
        # PII compliance
        pii_score = 1.0 if audit_results['pii_analysis']['compliance'] else 0.5
        scores.append(pii_score)
        
        # Anonymity compliance
        anon_score = audit_results['anonymity_analysis']['compliance_rate']
        scores.append(anon_score)
        
        # Cluster size compliance
        cluster_score = 1.0 if audit_results['cluster_analysis']['size_compliance'] else 0.7
        scores.append(cluster_score)
        
        return sum(scores) / len(scores)
    
    def _generate_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate privacy improvement recommendations"""
        recommendations = []
        
        if not audit_results['pii_analysis']['compliance']:
            recommendations.append("Enable PII removal to improve privacy protection")
        
        if audit_results['anonymity_analysis']['compliance_rate'] < 0.8:
            recommendations.append(f"Increase k-anonymity parameter to improve group sizes")
        
        if not audit_results['cluster_analysis']['size_compliance']:
            recommendations.append("Increase minimum cluster size to enhance privacy")
        
        if audit_results['overall_score'] < 0.8:
            recommendations.append("Consider raising privacy level to 'high' for better protection")
        
        return recommendations

class ClioAnalysisPipeline:
    """Main Clio analysis pipeline"""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        session_manager: Optional[SessionManager] = None
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.session_manager = session_manager
        self.pii_detector = PIIDetector()
        
    async def analyze(self, request: ClioAnalysisRequest) -> ClioVisualizationData:
        """Execute complete Clio analysis pipeline"""
        session_id = str(uuid.uuid4())
        
        if self.session_manager:
            self.session_manager.create_session(session_id, total_steps=6)
            self.session_manager.update_progress(session_id, 0, "Starting Clio analysis...")
        
        try:
            # Stage 1: Preprocessing and Privacy Protection
            conversations = await self._preprocess_conversations(request, session_id)
            
            # Stage 2: Generate Embeddings
            await self._generate_embeddings(conversations, session_id)
            
            # Stage 3: Clustering
            clusters = await self._perform_clustering(conversations, request, session_id)
            
            # Stage 4: Build Hierarchy
            hierarchy = await self._build_hierarchy(clusters, conversations, session_id)
            
            # Stage 5: Pattern Discovery
            patterns = await self._discover_patterns(clusters, conversations, session_id)
            
            # Stage 6: Privacy Audit
            privacy_audit = await self._perform_privacy_audit(conversations, clusters, request, session_id)
            
            # Calculate analysis cost
            analysis_cost = self._calculate_cost(conversations, patterns)
            
            result = ClioVisualizationData(
                clusters=clusters,
                patterns=patterns,
                hierarchy=hierarchy,
                conversations=conversations,
                total_conversations=len(conversations),
                analysis_cost=analysis_cost,
                privacy_audit=privacy_audit,
                stage=ClioAnalysisStage.COMPLETE,
                metadata={
                    'session_id': session_id,
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'privacy_config': request.privacy_config.__dict__,
                    'clustering_config': request.clustering_config.__dict__
                }
            )
            
            if self.session_manager:
                self.session_manager.update_progress(session_id, 100, "Clio analysis complete!")
            
            return result
            
        except Exception as e:
            if self.session_manager:
                self.session_manager.update_progress(session_id, -1, f"Error: {str(e)}")
            raise
    
    async def _preprocess_conversations(
        self, 
        request: ClioAnalysisRequest, 
        session_id: str
    ) -> List[ClioConversation]:
        """Preprocess conversations with privacy protection"""
        if self.session_manager:
            self.session_manager.update_progress(session_id, 10, "Preprocessing conversations...")
        
        conversations = []
        total_redactions = 0
        
        for i, conv_data in enumerate(request.conversations):
            # Create Clio conversation
            clio_conv = ClioConversation(
                id=str(i),
                original_id=getattr(conv_data, 'id', str(i)),
                content=conv_data.content,
                metadata=getattr(conv_data, 'metadata', {}),
                privacy_level=request.privacy_config.privacy_level,
                pii_removed=False
            )
            
            # Apply privacy protection
            if request.privacy_config.remove_pii:
                redacted_content, redaction_count = self.pii_detector.redact_pii(clio_conv.content)
                clio_conv.content = redacted_content
                clio_conv.pii_removed = sum(redaction_count.values()) > 0
                total_redactions += sum(redaction_count.values())
            
            # Assign anonymity group (simple hash-based grouping for now)
            content_hash = hashlib.md5(clio_conv.content.encode()).hexdigest()
            clio_conv.anonymity_group = f"group_{content_hash[:8]}"
            
            conversations.append(clio_conv)
        
        logger.info(f"Preprocessed {len(conversations)} conversations, removed {total_redactions} PII instances")
        return conversations
    
    async def _generate_embeddings(self, conversations: List[ClioConversation], session_id: str):
        """Generate embeddings for conversations"""
        if self.session_manager:
            self.session_manager.update_progress(session_id, 25, "Generating embeddings...")
        
        texts = [conv.content for conv in conversations]
        embeddings = await self.embedding_provider.embed_texts(texts)
        
        for conv, embedding in zip(conversations, embeddings):
            conv.embedding = embedding
    
    async def _perform_clustering(
        self, 
        conversations: List[ClioConversation], 
        request: ClioAnalysisRequest, 
        session_id: str
    ) -> List[ClioCluster]:
        """Perform privacy-aware clustering"""
        if self.session_manager:
            self.session_manager.update_progress(session_id, 50, "Clustering conversations...")
        
        embeddings = np.array([conv.embedding for conv in conversations])
        
        # Use DBSCAN for privacy-friendly clustering
        clustering = DBSCAN(
            eps=request.clustering_config.eps,
            min_samples=max(request.clustering_config.min_samples, request.privacy_config.k_anonymity)
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Create clusters
        clusters = []
        cluster_conversations = defaultdict(list)
        
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                cluster_conversations[label].append(i)
        
        # Filter out clusters smaller than min_cluster_size
        valid_clusters = {
            label: conv_ids for label, conv_ids in cluster_conversations.items()
            if len(conv_ids) >= request.privacy_config.min_cluster_size
        }
        
        for label, conv_ids in valid_clusters.items():
            cluster_id = f"cluster_{label}"
            
            # Calculate centroid
            cluster_embeddings = embeddings[conv_ids]
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            
            # Generate cluster summary
            sample_texts = [conversations[i].content[:200] for i in conv_ids[:5]]
            cluster_name = await self._generate_cluster_name(sample_texts)
            cluster_summary = await self._generate_cluster_summary(sample_texts)
            
            # Extract keywords
            keywords = self._extract_keywords(sample_texts)
            
            # Calculate quality metrics
            if len(cluster_embeddings) > 1:
                # Intra-cluster distance (lower is better)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                cohesion = float(np.mean(distances))
            else:
                cohesion = 0.0
            
            cluster = ClioCluster(
                id=cluster_id,
                name=cluster_name,
                summary=cluster_summary,
                size=len(conv_ids),
                centroid=centroid,
                conversation_ids=conv_ids,
                keywords=keywords,
                sentiment_score=0.0,  # TODO: Implement sentiment analysis
                privacy_level=request.privacy_config.privacy_level,
                anonymity_group=f"cluster_group_{label}",
                patterns=[],
                quality_metrics={'cohesion': cohesion}
            )
            
            clusters.append(cluster)
            
            # Update conversation cluster assignments
            for conv_id in conv_ids:
                conversations[conv_id].cluster_assignment = cluster_id
        
        return clusters
    
    async def _build_hierarchy(
        self, 
        clusters: List[ClioCluster], 
        conversations: List[ClioConversation], 
        session_id: str
    ) -> ClioHierarchy:
        """Build cluster hierarchy"""
        if self.session_manager:
            self.session_manager.update_progress(session_id, 70, "Building hierarchy...")
        
        if not clusters:
            return ClioHierarchy(
                root_id="empty",
                nodes={},
                edges=[],
                depth=0,
                total_nodes=0
            )
        
        hierarchy_builder = ClioHierarchyBuilder()
        embeddings = np.array([cluster.centroid for cluster in clusters])
        
        return hierarchy_builder.build_hierarchy(clusters, embeddings)
    
    async def _discover_patterns(
        self, 
        clusters: List[ClioCluster], 
        conversations: List[ClioConversation], 
        session_id: str
    ) -> List[ClioPattern]:
        """Discover patterns across clusters"""
        if self.session_manager:
            self.session_manager.update_progress(session_id, 85, "Discovering patterns...")
        
        pattern_discovery = ClioPatternDiscovery(self.llm_provider)
        return await pattern_discovery.discover_patterns(clusters, conversations)
    
    async def _perform_privacy_audit(
        self, 
        conversations: List[ClioConversation], 
        clusters: List[ClioCluster], 
        request: ClioAnalysisRequest, 
        session_id: str
    ) -> Dict[str, Any]:
        """Perform privacy compliance audit"""
        if self.session_manager:
            self.session_manager.update_progress(session_id, 95, "Auditing privacy compliance...")
        
        auditor = ClioPrivacyAuditor(request.privacy_config)
        return auditor.audit_analysis(conversations, clusters)
    
    async def _generate_cluster_name(self, sample_texts: List[str]) -> str:
        """Generate a descriptive name for a cluster"""
        combined_text = " ".join(sample_texts)[:1000]  # Limit text length
        
        prompt = f"""Based on these conversation samples, suggest a concise, descriptive name (2-4 words) for this topic cluster:

{combined_text}

Respond with just the cluster name, no explanation."""
        
        try:
            name = await self.llm_provider.generate_completion(prompt, max_tokens=20)
            return name.strip().replace('"', '').replace("'", '')
        except:
            # Fallback to keyword-based naming
            keywords = self._extract_keywords(sample_texts)
            return f"Topic: {', '.join(keywords[:2])}" if keywords else "Unnamed Topic"
    
    async def _generate_cluster_summary(self, sample_texts: List[str]) -> str:
        """Generate a summary for a cluster"""
        combined_text = " ".join(sample_texts)[:1500]
        
        prompt = f"""Summarize the main theme and characteristics of these conversations in 1-2 sentences:

{combined_text}"""
        
        try:
            return await self.llm_provider.generate_completion(prompt, max_tokens=100)
        except:
            return "A collection of related conversations."
    
    def _extract_keywords(self, texts: List[str]) -> List[str]:
        """Extract keywords from texts using TF-IDF"""
        if not texts:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=10,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-5:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except:
            return []
    
    def _calculate_cost(self, conversations: List[ClioConversation], patterns: List[ClioPattern]) -> float:
        """Calculate estimated analysis cost using briefcase_ai.CostCalculator"""
        cost_calc = briefcase_ai.CostCalculator()
        combined_text = ' '.join(conv.content for conv in conversations)
        pattern_text = ' '.join(pattern.description for pattern in patterns)
        full_input = combined_text + ' ' + pattern_text
        estimated_output_tokens = max(1, len(patterns) * 50)
        estimate = cost_calc.estimate_cost_from_text(
            self._model_name_for_cost(),
            full_input,
            estimated_output_tokens,
        )
        return round(estimate.total_cost, 4)

    def _model_name_for_cost(self) -> str:
        """Return the model name to use for cost estimation"""
        if self.llm_provider and hasattr(self.llm_provider, 'model'):
            return self.llm_provider.model
        return 'gpt-4o-mini'