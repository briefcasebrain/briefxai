"""Clustering algorithms for conversation analysis"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from ..data.models import ConversationData, ConversationCluster, ConversationAnalysis
from ..utils import determine_category

logger = logging.getLogger(__name__)

class ClusterEvaluator:
    """Evaluates clustering quality using multiple metrics"""
    
    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Comprehensive clustering evaluation"""
        if len(set(labels)) <= 1:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'n_clusters': len(set(labels)),
                'n_outliers': np.sum(labels == -1) if -1 in labels else 0
            }
        
        try:
            # Filter out outliers for evaluation (DBSCAN assigns -1 to outliers)
            valid_mask = labels != -1
            if np.sum(valid_mask) <= 1:
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'n_clusters': 0,
                    'n_outliers': len(labels)
                }
            
            X_valid = X[valid_mask]
            labels_valid = labels[valid_mask]
            
            metrics = {
                'silhouette_score': silhouette_score(X_valid, labels_valid),
                'calinski_harabasz_score': calinski_harabasz_score(X_valid, labels_valid),
                'davies_bouldin_score': davies_bouldin_score(X_valid, labels_valid),
                'n_clusters': len(set(labels_valid)),
                'n_outliers': np.sum(labels == -1) if -1 in labels else 0
            }
            
            return metrics
        except Exception as e:
            logger.warning(f"Error evaluating clustering: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'n_clusters': len(set(labels)),
                'n_outliers': np.sum(labels == -1) if -1 in labels else 0
            }
    
    @staticmethod
    def select_best_clustering(clustering_results: List[Tuple[str, np.ndarray, Dict[str, float]]]) -> Tuple[str, np.ndarray]:
        """Select the best clustering based on multiple metrics"""
        if not clustering_results:
            return "none", np.array([])
        
        best_score = -float('inf')
        best_method = None
        best_labels = None
        
        for method, labels, metrics in clustering_results:
            # Composite score: weighted combination of metrics
            score = (
                0.4 * metrics.get('silhouette_score', 0) +
                0.3 * min(1.0, metrics.get('calinski_harabasz_score', 0) / 1000) +  # Normalize CH score
                0.2 * (1.0 / (1.0 + metrics.get('davies_bouldin_score', float('inf')))) +  # Lower DB is better
                0.1 * min(1.0, metrics.get('n_clusters', 0) / 10)  # Moderate cluster count bonus
            )
            
            # Penalty for too many outliers
            outlier_ratio = metrics.get('n_outliers', 0) / len(labels) if len(labels) > 0 else 0
            score -= 0.3 * outlier_ratio
            
            logger.debug(f"{method}: score={score:.3f}, metrics={metrics}")
            
            if score > best_score:
                best_score = score
                best_method = method
                best_labels = labels
        
        logger.info(f"Best clustering method: {best_method} (score: {best_score:.3f})")
        return best_method, best_labels

class ConversationClusterer:
    """Handles conversation clustering using various algorithms"""
    
    def __init__(self, method: str = "auto", max_clusters: int = 10, min_cluster_size: int = 2, llm_provider=None):
        self.method = method  # "kmeans", "hierarchical", "dbscan", "auto"
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.scaler = StandardScaler()
        self.evaluator = ClusterEvaluator()
        self.llm_provider = llm_provider
    
    def cluster_conversations(
        self, 
        conversations: List[ConversationData],
        embeddings: List[List[float]]
    ) -> List[ConversationCluster]:
        """Cluster conversations based on embeddings"""
        
        if not embeddings or len(embeddings) != len(conversations):
            logger.warning("No embeddings provided, using simple category-based clustering")
            return self._cluster_by_category(conversations)
        
        # Convert embeddings to numpy array
        X = np.array(embeddings)
        
        # Normalize embeddings
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        n_clusters = self._find_optimal_clusters(X_scaled, max_clusters=min(self.max_clusters, len(conversations)))
        
        logger.info(f"Clustering {len(conversations)} conversations with method: {self.method}")
        
        # Perform clustering
        if self.method == "auto":
            cluster_labels = self._auto_clustering(X_scaled, n_clusters)
        elif self.method == "kmeans":
            cluster_labels = self._kmeans_clustering(X_scaled, n_clusters)
        elif self.method == "hierarchical":
            cluster_labels = self._hierarchical_clustering(X_scaled, n_clusters)
        elif self.method == "dbscan":
            cluster_labels = self._dbscan_clustering(X_scaled)
        else:
            logger.warning(f"Unknown clustering method: {self.method}, falling back to auto")
            cluster_labels = self._auto_clustering(X_scaled, n_clusters)
        
        # Create cluster objects
        clusters = self._create_clusters(conversations, cluster_labels)
        
        return clusters
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int) -> int:
        """Find optimal number of clusters using silhouette score"""
        
        if len(X) < 2:
            return 1
        
        if len(X) < max_clusters:
            max_clusters = len(X)
        
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Skip if all points in one cluster
                if len(set(labels)) == 1:
                    continue
                
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for k={k}: {e}")
                continue
        
        # Default to simple heuristic if silhouette analysis fails
        if best_score == -1:
            best_k = min(5, max(2, len(X) // 3))
        
        logger.debug(f"Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def _kmeans_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(X)
    
    def _hierarchical_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform hierarchical clustering"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        return clustering.fit_predict(X)
    
    def _dbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering"""
        # Auto-determine eps using k-distance graph
        eps = self._estimate_eps(X)
        dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
        labels = dbscan.fit_predict(X)
        
        logger.info(f"DBSCAN: eps={eps:.3f}, found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels
    
    def _auto_clustering(self, X: np.ndarray, suggested_k: int) -> np.ndarray:
        """Try multiple clustering methods and select the best one"""
        clustering_results = []
        
        # Try K-means with different cluster counts
        for k in range(2, min(suggested_k + 3, len(X))):
            try:
                labels = self._kmeans_clustering(X, k)
                metrics = self.evaluator.evaluate_clustering(X, labels)
                clustering_results.append((f"kmeans_k{k}", labels, metrics))
            except Exception as e:
                logger.warning(f"K-means with k={k} failed: {e}")
        
        # Try hierarchical clustering
        for k in range(2, min(suggested_k + 2, len(X))):
            try:
                labels = self._hierarchical_clustering(X, k)
                metrics = self.evaluator.evaluate_clustering(X, labels)
                clustering_results.append((f"hierarchical_k{k}", labels, metrics))
            except Exception as e:
                logger.warning(f"Hierarchical clustering with k={k} failed: {e}")
        
        # Try DBSCAN
        try:
            labels = self._dbscan_clustering(X)
            metrics = self.evaluator.evaluate_clustering(X, labels)
            clustering_results.append(("dbscan", labels, metrics))
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
        
        # Select best clustering
        best_method, best_labels = self.evaluator.select_best_clustering(clustering_results)
        
        if best_method == "none" or len(best_labels) == 0:
            logger.warning("All clustering methods failed, using simple k-means")
            return self._kmeans_clustering(X, min(3, len(X)))
        
        return best_labels
    
    def _estimate_eps(self, X: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN using k-distance graph"""
        try:
            k = min(4, len(X) - 1)  # Use 4-NN or less if dataset is small
            if k <= 0:
                return 0.5
                
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(X)
            distances, indices = neighbors_fit.kneighbors(X)
            
            # Sort k-distances and find the "elbow"
            k_distances = np.sort(distances[:, k-1])
            
            # Simple heuristic: use median of k-distances
            eps = np.median(k_distances)
            
            # Ensure reasonable bounds
            eps = max(0.1, min(eps, 2.0))
            
            return eps
        except Exception as e:
            logger.warning(f"Error estimating eps: {e}")
            return 0.5
    
    def _cluster_by_category(self, conversations: List[ConversationData]) -> List[ConversationCluster]:
        """Simple clustering based on content categories"""
        category_groups: Dict[str, List[ConversationData]] = {}
        
        for conv in conversations:
            category = determine_category(conv.get_text())
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(conv)
        
        clusters = []
        for cluster_id, (category, convs) in enumerate(category_groups.items()):
            cluster = ConversationCluster(
                id=cluster_id + 1,
                name=category,
                description=self._get_category_description(category),
                count=len(convs),
                conversations=convs,
                keywords=self._extract_keywords_from_category(category)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _create_clusters(self, conversations: List[ConversationData], labels: np.ndarray) -> List[ConversationCluster]:
        """Create cluster objects from conversation data and labels"""
        
        # Group conversations by cluster label
        cluster_groups: Dict[int, List[ConversationData]] = {}
        for conv, label in zip(conversations, labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(conv)
        
        clusters = []
        for cluster_id, convs in cluster_groups.items():
            # Generate cluster name and description
            cluster_name, cluster_desc = self._generate_cluster_info(convs)
            
            cluster = ConversationCluster(
                id=cluster_id + 1,
                name=cluster_name,
                description=cluster_desc,
                count=len(convs),
                conversations=convs,
                keywords=self._extract_cluster_keywords(convs),
                representative_messages=self._get_representative_messages(convs)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _generate_cluster_info(self, conversations: List[ConversationData]) -> tuple[str, str]:
        """Generate name and description for a cluster"""
        
        # Try LLM-based naming first if available
        if self.llm_provider and len(conversations) >= 2:
            try:
                return self._generate_llm_cluster_info(conversations)
            except Exception as e:
                logger.warning(f"LLM cluster naming failed: {e}")
        
        # Fallback to category-based naming
        return self._generate_category_cluster_info(conversations)
    
    def _generate_llm_cluster_info(self, conversations: List[ConversationData]) -> tuple[str, str]:
        """Generate cluster name and description using LLM"""
        
        # Collect sample texts from cluster
        sample_texts = []
        for conv in conversations[:5]:  # Use first 5 conversations as samples
            text = conv.get_text()[:200]  # First 200 chars
            sample_texts.append(text)
        
        # Create prompt for cluster naming
        prompt = f"""Analyze these {len(conversations)} similar conversations and provide:
1. A short, descriptive cluster name (2-4 words)
2. A brief description (one sentence)

Sample conversations from this cluster:
{chr(10).join(f"- {text}" for text in sample_texts)}

Respond in this format:
Name: [cluster name]
Description: [brief description]"""
        
        # Get LLM response
        response = self.llm_provider.complete(prompt)
        
        # Parse response
        lines = response.strip().split('\n')
        name = "Conversation Cluster"
        description = "A group of related conversations"
        
        for line in lines:
            if line.startswith('Name:'):
                name = line.replace('Name:', '').strip()
            elif line.startswith('Description:'):
                description = line.replace('Description:', '').strip()
        
        # Ensure reasonable length
        if len(name) > 50:
            name = name[:50]
        if len(description) > 200:
            description = description[:200]
            
        logger.debug(f"Generated cluster name: '{name}' - {description}")
        return name, description
    
    def _generate_category_cluster_info(self, conversations: List[ConversationData]) -> tuple[str, str]:
        """Generate cluster info using category analysis (fallback)"""
        
        # Analyze categories in this cluster
        categories = [determine_category(conv.get_text()) for conv in conversations]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Use most common category as cluster name
        if category_counts:
            most_common_category = max(category_counts, key=category_counts.get)
            
            # If cluster is diverse, use a more general name
            if len(category_counts) > 1 and category_counts[most_common_category] < len(conversations) * 0.6:
                cluster_name = f"Mixed - {most_common_category}"
            else:
                cluster_name = most_common_category
        else:
            cluster_name = f"Cluster {len(conversations)} conversations"
        
        # Generate description
        description = self._get_category_description(most_common_category if category_counts else "General")
        
        return cluster_name, description
    
    def _get_category_description(self, category: str) -> str:
        """Get description for a category"""
        descriptions = {
            'Bug Report': 'Technical issues and system problems reported by users',
            'Feature Request': 'User suggestions for new features and improvements', 
            'Support': 'Customer support and technical assistance requests',
            'Feedback': 'User feedback, testimonials and reviews',
            'Sales': 'Pricing inquiries and subscription questions',
            'Documentation': 'Documentation and API reference requests',
            'General': 'General conversation topics and discussions'
        }
        return descriptions.get(category, 'Various conversation topics')
    
    def _extract_keywords_from_category(self, category: str) -> List[str]:
        """Extract keywords associated with a category"""
        keywords = {
            'Bug Report': ['bug', 'error', 'crash', 'broken', 'fix', 'issue'],
            'Feature Request': ['feature', 'request', 'add', 'implement', 'enhancement'],
            'Support': ['help', 'support', 'how to', 'question', 'assistance'],
            'Feedback': ['feedback', 'review', 'opinion', 'great', 'love'],
            'Sales': ['price', 'cost', 'subscription', 'billing', 'purchase'],
            'Documentation': ['documentation', 'api', 'guide', 'tutorial', 'reference'],
            'General': ['conversation', 'discussion', 'chat']
        }
        return keywords.get(category, [])
    
    def _extract_cluster_keywords(self, conversations: List[ConversationData]) -> List[str]:
        """Extract keywords from cluster conversations"""
        # Simple keyword extraction - count most common words
        from collections import Counter
        import re
        
        all_text = " ".join(conv.get_text() for conv in conversations).lower()
        
        # Remove common stop words and extract words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'they', 'we', 'say', 'her', 'she', 'he', 'has', 'had'}
        
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        words = [w for w in words if w not in stop_words]
        
        # Get top 10 most common words
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(10)]
    
    def _get_representative_messages(self, conversations: List[ConversationData], max_messages: int = 3) -> List[str]:
        """Get representative messages from the cluster"""
        messages = []
        
        for conv in conversations[:max_messages]:
            # Get first user message if available
            user_messages = conv.get_user_messages()
            if user_messages:
                text = user_messages[0].content[:150]  # First 150 chars
                messages.append(text)
        
        return messages