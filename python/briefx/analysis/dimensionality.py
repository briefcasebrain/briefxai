"""Dimensionality reduction for embeddings using UMAP"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction"""
    n_components: int = 2  # Target dimensionality
    n_neighbors: int = 15  # Local neighborhood size
    min_dist: float = 0.1  # Minimum distance between points
    metric: str = 'cosine'  # Distance metric
    random_state: Optional[int] = 42
    n_epochs: Optional[int] = None  # Auto-determine if None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'random_state': self.random_state,
            'n_epochs': self.n_epochs
        }

class UMAPReducer:
    """UMAP-based dimensionality reduction for conversation embeddings"""
    
    def __init__(self, config: Optional[UMAPConfig] = None):
        self.config = config or UMAPConfig()
        self.reducer = None
        self.is_fitted = False
        
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available. Install with: pip install umap-learn")
    
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit UMAP on embeddings and return reduced embeddings
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            
        Returns:
            Tuple of (reduced_embeddings, metadata)
        """
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available, returning original embeddings")
            return embeddings, {"method": "passthrough", "reason": "umap_not_available"}
        
        if embeddings.shape[0] < self.config.n_neighbors:
            logger.warning(f"Not enough samples ({embeddings.shape[0]}) for n_neighbors={self.config.n_neighbors}")
            # Adjust n_neighbors for small datasets
            adjusted_neighbors = max(2, embeddings.shape[0] - 1)
            config = UMAPConfig(
                n_components=self.config.n_components,
                n_neighbors=adjusted_neighbors,
                min_dist=self.config.min_dist,
                metric=self.config.metric,
                random_state=self.config.random_state,
                n_epochs=self.config.n_epochs
            )
        else:
            config = self.config
        
        try:
            # Initialize UMAP
            self.reducer = umap.UMAP(
                n_components=config.n_components,
                n_neighbors=config.n_neighbors,
                min_dist=config.min_dist,
                metric=config.metric,
                random_state=config.random_state,
                n_epochs=config.n_epochs,
                verbose=False
            )
            
            logger.info(f"Fitting UMAP with {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")
            
            # Fit and transform
            reduced_embeddings = self.reducer.fit_transform(embeddings)
            self.is_fitted = True
            
            logger.info(f"UMAP completed: {embeddings.shape[1]}D → {reduced_embeddings.shape[1]}D")
            
            metadata = {
                "method": "umap",
                "original_shape": embeddings.shape,
                "reduced_shape": reduced_embeddings.shape,
                "config": config.to_dict(),
                "trustworthiness": self._calculate_trustworthiness(embeddings, reduced_embeddings)
            }
            
            return reduced_embeddings, metadata
            
        except Exception as e:
            logger.error(f"UMAP failed: {e}")
            return embeddings, {"method": "passthrough", "error": str(e)}
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted UMAP"""
        if not self.is_fitted or self.reducer is None:
            raise ValueError("UMAP not fitted. Call fit_transform first.")
        
        if not UMAP_AVAILABLE:
            return embeddings
            
        return self.reducer.transform(embeddings)
    
    def _calculate_trustworthiness(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Calculate trustworthiness score of the embedding"""
        try:
            from sklearn.manifold import trustworthiness
            # Use smaller k for small datasets
            k = min(5, original.shape[0] - 1)
            if k <= 0:
                return 0.0
            return float(trustworthiness(original, reduced, n_neighbors=k))
        except ImportError:
            logger.warning("sklearn.manifold.trustworthiness not available")
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate trustworthiness: {e}")
            return 0.0

def reduce_embeddings_for_clustering(
    embeddings: List[List[float]], 
    target_dim: int = 10,
    config: Optional[UMAPConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce embedding dimensionality for more effective clustering
    
    Args:
        embeddings: List of embedding vectors
        target_dim: Target dimensionality for clustering (default: 10)
        config: UMAP configuration
        
    Returns:
        Tuple of (reduced_embeddings, metadata)
    """
    if not embeddings:
        return np.array([]), {"method": "empty_input"}
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    logger.info(f"Input embeddings shape: {embeddings_array.shape}")
    
    # If already small enough, don't reduce
    if embeddings_array.shape[1] <= target_dim:
        logger.info(f"Embeddings already {embeddings_array.shape[1]}D, no reduction needed")
        return embeddings_array, {"method": "no_reduction", "reason": "already_small"}
    
    # Configure UMAP for clustering (higher dimensional output)
    if config is None:
        config = UMAPConfig(
            n_components=target_dim,
            n_neighbors=min(15, embeddings_array.shape[0] - 1),
            min_dist=0.0,  # Preserve local structure for clustering
            metric='cosine'
        )
    
    # Perform reduction
    reducer = UMAPReducer(config)
    reduced_embeddings, metadata = reducer.fit_transform(embeddings_array)
    
    return reduced_embeddings, metadata

def reduce_embeddings_for_visualization(
    embeddings: List[List[float]], 
    config: Optional[UMAPConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce embedding dimensionality for 2D visualization
    
    Args:
        embeddings: List of embedding vectors
        config: UMAP configuration
        
    Returns:
        Tuple of (2D_embeddings, metadata)
    """
    if not embeddings:
        return np.array([]), {"method": "empty_input"}
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Configure UMAP for visualization (2D output)
    if config is None:
        config = UMAPConfig(
            n_components=2,
            n_neighbors=min(15, embeddings_array.shape[0] - 1),
            min_dist=0.1,  # Spread points for better visualization
            metric='cosine'
        )
    
    # Perform reduction
    reducer = UMAPReducer(config)
    reduced_embeddings, metadata = reducer.fit_transform(embeddings_array)
    
    return reduced_embeddings, metadata