use crate::types::FacetValue;
use crate::types_extended::AnalysisCluster;
use serde::{Serialize, Deserialize};

/// Discovery recommendation for serendipitous exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryRecommendation {
    pub cluster_id: usize,
    pub title: String,
    pub description: String,
    pub confidence_score: f64,
    pub related_facets: Vec<String>,
    pub recommendation_type: RecommendationType,
}

/// Types of discovery recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    SimilarPattern,
    UnexpectedConnection,
    OutlierAnalysis,
    TrendIdentification,
    CrossClusterCorrelation,
}

/// Discovery engine for serendipitous exploration
#[derive(Debug, Clone)]
pub struct DiscoveryEngine {
    clusters: Vec<AnalysisCluster>,
    facet_data: Vec<Vec<FacetValue>>,
}

impl DiscoveryEngine {
    pub fn new(
        clusters: Vec<AnalysisCluster>,
        facet_data: Vec<Vec<FacetValue>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            clusters,
            facet_data,
        })
    }

    pub fn get_clusters(&self) -> &[AnalysisCluster] {
        &self.clusters
    }

    /// Get discovery recommendations based on the current cluster
    pub fn get_recommendations(
        &self,
        current_cluster: Option<usize>,
        limit: usize,
    ) -> anyhow::Result<Vec<DiscoveryRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Generate recommendations based on current cluster or global analysis
        match current_cluster {
            Some(cluster_id) => {
                if cluster_id >= self.clusters.len() {
                    return Err(anyhow::anyhow!("Invalid cluster ID: {}", cluster_id));
                }
                
                // Generate recommendations from the specific cluster
                recommendations.extend(self.generate_cluster_recommendations(cluster_id, limit)?);
            }
            None => {
                // Generate global recommendations across all clusters
                recommendations.extend(self.generate_global_recommendations(limit)?);
            }
        }
        
        // Sort by confidence score and limit results
        recommendations.sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap());
        recommendations.truncate(limit);
        
        Ok(recommendations)
    }

    /// Update user preferences for discovery recommendations
    pub fn update_preferences(&mut self, preferences: UserPreferences) -> anyhow::Result<()> {
        // Stub implementation - in a real implementation, this would update the recommendation algorithm
        tracing::info!("Updated discovery preferences: {:?}", preferences);
        Ok(())
    }
    
    /// Generate recommendations from a specific cluster
    fn generate_cluster_recommendations(
        &self,
        cluster_id: usize,
        limit: usize,
    ) -> anyhow::Result<Vec<DiscoveryRecommendation>> {
        let mut recommendations = Vec::new();
        let cluster = &self.clusters[cluster_id];
        
        // Find similar patterns in other clusters
        for (i, other_cluster) in self.clusters.iter().enumerate() {
            if i == cluster_id {
                continue;
            }
            
            // Simple similarity based on member count and description similarity
            let similarity_score = self.calculate_cluster_similarity(cluster, other_cluster);
            
            if similarity_score > 0.3 && recommendations.len() < limit {
                recommendations.push(DiscoveryRecommendation {
                    cluster_id: i,
                    title: format!("Similar Pattern: {}", other_cluster.name),
                    description: format!("This cluster shows similar patterns to your current focus area"),
                    confidence_score: similarity_score,
                    related_facets: vec![], // TODO: Extract facets from cluster data
                    recommendation_type: RecommendationType::SimilarPattern,
                });
            }
        }
        
        // Add cross-cluster correlation recommendations
        if recommendations.len() < limit {
            recommendations.extend(self.generate_correlation_recommendations(cluster_id, limit - recommendations.len())?);
        }
        
        Ok(recommendations)
    }
    
    /// Generate global recommendations across all clusters
    fn generate_global_recommendations(&self, limit: usize) -> anyhow::Result<Vec<DiscoveryRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Find outlier clusters (small clusters)
        for (i, cluster) in self.clusters.iter().enumerate() {
            if cluster.conversation_ids.len() < 3 && recommendations.len() < limit / 2 {
                recommendations.push(DiscoveryRecommendation {
                    cluster_id: i,
                    title: format!("Outlier Analysis: {}", cluster.name),
                    description: "This small cluster might contain unique or unusual patterns worth investigating".to_string(),
                    confidence_score: 0.7,
                    related_facets: vec![], // TODO: Extract facets from cluster data
                    recommendation_type: RecommendationType::OutlierAnalysis,
                });
            }
        }
        
        // Find trending patterns (largest clusters)
        let mut cluster_sizes: Vec<(usize, usize)> = self.clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.conversation_ids.len()))
            .collect();
        cluster_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
        
        for (cluster_id, _) in cluster_sizes.into_iter().take(limit - recommendations.len()) {
            let cluster = &self.clusters[cluster_id];
            recommendations.push(DiscoveryRecommendation {
                cluster_id,
                title: format!("Trending Pattern: {}", cluster.name),
                description: "This is one of the most common patterns in your data".to_string(),
                confidence_score: 0.8,
                related_facets: vec![], // TODO: Extract facets from cluster data
                recommendation_type: RecommendationType::TrendIdentification,
            });
        }
        
        Ok(recommendations)
    }
    
    /// Generate cross-cluster correlation recommendations
    fn generate_correlation_recommendations(
        &self,
        cluster_id: usize,
        limit: usize,
    ) -> anyhow::Result<Vec<DiscoveryRecommendation>> {
        let mut recommendations = Vec::new();
        let cluster = &self.clusters[cluster_id];
        
        // Find clusters with unexpected connections
        for (i, other_cluster) in self.clusters.iter().enumerate() {
            if i == cluster_id || recommendations.len() >= limit {
                continue;
            }
            
            // Look for potential connections based on cluster names or descriptions
            let name_similarity = cluster.name.len().min(other_cluster.name.len()) as f64 / 
                                cluster.name.len().max(other_cluster.name.len()) as f64;
            
            if name_similarity > 0.3 {
                recommendations.push(DiscoveryRecommendation {
                    cluster_id: i,
                    title: format!("Unexpected Connection: {}", other_cluster.name),
                    description: "This cluster might have interesting connections to your current focus area".to_string(),
                    confidence_score: 0.6,
                    related_facets: vec![], // TODO: Extract facets from cluster data
                    recommendation_type: RecommendationType::UnexpectedConnection,
                });
            }
        }
        
        Ok(recommendations)
    }
    
    /// Calculate similarity between two clusters
    fn calculate_cluster_similarity(
        &self,
        cluster1: &AnalysisCluster,
        cluster2: &AnalysisCluster,
    ) -> f64 {
        // Simple similarity based on member count ratio and name similarity
        let size_ratio = (cluster1.conversation_ids.len().min(cluster2.conversation_ids.len()) as f64) / 
                        (cluster1.conversation_ids.len().max(cluster2.conversation_ids.len()) as f64);
        
        // Simple name-based similarity
        let name_similarity = if cluster1.name == cluster2.name {
            1.0
        } else {
            let common_words = cluster1.name.split_whitespace()
                .filter(|word| cluster2.name.contains(word))
                .count();
            let total_words = cluster1.name.split_whitespace().count().max(
                cluster2.name.split_whitespace().count()
            );
            if total_words > 0 {
                common_words as f64 / total_words as f64
            } else {
                0.0
            }
        };
        
        (size_ratio + name_similarity) / 2.0
    }
}

/// User preferences for discovery recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub preferred_recommendation_types: Vec<RecommendationType>,
    pub min_confidence_score: f64,
    pub max_recommendations: usize,
    pub preferred_facets: Vec<String>,
}