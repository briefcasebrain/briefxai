use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, warn};

use crate::config::BriefXAIConfig;
use crate::discovery::serendipitous::DiscoveryEngine;
use crate::investigation::targeted_search::InvestigationEngine;
use crate::privacy::threshold_protection::{PrivacyConfig, PrivacyFilter, PrivacyReport};
use crate::types::{ConversationCluster, ConversationData, FacetValue};
use crate::types_extended::AnalysisCluster;
use crate::visualization::interactive_map::InteractiveMap;

/// Enhanced analysis results that include Clio features
#[derive(Debug, Clone)]
pub struct EnhancedAnalysisResults {
    /// Original analysis results
    pub clusters: Vec<AnalysisCluster>,
    pub facet_data: Vec<Vec<FacetValue>>,
    pub embeddings: Vec<Vec<f32>>,
    pub umap_coords: Vec<(f32, f32)>,

    /// Privacy-filtered results
    pub privacy_filtered_clusters: Vec<AnalysisCluster>,
    pub privacy_report: PrivacyReport,

    /// Interactive visualization components
    pub interactive_map: Option<InteractiveMap>,

    /// Investigation and discovery engines
    pub investigation_engine: Option<InvestigationEngine>,
    pub discovery_engine: Option<DiscoveryEngine>,

    /// Analysis metadata
    pub total_conversations: usize,
    pub visible_conversations: usize,
    pub privacy_level: String,
}

/// Configuration for Clio features
#[derive(Debug, Clone)]
pub struct ClioConfig {
    /// Enable privacy filtering
    pub enable_privacy_filtering: bool,
    pub privacy_config: PrivacyConfig,

    /// Enable interactive map
    pub enable_interactive_map: bool,

    /// Enable investigation features
    pub enable_investigation: bool,

    /// Enable discovery features
    pub enable_discovery: bool,

    /// Minimum dataset size for Clio features
    pub min_dataset_size: usize,
}

impl Default for ClioConfig {
    fn default() -> Self {
        Self {
            enable_privacy_filtering: true,
            privacy_config: PrivacyConfig::default(),
            enable_interactive_map: true,
            enable_investigation: true,
            enable_discovery: true,
            min_dataset_size: 10,
        }
    }
}

/// Enhanced analysis engine that integrates Clio features
pub struct EnhancedAnalysisEngine {
    #[allow(dead_code)]
    config: BriefXAIConfig,
    clio_config: ClioConfig,
}

impl EnhancedAnalysisEngine {
    pub fn new(config: BriefXAIConfig, clio_config: ClioConfig) -> Self {
        Self {
            config,
            clio_config,
        }
    }

    /// Run enhanced analysis with Clio features
    pub async fn analyze_with_clio_features(
        &self,
        conversations: Vec<ConversationData>,
        facet_data: Vec<Vec<FacetValue>>,
        embeddings: Vec<Vec<f32>>,
        umap_coords: Vec<(f32, f32)>,
        original_clusters: Vec<ConversationCluster>,
    ) -> Result<EnhancedAnalysisResults> {
        info!(
            "Starting enhanced analysis with Clio features for {} conversations",
            conversations.len()
        );

        // Convert original clusters to analysis clusters
        let clusters = self.convert_to_analysis_clusters(original_clusters, &conversations)?;

        // Check minimum dataset size
        if conversations.len() < self.clio_config.min_dataset_size {
            warn!(
                "Dataset size {} below minimum {} for Clio features",
                conversations.len(),
                self.clio_config.min_dataset_size
            );

            return Ok(EnhancedAnalysisResults {
                clusters: clusters.clone(),
                facet_data,
                embeddings,
                umap_coords,
                privacy_filtered_clusters: clusters.clone(),
                privacy_report: PrivacyReport {
                    total_clusters: clusters.len(),
                    filtered_clusters: 0,
                    small_clusters_merged: 0,
                    sensitive_facets_filtered: vec![],
                    privacy_level: crate::privacy::threshold_protection::PrivacyLevel::Low,
                },
                interactive_map: None,
                investigation_engine: None,
                discovery_engine: None,
                total_conversations: conversations.len(),
                visible_conversations: conversations.len(),
                privacy_level: "None (dataset too small)".to_string(),
            });
        }

        // Apply privacy filtering
        let (privacy_filtered_clusters, privacy_report) =
            if self.clio_config.enable_privacy_filtering {
                self.apply_privacy_filtering(&clusters, &conversations)
                    .await?
            } else {
                (
                    clusters.clone(),
                    PrivacyReport {
                        total_clusters: clusters.len(),
                        filtered_clusters: 0,
                        small_clusters_merged: 0,
                        sensitive_facets_filtered: vec![],
                        privacy_level: crate::privacy::threshold_protection::PrivacyLevel::Low,
                    },
                )
            };

        info!(
            "Privacy filtering: {} clusters -> {} clusters",
            clusters.len(),
            privacy_filtered_clusters.len()
        );

        // Create interactive map
        let interactive_map = if self.clio_config.enable_interactive_map {
            self.create_interactive_map(&privacy_filtered_clusters, &umap_coords, &facet_data)
                .await?
        } else {
            None
        };

        // Create investigation engine
        let investigation_engine = if self.clio_config.enable_investigation {
            self.create_investigation_engine(
                &privacy_filtered_clusters,
                &conversations,
                &facet_data,
                &embeddings,
            )
            .await?
        } else {
            None
        };

        // Create discovery engine
        let discovery_engine = if self.clio_config.enable_discovery {
            self.create_discovery_engine(&privacy_filtered_clusters, &facet_data)
                .await?
        } else {
            None
        };

        let visible_conversations = privacy_filtered_clusters
            .iter()
            .map(|c| c.conversation_ids.len())
            .sum();

        let privacy_level = self.assess_privacy_level(&privacy_report);

        info!(
            "Enhanced analysis complete: {} visible conversations, privacy level: {}",
            visible_conversations, privacy_level
        );

        Ok(EnhancedAnalysisResults {
            clusters,
            facet_data,
            embeddings,
            umap_coords,
            privacy_filtered_clusters,
            privacy_report,
            interactive_map,
            investigation_engine,
            discovery_engine,
            total_conversations: conversations.len(),
            visible_conversations,
            privacy_level,
        })
    }

    /// Convert ConversationCluster to AnalysisCluster
    fn convert_to_analysis_clusters(
        &self,
        original_clusters: Vec<ConversationCluster>,
        conversations: &[ConversationData],
    ) -> Result<Vec<AnalysisCluster>> {
        let mut analysis_clusters = Vec::new();

        for (idx, cluster) in original_clusters.iter().enumerate() {
            // Extract conversation IDs - this is a simplification
            // In practice, you'd need to map from the original cluster structure
            let conversation_ids = if let Some(indices) = &cluster.indices {
                indices.clone()
            } else {
                // Fallback: assign conversations based on cluster index
                let chunk_size = conversations.len() / original_clusters.len().max(1);
                let start = idx * chunk_size;
                let end = ((idx + 1) * chunk_size).min(conversations.len());
                (start..end).collect()
            };

            let analysis_cluster = AnalysisCluster {
                conversation_ids,
                name: cluster.name.clone(),
                description: cluster.summary.clone(),
                children: cluster
                    .children
                    .as_ref()
                    .map(|children| {
                        self.convert_to_analysis_clusters(children.clone(), conversations)
                            .unwrap_or_default()
                    })
                    .unwrap_or_default(),
            };

            analysis_clusters.push(analysis_cluster);
        }

        Ok(analysis_clusters)
    }

    /// Apply privacy filtering to clusters
    async fn apply_privacy_filtering(
        &self,
        clusters: &[AnalysisCluster],
        conversations: &[ConversationData],
    ) -> Result<(Vec<AnalysisCluster>, PrivacyReport)> {
        info!("Applying privacy filtering to {} clusters", clusters.len());

        let mut privacy_filter = PrivacyFilter::new(self.clio_config.privacy_config.clone());

        let filtered_clusters = privacy_filter
            .filter_clusters(clusters.to_vec(), conversations)
            .await?;

        let report = privacy_filter.generate_report();

        Ok((filtered_clusters, report))
    }

    /// Create interactive map for visualization
    async fn create_interactive_map(
        &self,
        clusters: &[AnalysisCluster],
        umap_coords: &[(f32, f32)],
        facet_data: &[Vec<FacetValue>],
    ) -> Result<Option<InteractiveMap>> {
        info!("Creating interactive map for {} clusters", clusters.len());

        match InteractiveMap::new(clusters.to_vec(), umap_coords.to_vec(), facet_data) {
            Ok(map) => Ok(Some(map)),
            Err(e) => {
                warn!("Failed to create interactive map: {}", e);
                Ok(None)
            }
        }
    }

    /// Create investigation engine
    async fn create_investigation_engine(
        &self,
        clusters: &[AnalysisCluster],
        conversations: &[ConversationData],
        facet_data: &[Vec<FacetValue>],
        embeddings: &[Vec<f32>],
    ) -> Result<Option<InvestigationEngine>> {
        info!(
            "Creating investigation engine for {} clusters",
            clusters.len()
        );

        match InvestigationEngine::new(
            clusters.to_vec(),
            conversations.to_vec(),
            facet_data.to_vec(),
            Some(embeddings.to_vec()),
        ) {
            Ok(engine) => Ok(Some(engine)),
            Err(e) => {
                warn!("Failed to create investigation engine: {}", e);
                Ok(None)
            }
        }
    }

    /// Create discovery engine
    async fn create_discovery_engine(
        &self,
        clusters: &[AnalysisCluster],
        facet_data: &[Vec<FacetValue>],
    ) -> Result<Option<DiscoveryEngine>> {
        info!("Creating discovery engine for {} clusters", clusters.len());

        match DiscoveryEngine::new(clusters.to_vec(), facet_data.to_vec()) {
            Ok(engine) => Ok(Some(engine)),
            Err(e) => {
                warn!("Failed to create discovery engine: {}", e);
                Ok(None)
            }
        }
    }

    /// Assess privacy level based on configuration and filtering results
    fn assess_privacy_level(&self, report: &PrivacyReport) -> String {
        let config = &self.clio_config.privacy_config;

        let level = match config.min_cluster_size {
            0..=5 => "Low",
            6..=15 => "Medium",
            16..=50 => "High",
            _ => "Maximum",
        };

        if report.small_clusters_merged > 0 || report.filtered_clusters > 0 {
            format!(
                "{} (filtered {} clusters)",
                level,
                report.small_clusters_merged + report.filtered_clusters
            )
        } else {
            level.to_string()
        }
    }

    /// Get configuration recommendations based on dataset characteristics
    pub fn recommend_clio_config(
        &self,
        conversation_count: usize,
        cluster_count: usize,
        has_sensitive_data: bool,
    ) -> ClioConfig {
        let mut config = ClioConfig::default();

        // Adjust privacy settings based on dataset size
        if conversation_count < 100 {
            config.privacy_config.min_cluster_size = 5;
            config.privacy_config.merge_small_clusters = true;
        } else if conversation_count < 1000 {
            config.privacy_config.min_cluster_size = 10;
        } else {
            config.privacy_config.min_cluster_size = 20;
        }

        // Adjust for sensitive data
        if has_sensitive_data {
            config.privacy_config.min_cluster_size *= 2;
            config.privacy_config.min_facet_prevalence = 0.1; // Higher threshold
                                                              // Note: Removed sensitive_facet_threshold as it doesn't exist in PrivacyConfig
        }

        // Disable features for very small datasets
        if conversation_count < 20 {
            config.enable_discovery = false;
            config.enable_investigation = false;
        }

        if cluster_count < 3 {
            config.enable_interactive_map = false;
        }

        info!("Recommended Clio config for {} conversations, {} clusters: privacy_level={}, features_enabled={}",
              conversation_count, cluster_count,
              config.privacy_config.min_cluster_size,
              format!("map:{}, inv:{}, disc:{}",
                     config.enable_interactive_map,
                     config.enable_investigation,
                     config.enable_discovery));

        config
    }
}

/// Helper functions for integration with existing analysis pipeline
pub mod integration_helpers {
    use super::*;

    /// Extract Clio configuration from main BriefXAI config
    pub fn extract_clio_config(_config: &BriefXAIConfig) -> ClioConfig {
        ClioConfig {
            enable_privacy_filtering: true,
            privacy_config: PrivacyConfig {
                min_cluster_size: 10, // Could be configurable
                merge_small_clusters: true,
                ..Default::default()
            },
            enable_interactive_map: true,
            enable_investigation: true,
            enable_discovery: true,
            min_dataset_size: 10,
        }
    }

    /// Check if dataset is suitable for Clio features
    pub fn validate_dataset_for_clio(
        conversations: &[ConversationData],
        clusters: &[AnalysisCluster],
    ) -> Result<bool> {
        if conversations.is_empty() {
            return Ok(false);
        }

        if clusters.is_empty() {
            return Ok(false);
        }

        // Check for minimum diversity
        let avg_cluster_size = conversations.len() / clusters.len();
        if avg_cluster_size < 2 {
            warn!("Average cluster size too small for meaningful analysis");
            return Ok(false);
        }

        Ok(true)
    }

    /// Generate analysis summary for UI
    pub fn generate_analysis_summary(
        results: &EnhancedAnalysisResults,
    ) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();

        summary.insert(
            "total_conversations".to_string(),
            serde_json::Value::Number(results.total_conversations.into()),
        );
        summary.insert(
            "visible_conversations".to_string(),
            serde_json::Value::Number(results.visible_conversations.into()),
        );
        summary.insert(
            "total_clusters".to_string(),
            serde_json::Value::Number(results.clusters.len().into()),
        );
        summary.insert(
            "filtered_clusters".to_string(),
            serde_json::Value::Number(results.privacy_filtered_clusters.len().into()),
        );
        summary.insert(
            "privacy_level".to_string(),
            serde_json::Value::String(results.privacy_level.clone()),
        );
        summary.insert(
            "has_interactive_map".to_string(),
            serde_json::Value::Bool(results.interactive_map.is_some()),
        );
        summary.insert(
            "has_investigation".to_string(),
            serde_json::Value::Bool(results.investigation_engine.is_some()),
        );
        summary.insert(
            "has_discovery".to_string(),
            serde_json::Value::Bool(results.discovery_engine.is_some()),
        );

        // Privacy metrics
        summary.insert(
            "privacy_merged_clusters".to_string(),
            serde_json::Value::Number(results.privacy_report.small_clusters_merged.into()),
        );
        summary.insert(
            "privacy_redacted_clusters".to_string(),
            serde_json::Value::Number(results.privacy_report.filtered_clusters.into()),
        );

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Facet, Message};

    fn create_test_data() -> (
        Vec<ConversationData>,
        Vec<Vec<FacetValue>>,
        Vec<ConversationCluster>,
    ) {
        let conversations = vec![
            ConversationData {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: "Help with billing".to_string(),
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: "I can help with billing".to_string(),
                    },
                ],
                metadata: Default::default(),
            },
            ConversationData {
                messages: vec![
                    Message {
                        role: "user".to_string(),
                        content: "Technical issue".to_string(),
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: "Let me assist".to_string(),
                    },
                ],
                metadata: Default::default(),
            },
        ];

        let facet_data = vec![
            vec![FacetValue {
                facet: Facet {
                    name: "category".to_string(),
                    question: "Category?".to_string(),
                    prefill: String::new(),
                    summary_criteria: None,
                    numeric: None,
                },
                value: "billing".to_string(),
            }],
            vec![FacetValue {
                facet: Facet {
                    name: "category".to_string(),
                    question: "Category?".to_string(),
                    prefill: String::new(),
                    summary_criteria: None,
                    numeric: None,
                },
                value: "technical".to_string(),
            }],
        ];

        let clusters = vec![ConversationCluster {
            facet: Facet {
                name: "cluster_1".to_string(),
                question: "Test".to_string(),
                prefill: String::new(),
                summary_criteria: None,
                numeric: None,
            },
            summary: "Billing support cluster".to_string(),
            name: "Billing Support".to_string(),
            children: None,
            parent: None,
            indices: Some(vec![0]),
        }];

        (conversations, facet_data, clusters)
    }

    #[tokio::test]
    async fn test_enhanced_analysis_integration() {
        let config = BriefXAIConfig::default();
        let clio_config = ClioConfig::default();
        let engine = EnhancedAnalysisEngine::new(config, clio_config);

        let (conversations, facet_data, clusters) = create_test_data();
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let umap_coords = vec![(0.1, 0.2), (0.3, 0.4)];

        let results = engine
            .analyze_with_clio_features(
                conversations,
                facet_data,
                embeddings,
                umap_coords,
                clusters,
            )
            .await
            .unwrap();

        assert!(results.total_conversations > 0);
        assert!(!results.clusters.is_empty());
    }

    #[test]
    fn test_clio_config_recommendations() {
        let config = BriefXAIConfig::default();
        let clio_config = ClioConfig::default();
        let engine = EnhancedAnalysisEngine::new(config, clio_config);

        // Small dataset
        let small_config = engine.recommend_clio_config(50, 3, false);
        assert_eq!(small_config.privacy_config.min_cluster_size, 5);

        // Large dataset with sensitive data
        let large_config = engine.recommend_clio_config(10000, 50, true);
        assert_eq!(large_config.privacy_config.min_cluster_size, 40); // 20 * 2 for sensitive

        // Tiny dataset
        let tiny_config = engine.recommend_clio_config(5, 1, false);
        assert!(!tiny_config.enable_discovery);
        assert!(!tiny_config.enable_investigation);
    }
}
