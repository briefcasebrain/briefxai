use crate::types::ConversationData;
use crate::types_extended::AnalysisCluster;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub min_cluster_size: usize,
    pub merge_small_clusters: bool,
    pub min_facet_prevalence: f64,
    pub sensitive_facets: Vec<String>,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 10,
            merge_small_clusters: true,
            min_facet_prevalence: 0.05,
            sensitive_facets: vec![
                "personal_info".to_string(),
                "health_info".to_string(),
                "financial_info".to_string(),
            ],
        }
    }
}

/// Privacy report for analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyReport {
    pub total_clusters: usize,
    pub filtered_clusters: usize,
    pub small_clusters_merged: usize,
    pub sensitive_facets_filtered: Vec<String>,
    pub privacy_level: PrivacyLevel,
}

/// Privacy level indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Low,
    Medium,
    High,
    Maximum,
}

impl Default for PrivacyReport {
    fn default() -> Self {
        Self {
            total_clusters: 0,
            filtered_clusters: 0,
            small_clusters_merged: 0,
            sensitive_facets_filtered: vec![],
            privacy_level: PrivacyLevel::Medium,
        }
    }
}

/// Privacy protection through threshold-based filtering
#[derive(Debug, Clone)]
pub struct PrivacyFilter {
    config: PrivacyConfig,
    stats: PrivacyStats,
}

#[derive(Debug, Clone, Default)]
struct PrivacyStats {
    total_clusters: usize,
    filtered_clusters: usize,
    small_clusters_merged: usize,
    sensitive_facets_filtered: Vec<String>,
}

impl PrivacyFilter {
    pub fn new(config: PrivacyConfig) -> Self {
        Self {
            config,
            stats: PrivacyStats::default(),
        }
    }

    pub async fn filter_clusters(
        &mut self,
        clusters: Vec<AnalysisCluster>,
        _conversations: &[ConversationData],
    ) -> Result<Vec<AnalysisCluster>> {
        self.stats.total_clusters = clusters.len();

        let mut filtered = Vec::new();
        let mut small_clusters = Vec::new();

        for cluster in clusters {
            if cluster.conversation_ids.len() >= self.config.min_cluster_size {
                filtered.push(cluster);
            } else {
                small_clusters.push(cluster);
            }
        }

        // Merge small clusters if enabled
        if self.config.merge_small_clusters && !small_clusters.is_empty() {
            let mut merged_ids = Vec::new();
            let mut merged_descriptions = Vec::new();

            for cluster in small_clusters {
                merged_ids.extend(cluster.conversation_ids);
                merged_descriptions.push(cluster.description);
            }

            if merged_ids.len() >= self.config.min_cluster_size {
                filtered.push(AnalysisCluster {
                    name: "merged_small_clusters".to_string(),
                    conversation_ids: merged_ids,
                    description: format!(
                        "Merged small clusters: {}",
                        merged_descriptions.join(", ")
                    ),
                    children: vec![],
                });
                self.stats.small_clusters_merged = merged_descriptions.len();
            } else {
                self.stats.filtered_clusters = merged_descriptions.len();
            }
        } else {
            self.stats.filtered_clusters = small_clusters.len();
        }

        Ok(filtered)
    }

    /// Generate a privacy report for the filtering results
    pub fn generate_report(&self) -> PrivacyReport {
        let privacy_level =
            if self.stats.filtered_clusters > 0 || self.stats.small_clusters_merged > 0 {
                PrivacyLevel::High
            } else {
                PrivacyLevel::Medium
            };

        PrivacyReport {
            total_clusters: self.stats.total_clusters,
            filtered_clusters: self.stats.filtered_clusters,
            small_clusters_merged: self.stats.small_clusters_merged,
            sensitive_facets_filtered: self.stats.sensitive_facets_filtered.clone(),
            privacy_level,
        }
    }
}
