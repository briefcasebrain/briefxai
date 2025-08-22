use crate::types::{ConversationData, FacetValue};
use crate::types_extended::AnalysisCluster;
use serde::{Deserialize, Serialize};

/// Investigation query for targeted search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationQuery {
    pub search_terms: Vec<String>,
    pub facet_filters: Vec<FacetFilter>,
    pub metric_filters: Vec<MetricFilter>,
    pub sort_criterion: SortCriterion,
    pub limit: Option<usize>,
    pub similar_to_cluster: Option<usize>,
    pub sort_by: Option<String>,
    pub highlight_matches: Option<bool>,
}

/// Investigation result from search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationResult {
    pub clusters: Vec<AnalysisCluster>,
    pub total_matches: usize,
    pub query_time_ms: u64,
    pub relevant_facets: Vec<String>,
}

/// Sort criteria for investigation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortCriterion {
    Relevance,
    ClusterSize,
    Alphabetical,
    CreationTime,
    Size,
}

/// Filter for specific facet values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetFilter {
    pub facet_name: String,
    pub operator: FilterOperator,
    pub value: String,
    pub threshold: Option<f64>,
}

/// Filter for cluster metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricFilter {
    pub metric: ClusterMetric,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub value: f64,
}

/// Available cluster metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMetric {
    Size,
    Density,
    Coherence,
    Diversity,
    RefusalRate,
    SentimentScore,
    ConversationLength,
}

/// Comparison operators for numeric filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterOrEqual,
    LessOrEqual,
    Equals,
    Between,
}

/// Filter operators for facet values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Contains,
    Equals,
    StartsWith,
    EndsWith,
    Regex,
    GreaterThan,
    LessThan,
    Prevalence,
}

/// Investigation engine for targeted cluster search
#[derive(Debug, Clone)]
pub struct InvestigationEngine {
    clusters: Vec<AnalysisCluster>,
    #[allow(dead_code)]
    conversations: Vec<ConversationData>,
    #[allow(dead_code)]
    facet_data: Vec<Vec<FacetValue>>,
    #[allow(dead_code)]
    embeddings: Option<Vec<Vec<f32>>>,
}

impl InvestigationEngine {
    pub fn new(
        clusters: Vec<AnalysisCluster>,
        conversations: Vec<ConversationData>,
        facet_data: Vec<Vec<FacetValue>>,
        embeddings: Option<Vec<Vec<f32>>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            clusters,
            conversations,
            facet_data,
            embeddings,
        })
    }

    pub fn get_clusters(&self) -> &[AnalysisCluster] {
        &self.clusters
    }

    /// Run investigation query
    pub fn investigate(&self, query: &InvestigationQuery) -> anyhow::Result<InvestigationResult> {
        // Stub implementation - in a real implementation, this would perform the actual search
        tracing::info!(
            "Running investigation with {} search terms",
            query.search_terms.len()
        );

        Ok(InvestigationResult {
            clusters: self.clusters.clone(),
            total_matches: self.clusters.len(),
            query_time_ms: 10,
            relevant_facets: vec!["example_facet".to_string()],
        })
    }
}
