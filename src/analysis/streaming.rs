use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{info, debug};
use std::time::Instant;

use crate::types::{ConversationCluster, FacetValue};
use crate::persistence_v2::ResultType;

// ============================================================================
// Streaming Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingUpdate {
    pub update_type: UpdateType,
    #[serde(skip)]
    pub timestamp: Option<Instant>,
    pub batch_number: Option<i32>,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    FacetExtracted,
    EmbeddingGenerated,
    ClusterFormed,
    InsightDiscovered,
    ProgressUpdate,
    PartialResultReady,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialResults {
    pub facets: Vec<Vec<FacetValue>>,
    pub clusters: Vec<ConversationCluster>,
    pub insights: Vec<Insight>,
    pub embeddings_count: usize,
    pub completion_percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub insight_type: InsightType,
    pub title: String,
    pub description: String,
    pub confidence: f32,
    pub supporting_data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    TrendIdentified,
    AnomalyDetected,
    PatternDiscovered,
    CommonIssue,
    EmergingTopic,
}

// ============================================================================
// Result Aggregator
// ============================================================================

pub struct ResultAggregator {
    facets: Arc<RwLock<Vec<Vec<FacetValue>>>>,
    clusters: Arc<RwLock<Vec<ConversationCluster>>>,
    insights: Arc<RwLock<Vec<Insight>>>,
    embeddings_count: Arc<RwLock<usize>>,
    total_expected: usize,
}

impl ResultAggregator {
    pub fn new(total_expected: usize) -> Self {
        Self {
            facets: Arc::new(RwLock::new(Vec::new())),
            clusters: Arc::new(RwLock::new(Vec::new())),
            insights: Arc::new(RwLock::new(Vec::new())),
            embeddings_count: Arc::new(RwLock::new(0)),
            total_expected,
        }
    }
    
    pub async fn add_facets(&self, facets: Vec<Vec<FacetValue>>) {
        let mut all_facets = self.facets.write().await;
        all_facets.extend(facets);
    }
    
    pub async fn add_clusters(&self, clusters: Vec<ConversationCluster>) {
        let mut all_clusters = self.clusters.write().await;
        *all_clusters = clusters;  // Replace rather than extend for clusters
    }
    
    pub async fn add_insight(&self, insight: Insight) {
        let mut insights = self.insights.write().await;
        insights.push(insight);
    }
    
    pub async fn increment_embeddings(&self, count: usize) {
        let mut embeddings_count = self.embeddings_count.write().await;
        *embeddings_count += count;
    }
    
    pub async fn get_partial_results(&self) -> PartialResults {
        let facets = self.facets.read().await.clone();
        let clusters = self.clusters.read().await.clone();
        let insights = self.insights.read().await.clone();
        let embeddings_count = *self.embeddings_count.read().await;
        
        let completion_percentage = if self.total_expected > 0 {
            (facets.len() as f32 / self.total_expected as f32) * 100.0
        } else {
            0.0
        };
        
        PartialResults {
            facets,
            clusters,
            insights,
            embeddings_count,
            completion_percentage,
        }
    }
    
    pub async fn detect_early_insights(&self) -> Vec<Insight> {
        let facets = self.facets.read().await;
        let mut insights = Vec::new();
        
        // Detect common patterns
        if facets.len() >= 10 {
            insights.extend(self.detect_common_patterns(&facets).await);
        }
        
        // Detect trends
        if facets.len() >= 20 {
            insights.extend(self.detect_trends(&facets).await);
        }
        
        insights
    }
    
    async fn detect_common_patterns(&self, facets: &[Vec<FacetValue>]) -> Vec<Insight> {
        let mut insights = Vec::new();
        
        // Count facet value frequencies
        let mut value_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for facet_set in facets {
            for facet_value in facet_set {
                *value_counts.entry(facet_value.value.clone()).or_insert(0) += 1;
            }
        }
        
        // Find most common values
        let mut common_values: Vec<_> = value_counts.iter().collect();
        common_values.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        
        if let Some((value, count)) = common_values.first() {
            if **count > facets.len() / 3 {
                insights.push(Insight {
                    insight_type: InsightType::PatternDiscovered,
                    title: "Common Pattern Detected".to_string(),
                    description: format!("'{}' appears in {}% of conversations", 
                        value, (**count * 100) / facets.len()),
                    confidence: 0.8,
                    supporting_data: serde_json::json!({
                        "value": value,
                        "count": count,
                        "percentage": (**count * 100) / facets.len()
                    }),
                });
            }
        }
        
        insights
    }
    
    async fn detect_trends(&self, facets: &[Vec<FacetValue>]) -> Vec<Insight> {
        let mut insights = Vec::new();
        
        // Simple trend detection - check if certain values are increasing
        let window_size = 10;
        if facets.len() >= window_size * 2 {
            let first_half = &facets[..facets.len() / 2];
            let second_half = &facets[facets.len() / 2..];
            
            let mut first_counts = std::collections::HashMap::new();
            let mut second_counts = std::collections::HashMap::new();
            
            for facet_set in first_half {
                for facet_value in facet_set {
                    *first_counts.entry(facet_value.value.clone()).or_insert(0) += 1;
                }
            }
            
            for facet_set in second_half {
                for facet_value in facet_set {
                    *second_counts.entry(facet_value.value.clone()).or_insert(0) += 1;
                }
            }
            
            // Find values that increased significantly
            for (value, second_count) in second_counts {
                let first_count = first_counts.get(&value).unwrap_or(&0);
                
                if second_count > first_count * 2 && second_count > 5 {
                    insights.push(Insight {
                        insight_type: InsightType::TrendIdentified,
                        title: "Emerging Trend".to_string(),
                        description: format!("'{}' is appearing more frequently", value),
                        confidence: 0.7,
                        supporting_data: serde_json::json!({
                            "value": value,
                            "first_half_count": first_count,
                            "second_half_count": second_count,
                            "increase_factor": second_count as f32 / (*first_count as f32).max(1.0)
                        }),
                    });
                }
            }
        }
        
        insights
    }
}

// ============================================================================
// Streaming Analyzer
// ============================================================================

pub struct StreamingAnalyzer {
    result_sender: broadcast::Sender<StreamingUpdate>,
    aggregator: Arc<ResultAggregator>,
    persistence: Arc<crate::persistence_v2::EnhancedPersistenceLayer>,
}

impl StreamingAnalyzer {
    pub fn new(
        total_conversations: usize,
        persistence: Arc<crate::persistence_v2::EnhancedPersistenceLayer>,
    ) -> (Self, broadcast::Receiver<StreamingUpdate>) {
        let (sender, receiver) = broadcast::channel(1000);
        
        let analyzer = Self {
            result_sender: sender,
            aggregator: Arc::new(ResultAggregator::new(total_conversations)),
            persistence,
        };
        
        (analyzer, receiver)
    }
    
    pub async fn process_batch_streaming(
        &self,
        batch_number: i32,
        facets: Vec<Vec<FacetValue>>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        debug!("Processing batch {} with streaming", batch_number);
        
        // Add facets
        self.aggregator.add_facets(facets.clone()).await;
        self.send_update(UpdateType::FacetExtracted, Some(batch_number), serde_json::json!({
            "count": facets.len()
        })).await?;
        
        // Add embeddings count
        self.aggregator.increment_embeddings(embeddings.len()).await;
        self.send_update(UpdateType::EmbeddingGenerated, Some(batch_number), serde_json::json!({
            "count": embeddings.len()
        })).await?;
        
        // Check for early insights
        let insights = self.aggregator.detect_early_insights().await;
        for insight in insights {
            self.aggregator.add_insight(insight.clone()).await;
            self.send_update(UpdateType::InsightDiscovered, Some(batch_number), 
                serde_json::to_value(&insight)?).await?;
        }
        
        // Store partial results
        self.persistence.store_partial_result(
            "current_session",  // This would be the actual session ID
            Some(batch_number),
            ResultType::Facet,
            serde_json::to_value(&facets)?,
        ).await?;
        
        // Send partial result ready notification
        let partial = self.aggregator.get_partial_results().await;
        self.send_update(UpdateType::PartialResultReady, Some(batch_number), 
            serde_json::to_value(&partial)?).await?;
        
        Ok(())
    }
    
    pub async fn get_partial_results(&self) -> PartialResults {
        self.aggregator.get_partial_results().await
    }
    
    pub async fn get_early_insights(&self) -> Vec<Insight> {
        self.aggregator.insights.read().await.clone()
    }
    
    pub async fn finalize_clusters(&self, clusters: Vec<ConversationCluster>) -> Result<()> {
        self.aggregator.add_clusters(clusters.clone()).await;
        
        self.send_update(UpdateType::ClusterFormed, None, serde_json::json!({
            "cluster_count": clusters.len()
        })).await?;
        
        Ok(())
    }
    
    async fn send_update(
        &self,
        update_type: UpdateType,
        batch_number: Option<i32>,
        data: serde_json::Value,
    ) -> Result<()> {
        let update = StreamingUpdate {
            update_type,
            timestamp: Some(Instant::now()),
            batch_number,
            data,
        };
        
        // Ignore send errors (no receivers)
        let _ = self.result_sender.send(update);
        
        Ok(())
    }
}

// ============================================================================
// WebSocket Handler for Streaming
// ============================================================================

use axum::extract::ws::{WebSocket, Message};
use futures::{stream::StreamExt, SinkExt};

pub async fn handle_streaming_websocket(
    mut socket: WebSocket,
    mut receiver: broadcast::Receiver<StreamingUpdate>,
) {
    // Send updates to the client
    loop {
        tokio::select! {
            update = receiver.recv() => {
                match update {
                    Ok(update) => {
                        let json = serde_json::to_string(&update).unwrap_or_default();
                        if socket.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        // Channel closed
                        break;
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Handle client messages if needed
                        if text == "ping" {
                            if socket.send(Message::Text("pong".to_string())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    
    info!("WebSocket connection closed");
}

// ============================================================================
// Export Functions
// ============================================================================

pub async fn export_partial_results(
    results: &PartialResults,
    format: ExportFormat,
) -> Result<Vec<u8>> {
    match format {
        ExportFormat::Json => {
            Ok(serde_json::to_vec_pretty(results)?)
        }
        ExportFormat::Csv => {
            export_to_csv(results).await
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
}

async fn export_to_csv(results: &PartialResults) -> Result<Vec<u8>> {
    use std::io::Write;
    
    let mut buffer = Vec::new();
    writeln!(buffer, "Type,Data,Count")?;
    
    writeln!(buffer, "Facets,Extracted,{}", results.facets.len())?;
    writeln!(buffer, "Clusters,Formed,{}", results.clusters.len())?;
    writeln!(buffer, "Insights,Discovered,{}", results.insights.len())?;
    writeln!(buffer, "Embeddings,Generated,{}", results.embeddings_count)?;
    writeln!(buffer, "Completion,Percentage,{:.1}", results.completion_percentage)?;
    
    // Add insights details
    writeln!(buffer, "\nInsights:")?;
    for insight in &results.insights {
        writeln!(buffer, "\"{}\",\"{}\",{:.2}", 
            insight.title, 
            insight.description, 
            insight.confidence)?;
    }
    
    Ok(buffer)
}