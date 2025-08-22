use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    Router,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::clio_core::{
    ClioAnalysisPipeline, ClioAnalysisRequest, ClioCluster, ClioHierarchy,
    ClioPattern, ClioVisualizationData, PrivacyConfig, ClusteringConfig,
    SummarizationLevel, ClioConversation,
};
use crate::types::ConversationData;
use crate::config::{LlmProvider, EmbeddingProvider};

#[derive(Clone)]
pub struct ClioState {
    pub pipeline: Arc<ClioAnalysisPipeline>,
    pub current_analysis: Arc<RwLock<Option<ClioVisualizationData>>>,
    pub conversation_cache: Arc<RwLock<Vec<ConversationData>>>,
}

#[derive(Deserialize)]
pub struct ClioAnalysisParams {
    pub min_cluster_size: Option<usize>,
    pub max_hierarchy_depth: Option<usize>,
    pub enable_privacy: Option<bool>,
    pub pattern_threshold: Option<f32>,
}

#[derive(Serialize)]
pub struct ClioAnalysisResponse {
    pub status: String,
    pub message: String,
    pub analysis_id: Option<String>,
}

#[derive(Serialize)]
pub struct ClioHierarchyResponse {
    pub hierarchy: ClioHierarchy,
    pub clusters: Vec<ClioCluster>,
    pub conversations: usize,
}

#[derive(Serialize)]
pub struct ClioPatternsResponse {
    pub patterns: Vec<ClioPattern>,
    pub total_patterns: usize,
}

#[derive(Deserialize)]
pub struct ClioSearchQuery {
    pub query: String,
    pub cluster_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct ClioSearchResult {
    pub cluster_id: String,
    pub cluster_name: String,
    pub relevance: f32,
    pub size: usize,
    pub summary: String,
}

pub fn create_clio_routes(state: ClioState) -> Router {
    Router::new()
        .route("/api/clio/analyze", post(start_clio_analysis))
        .route("/api/clio/hierarchy", get(get_hierarchy))
        .route("/api/clio/cluster/:id", get(get_cluster_details))
        .route("/api/clio/patterns", get(get_patterns))
        .route("/api/clio/search", get(search_clusters))
        .route("/api/clio/export/:id", post(export_cluster))
        .route("/api/clio/investigate/:id", post(investigate_cluster))
        .route("/api/clio/status", get(get_analysis_status))
        .with_state(state)
}

async fn start_clio_analysis(
    State(state): State<ClioState>,
    Query(params): Query<ClioAnalysisParams>,
    Json(conversations): Json<Vec<ConversationData>>,
) -> Result<Json<ClioAnalysisResponse>, StatusCode> {
    // Store conversations in cache
    {
        let mut cache = state.conversation_cache.write().await;
        *cache = conversations.clone();
    }
    
    // Create analysis request
    let request = ClioAnalysisRequest {
        conversations,
        min_cluster_size: params.min_cluster_size.unwrap_or(10),
        max_hierarchy_depth: params.max_hierarchy_depth.unwrap_or(5),
        pattern_threshold: params.pattern_threshold.unwrap_or(0.3),
    };
    
    // Start analysis in background
    let pipeline = state.pipeline.clone();
    let current_analysis = state.current_analysis.clone();
    
    tokio::spawn(async move {
        match pipeline.analyze(request).await {
            Ok(result) => {
                let mut analysis = current_analysis.write().await;
                *analysis = Some(result);
            }
            Err(e) => {
                eprintln!("Clio analysis failed: {}", e);
            }
        }
    });
    
    Ok(Json(ClioAnalysisResponse {
        status: "started".to_string(),
        message: "Clio analysis started successfully".to_string(),
        analysis_id: Some(uuid::Uuid::new_v4().to_string()),
    }))
}

async fn get_hierarchy(
    State(state): State<ClioState>,
) -> Result<Json<ClioHierarchyResponse>, StatusCode> {
    let analysis = state.current_analysis.read().await;
    
    match &*analysis {
        Some(data) => {
            Ok(Json(ClioHierarchyResponse {
                hierarchy: data.hierarchy.clone(),
                clusters: data.clusters.clone(),
                conversations: data.total_conversations,
            }))
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_cluster_details(
    State(state): State<ClioState>,
    Path(cluster_id): Path<String>,
) -> Result<Json<ClioCluster>, StatusCode> {
    let analysis = state.current_analysis.read().await;
    
    match &*analysis {
        Some(data) => {
            // Find cluster by ID
            for cluster in &data.clusters {
                if cluster.id == cluster_id {
                    return Ok(Json(cluster.clone()));
                }
            }
            Err(StatusCode::NOT_FOUND)
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_patterns(
    State(state): State<ClioState>,
) -> Result<Json<ClioPatternsResponse>, StatusCode> {
    let analysis = state.current_analysis.read().await;
    
    match &*analysis {
        Some(data) => {
            Ok(Json(ClioPatternsResponse {
                total_patterns: data.patterns.len(),
                patterns: data.patterns.clone(),
            }))
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn search_clusters(
    State(state): State<ClioState>,
    Query(query): Query<ClioSearchQuery>,
) -> Result<Json<Vec<ClioSearchResult>>, StatusCode> {
    let analysis = state.current_analysis.read().await;
    
    match &*analysis {
        Some(data) => {
            let mut results = Vec::new();
            
            // Simple text search through cluster summaries
            for cluster in &data.clusters {
                let query_lower = query.query.to_lowercase();
                let summary_lower = cluster.summary.to_lowercase();
                let name_lower = cluster.name.to_lowercase();
                
                // Calculate relevance based on matches
                let mut relevance = 0.0;
                if name_lower.contains(&query_lower) {
                    relevance += 1.0;
                }
                if summary_lower.contains(&query_lower) {
                    relevance += 0.5;
                }
                
                if relevance > 0.0 {
                    results.push(ClioSearchResult {
                        cluster_id: cluster.id.clone(),
                        cluster_name: cluster.name.clone(),
                        relevance,
                        size: cluster.size,
                        summary: cluster.summary.clone(),
                    });
                }
            }
            
            // Sort by relevance
            results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
            
            // Apply limit
            if let Some(limit) = query.limit {
                results.truncate(limit);
            }
            
            Ok(Json(results))
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn export_cluster(
    State(state): State<ClioState>,
    Path(cluster_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let analysis = state.current_analysis.read().await;
    let conversations = state.conversation_cache.read().await;
    
    match &*analysis {
        Some(data) => {
            // Find cluster
            for cluster in &data.clusters {
                if cluster.id == cluster_id {
                    // Get conversations in this cluster
                    let cluster_conversations: Vec<&ConversationData> = conversations
                        .iter()
                        .enumerate()
                        .filter_map(|(i, conv)| {
                            if cluster.conversation_ids.contains(&i) {
                                Some(conv)
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    // Create export data
                    let export_data = serde_json::json!({
                        "cluster": cluster,
                        "conversations": cluster_conversations,
                        "metadata": {
                            "exported_at": chrono::Utc::now().to_rfc3339(),
                            "total_conversations": cluster_conversations.len(),
                            "cluster_name": cluster.name,
                        }
                    });
                    
                    return Ok(Json(export_data));
                }
            }
            Err(StatusCode::NOT_FOUND)
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn investigate_cluster(
    State(state): State<ClioState>,
    Path(cluster_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let analysis = state.current_analysis.read().await;
    let conversations = state.conversation_cache.read().await;
    
    match &*analysis {
        Some(data) => {
            // Find cluster
            for cluster in &data.clusters {
                if cluster.id == cluster_id {
                    // Get sample conversations
                    let sample_size = std::cmp::min(5, cluster.conversation_ids.len());
                    let sample_conversations: Vec<&ConversationData> = cluster.conversation_ids
                        .iter()
                        .take(sample_size)
                        .filter_map(|&id| conversations.get(id))
                        .collect();
                    
                    // Find related patterns
                    let related_patterns: Vec<&ClioPattern> = data.patterns
                        .iter()
                        .filter(|p| p.cluster_ids.contains(&cluster_id))
                        .collect();
                    
                    // Create investigation report
                    let investigation = serde_json::json!({
                        "cluster": cluster,
                        "sample_conversations": sample_conversations,
                        "related_patterns": related_patterns,
                        "statistics": {
                            "total_conversations": cluster.size,
                            "sentiment_distribution": cluster.metadata.get("sentiment_distribution"),
                            "top_keywords": cluster.metadata.get("top_keywords"),
                            "temporal_distribution": cluster.metadata.get("temporal_distribution"),
                        },
                        "recommendations": [
                            "Review sample conversations for common themes",
                            "Check related patterns for actionable insights",
                            "Consider creating automated responses for common issues",
                        ]
                    });
                    
                    return Ok(Json(investigation));
                }
            }
            Err(StatusCode::NOT_FOUND)
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_analysis_status(
    State(state): State<ClioState>,
) -> Json<serde_json::Value> {
    let analysis = state.current_analysis.read().await;
    
    let status = match &*analysis {
        Some(data) => {
            serde_json::json!({
                "status": "completed",
                "total_clusters": data.clusters.len(),
                "total_patterns": data.patterns.len(),
                "hierarchy_depth": data.hierarchy.depth,
                "total_conversations": data.total_conversations,
                "analysis_cost": data.analysis_cost,
            })
        }
        None => {
            serde_json::json!({
                "status": "not_started",
                "message": "No analysis has been performed yet",
            })
        }
    };
    
    Json(status)
}

// Helper function to create default Clio state would go here
// For now, we'll implement this when integrating with actual providers