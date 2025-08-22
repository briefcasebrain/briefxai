use anyhow::Result;
use axum::{
    extract::{Json, Path},
    http::StatusCode,
    response::Json as AxumJson,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use crate::discovery::serendipitous::{
    DiscoveryEngine, DiscoveryRecommendation, RecommendationType, UserPreferences,
};
use crate::investigation::targeted_search::{
    ClusterMetric, ComparisonOperator, FacetFilter, FilterOperator, InvestigationEngine,
    InvestigationQuery, InvestigationResult, MetricFilter, SortCriterion,
};
use crate::privacy::threshold_protection::{PrivacyConfig, PrivacyFilter, PrivacyReport};
use crate::types::FacetValue;
use crate::types_extended::AnalysisCluster;
use crate::visualization::interactive_map::{
    AggregationType, ColorScheme, FacetOverlay, InteractiveMap,
};

/// Response wrapper for API calls
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

/// Request for creating interactive map
#[derive(Debug, Deserialize)]
pub struct CreateMapRequest {
    pub clusters: Vec<AnalysisCluster>,
    pub umap_coords: Vec<(f32, f32)>,
    pub facet_data: Vec<Vec<FacetValue>>,
}

/// Request for applying facet overlay
#[derive(Debug, Deserialize)]
pub struct ApplyOverlayRequest {
    pub facet_name: String,
    pub color_scheme: String, // "sequential", "diverging", "categorical", "heatmap"
    pub aggregation: String,  // "mean", "median", "sum", "count", "percentage", "prevalence"
    pub threshold: Option<f32>,
}

/// Request for privacy filtering
#[derive(Debug, Deserialize)]
pub struct PrivacyFilterRequest {
    pub clusters: Vec<AnalysisCluster>,
    pub conversations: Vec<crate::types::ConversationData>,
    pub config: Option<PrivacyConfig>,
}

/// Request for targeted investigation
#[derive(Debug, Deserialize)]
pub struct InvestigationRequest {
    pub clusters: Vec<AnalysisCluster>,
    pub conversations: Vec<crate::types::ConversationData>,
    pub facet_data: Vec<Vec<FacetValue>>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub query: InvestigationQueryRequest,
}

#[derive(Debug, Deserialize)]
pub struct InvestigationQueryRequest {
    pub search_terms: Vec<String>,
    pub facet_filters: Vec<FacetFilterRequest>,
    pub metric_filters: Vec<MetricFilterRequest>,
    pub similar_to_cluster: Option<usize>,
    pub sort_by: String, // "relevance", "size", "alphabetical", etc.
    pub limit: Option<usize>,
    pub highlight_matches: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct FacetFilterRequest {
    pub facet_name: String,
    pub operator: String,
    pub value: String,
    pub threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct MetricFilterRequest {
    pub metric: String,   // "size", "refusal_rate", "sentiment_score", etc.
    pub operator: String, // "greater_than", "less_than", "equals", "between"
    pub value: f32,
    pub max_value: Option<f32>, // For "between" operator
}

/// Request for discovery recommendations
#[derive(Debug, Deserialize)]
pub struct DiscoveryRequest {
    pub clusters: Vec<AnalysisCluster>,
    pub facet_data: Vec<Vec<FacetValue>>,
    pub current_cluster: Option<usize>,
    pub limit: Option<usize>,
}

/// Query parameters for getting recommendations
#[derive(Debug, Deserialize)]
pub struct RecommendationQuery {
    pub current_cluster: Option<usize>,
    pub limit: Option<usize>,
}

/// Add Clio feature routes to the main router
pub fn create_clio_routes() -> Router {
    Router::new()
        // Interactive Map routes
        .route("/api/clio/map/create", post(create_interactive_map))
        .route("/api/clio/map/overlay", post(apply_facet_overlay))
        .route("/api/clio/map/export", get(export_map_data))
        // Privacy routes
        .route("/api/clio/privacy/filter", post(apply_privacy_filter))
        .route("/api/clio/privacy/config", get(get_privacy_config))
        // Investigation routes
        .route("/api/clio/investigate", post(run_investigation))
        .route(
            "/api/clio/investigate/suggest",
            get(get_investigation_suggestions),
        )
        // Discovery routes
        .route(
            "/api/clio/discovery/recommendations",
            post(get_discovery_recommendations),
        )
        .route(
            "/api/clio/discovery/update_preferences",
            post(update_discovery_preferences),
        )
        .route("/api/clio/discovery/patterns", get(get_discovery_patterns))
}

// Global state for storing map instances (in production, use proper state management)
lazy_static::lazy_static! {
    static ref MAP_INSTANCES: std::sync::Mutex<HashMap<String, InteractiveMap>> =
        std::sync::Mutex::new(HashMap::new());
    static ref DISCOVERY_ENGINES: std::sync::Mutex<HashMap<String, DiscoveryEngine>> =
        std::sync::Mutex::new(HashMap::new());
}

/// Create a new interactive map
async fn create_interactive_map(
    Json(request): Json<CreateMapRequest>,
) -> Result<AxumJson<ApiResponse<String>>, StatusCode> {
    info!(
        "Creating interactive map with {} clusters",
        request.clusters.len()
    );

    match InteractiveMap::new(request.clusters, request.umap_coords, &request.facet_data) {
        Ok(map) => {
            let map_id = uuid::Uuid::new_v4().to_string();

            if let Ok(mut instances) = MAP_INSTANCES.lock() {
                instances.insert(map_id.clone(), map);
            }

            Ok(AxumJson(ApiResponse::success(map_id)))
        }
        Err(e) => Ok(AxumJson(ApiResponse::error(format!(
            "Failed to create map: {}",
            e
        )))),
    }
}

/// Apply facet overlay to existing map
async fn apply_facet_overlay(
    Path(map_id): Path<String>,
    Json(request): Json<ApplyOverlayRequest>,
) -> Result<AxumJson<ApiResponse<String>>, StatusCode> {
    let color_scheme = match request.color_scheme.as_str() {
        "sequential" => ColorScheme::Sequential,
        "diverging" => ColorScheme::Diverging,
        "categorical" => ColorScheme::Categorical,
        "heatmap" => ColorScheme::Heatmap,
        _ => {
            return Ok(AxumJson(ApiResponse::error(
                "Invalid color scheme".to_string(),
            )))
        }
    };

    let aggregation = match request.aggregation.as_str() {
        "mean" => AggregationType::Mean,
        "median" => AggregationType::Median,
        "sum" => AggregationType::Sum,
        "count" => AggregationType::Count,
        "percentage" => AggregationType::Percentage,
        "prevalence" => AggregationType::Prevalence,
        _ => {
            return Ok(AxumJson(ApiResponse::error(
                "Invalid aggregation type".to_string(),
            )))
        }
    };

    if let Ok(mut instances) = MAP_INSTANCES.lock() {
        if let Some(map) = instances.get_mut(&map_id) {
            let overlay = FacetOverlay {
                facet_name: request.facet_name,
                enabled: true,
                opacity: 0.8,
                color_scheme,
                aggregation,
                threshold: request.threshold.unwrap_or(0.0) as f64,
            };

            match map.apply_facet_overlay(overlay) {
                Ok(_) => Ok(AxumJson(ApiResponse::success(
                    "Overlay applied".to_string(),
                ))),
                Err(e) => Ok(AxumJson(ApiResponse::error(format!(
                    "Failed to apply overlay: {}",
                    e
                )))),
            }
        } else {
            Ok(AxumJson(ApiResponse::error("Map not found".to_string())))
        }
    } else {
        Ok(AxumJson(ApiResponse::error(
            "Failed to access map instances".to_string(),
        )))
    }
}

/// Export map data for frontend
async fn export_map_data(
    Path(map_id): Path<String>,
) -> Result<AxumJson<ApiResponse<serde_json::Value>>, StatusCode> {
    if let Ok(instances) = MAP_INSTANCES.lock() {
        if let Some(map) = instances.get(&map_id) {
            let data = map.export_for_frontend();
            match serde_json::to_value(data) {
                Ok(value) => Ok(AxumJson(ApiResponse::success(value))),
                Err(e) => Ok(AxumJson(ApiResponse::error(format!(
                    "Serialization error: {}",
                    e
                )))),
            }
        } else {
            Ok(AxumJson(ApiResponse::error("Map not found".to_string())))
        }
    } else {
        Ok(AxumJson(ApiResponse::error(
            "Failed to access map instances".to_string(),
        )))
    }
}

/// Apply privacy filtering to clusters
async fn apply_privacy_filter(
    Json(request): Json<PrivacyFilterRequest>,
) -> Result<AxumJson<ApiResponse<(Vec<AnalysisCluster>, PrivacyReport)>>, StatusCode> {
    let config = request.config.unwrap_or_default();
    let mut filter = PrivacyFilter::new(config);

    match filter
        .filter_clusters(request.clusters, &request.conversations)
        .await
    {
        Ok(filtered_clusters) => {
            let report = filter.generate_report();
            Ok(AxumJson(ApiResponse::success((filtered_clusters, report))))
        }
        Err(e) => Ok(AxumJson(ApiResponse::error(format!(
            "Privacy filtering failed: {}",
            e
        )))),
    }
}

/// Get default privacy configuration
async fn get_privacy_config() -> AxumJson<ApiResponse<PrivacyConfig>> {
    AxumJson(ApiResponse::success(PrivacyConfig::default()))
}

/// Run targeted investigation
async fn run_investigation(
    Json(request): Json<InvestigationRequest>,
) -> Result<AxumJson<ApiResponse<InvestigationResult>>, StatusCode> {
    let engine = match InvestigationEngine::new(
        request.clusters,
        request.conversations,
        request.facet_data,
        request.embeddings,
    ) {
        Ok(engine) => engine,
        Err(e) => {
            return Ok(AxumJson(ApiResponse::error(format!(
                "Failed to create investigation engine: {}",
                e
            ))))
        }
    };

    // Convert request query to internal format
    let sort_by = match request.query.sort_by.as_str() {
        "relevance" => SortCriterion::Relevance,
        "size" => SortCriterion::Size,
        "alphabetical" => SortCriterion::Alphabetical,
        _ => SortCriterion::Relevance,
    };

    let facet_filters: Result<Vec<FacetFilter>, String> = request
        .query
        .facet_filters
        .into_iter()
        .map(|f| {
            let operator = match f.operator.as_str() {
                "equals" => FilterOperator::Equals,
                "contains" => FilterOperator::Contains,
                "starts_with" => FilterOperator::StartsWith,
                "greater_than" => FilterOperator::GreaterThan,
                "less_than" => FilterOperator::LessThan,
                "prevalence" => FilterOperator::Prevalence,
                _ => return Err("Invalid filter operator".to_string()),
            };

            Ok(FacetFilter {
                facet_name: f.facet_name,
                operator,
                value: f.value,
                threshold: f.threshold.map(|x| x as f64),
            })
        })
        .collect();

    let facet_filters = match facet_filters {
        Ok(filters) => filters,
        Err(e) => return Ok(AxumJson(ApiResponse::error(e))),
    };

    let metric_filters: Result<Vec<MetricFilter>, String> = request
        .query
        .metric_filters
        .into_iter()
        .map(|m| {
            let metric = match m.metric.as_str() {
                "size" => ClusterMetric::Size,
                "refusal_rate" => ClusterMetric::RefusalRate,
                "sentiment_score" => ClusterMetric::SentimentScore,
                "conversation_length" => ClusterMetric::ConversationLength,
                "diversity" => ClusterMetric::Diversity,
                "coherence" => ClusterMetric::Coherence,
                _ => return Err("Invalid metric".to_string()),
            };

            let operator = match m.operator.as_str() {
                "greater_than" => ComparisonOperator::GreaterThan,
                "less_than" => ComparisonOperator::LessThan,
                "equals" => ComparisonOperator::Equals,
                "between" => {
                    if m.max_value.is_none() {
                        return Err("Between operator requires max_value".to_string());
                    }
                    ComparisonOperator::Between
                }
                _ => return Err("Invalid comparison operator".to_string()),
            };

            Ok(MetricFilter {
                metric,
                operator,
                threshold: m.value as f64,
                value: m.max_value.unwrap_or(m.value) as f64,
            })
        })
        .collect();

    let metric_filters = match metric_filters {
        Ok(filters) => filters,
        Err(e) => return Ok(AxumJson(ApiResponse::error(e))),
    };

    let query = InvestigationQuery {
        search_terms: request.query.search_terms,
        facet_filters,
        metric_filters,
        sort_criterion: sort_by,
        similar_to_cluster: request.query.similar_to_cluster,
        sort_by: Some(request.query.sort_by),
        limit: request.query.limit,
        highlight_matches: Some(request.query.highlight_matches.unwrap_or(false)),
    };

    match engine.investigate(&query) {
        Ok(results) => Ok(AxumJson(ApiResponse::success(results))),
        Err(e) => Ok(AxumJson(ApiResponse::error(format!(
            "Investigation failed: {}",
            e
        )))),
    }
}

/// Get investigation suggestions
async fn get_investigation_suggestions() -> AxumJson<ApiResponse<Vec<String>>> {
    let suggestions = vec![
        "Show clusters with high refusal rates".to_string(),
        "Find clusters with positive sentiment".to_string(),
        "Search for technical support topics".to_string(),
        "Identify large conversation clusters".to_string(),
        "Find clusters with diverse topics".to_string(),
    ];

    AxumJson(ApiResponse::success(suggestions))
}

/// Get discovery recommendations
async fn get_discovery_recommendations(
    Json(request): Json<DiscoveryRequest>,
) -> Result<AxumJson<ApiResponse<Vec<DiscoveryRecommendation>>>, StatusCode> {
    let engine_id = uuid::Uuid::new_v4().to_string();

    let engine = match DiscoveryEngine::new(request.clusters, request.facet_data) {
        Ok(engine) => engine,
        Err(e) => {
            return Ok(AxumJson(ApiResponse::error(format!(
                "Failed to create discovery engine: {}",
                e
            ))))
        }
    };

    let limit = request.limit.unwrap_or(5);

    match engine.get_recommendations(request.current_cluster, limit) {
        Ok(recommendations) => {
            // Store engine for future requests
            if let Ok(mut engines) = DISCOVERY_ENGINES.lock() {
                engines.insert(engine_id, engine);
            }

            Ok(AxumJson(ApiResponse::success(recommendations)))
        }
        Err(e) => Ok(AxumJson(ApiResponse::error(format!(
            "Failed to get recommendations: {}",
            e
        )))),
    }
}

/// Update discovery preferences
async fn update_discovery_preferences(
    Path(engine_id): Path<String>,
    Json(_visited_sequence): Json<Vec<usize>>,
) -> Result<AxumJson<ApiResponse<String>>, StatusCode> {
    if let Ok(mut engines) = DISCOVERY_ENGINES.lock() {
        if let Some(engine) = engines.get_mut(&engine_id) {
            // Create preferences based on visited sequence
            let preferences = UserPreferences {
                preferred_recommendation_types: vec![
                    RecommendationType::SimilarPattern,
                    RecommendationType::UnexpectedConnection,
                ],
                min_confidence_score: 0.5,
                max_recommendations: 5,
                preferred_facets: vec![], // Could be extracted from visited clusters
            };
            engine.update_preferences(preferences).unwrap_or(());
            Ok(AxumJson(ApiResponse::success(
                "Preferences updated".to_string(),
            )))
        } else {
            Ok(AxumJson(ApiResponse::error(
                "Discovery engine not found".to_string(),
            )))
        }
    } else {
        Ok(AxumJson(ApiResponse::error(
            "Failed to access discovery engines".to_string(),
        )))
    }
}

/// Get discovery patterns
async fn get_discovery_patterns(
    Path(engine_id): Path<String>,
) -> Result<AxumJson<ApiResponse<serde_json::Value>>, StatusCode> {
    if let Ok(engines) = DISCOVERY_ENGINES.lock() {
        if let Some(_engine) = engines.get(&engine_id) {
            // For now, return sample patterns
            let patterns = serde_json::json!([
                {
                    "type": "anomaly",
                    "description": "Isolated cluster with unique characteristics",
                    "significance": 0.8
                },
                {
                    "type": "bridge",
                    "description": "Cluster connecting disparate topics",
                    "significance": 0.9
                }
            ]);

            Ok(AxumJson(ApiResponse::success(patterns)))
        } else {
            Ok(AxumJson(ApiResponse::error(
                "Discovery engine not found".to_string(),
            )))
        }
    } else {
        Ok(AxumJson(ApiResponse::error(
            "Failed to access discovery engines".to_string(),
        )))
    }
}
