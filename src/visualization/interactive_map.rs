use serde::{Serialize, Deserialize};
use crate::types::{FacetValue};
use crate::types_extended::AnalysisCluster;

/// Facet overlay configuration for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetOverlay {
    pub facet_name: String,
    pub enabled: bool,
    pub opacity: f32,
    pub color_scheme: ColorScheme,
    pub aggregation: AggregationType,
    pub threshold: f64,
}

/// Color scheme for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Categorical,
    Sequential,
    Diverging,
    Heatmap,
}

/// Aggregation type for data visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Count,
    Average,
    Sum,
    Maximum,
    Minimum,
    Mean,
    Median,
    Percentage,
    Prevalence,
}

/// Interactive map visualization for cluster exploration
#[derive(Debug, Clone)]
pub struct InteractiveMap {
    clusters: Vec<AnalysisCluster>,
    coordinates: Vec<(f32, f32)>,
    facet_data: Vec<Vec<FacetValue>>,
}

impl InteractiveMap {
    pub fn new(
        clusters: Vec<AnalysisCluster>,
        coordinates: Vec<(f32, f32)>,
        facet_data: &[Vec<FacetValue>],
    ) -> anyhow::Result<Self> {
        Ok(Self {
            clusters,
            coordinates,
            facet_data: facet_data.to_vec(),
        })
    }

    pub fn get_clusters(&self) -> &[AnalysisCluster] {
        &self.clusters
    }

    pub fn get_coordinates(&self) -> &[(f32, f32)] {
        &self.coordinates
    }

    /// Apply a facet overlay to the map
    pub fn apply_facet_overlay(&mut self, overlay: FacetOverlay) -> anyhow::Result<()> {
        // Stub implementation - in a real implementation, this would modify the visualization
        tracing::info!("Applied facet overlay: {} with color scheme {:?}", overlay.facet_name, overlay.color_scheme);
        Ok(())
    }

    /// Export map data for frontend consumption
    pub fn export_for_frontend(&self) -> serde_json::Value {
        // Stub implementation - in a real implementation, this would return structured map data
        serde_json::json!({
            "clusters": self.clusters.len(),
            "coordinates": self.coordinates.len(),
            "facet_data": self.facet_data.len()
        })
    }
}