use crate::types::FacetValue;
use crate::types_extended::AnalysisCluster;
use serde::{Deserialize, Serialize};

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
    active_overlay: Option<FacetOverlay>,
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
            active_overlay: None,
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
        tracing::info!(
            "Applied facet overlay: {} with color scheme {:?}",
            overlay.facet_name,
            overlay.color_scheme
        );
        self.active_overlay = Some(overlay);
        Ok(())
    }

    /// Export map data for frontend consumption
    pub fn export_for_frontend(&self) -> serde_json::Value {
        let points: Vec<serde_json::Value> = self
            .coordinates
            .iter()
            .enumerate()
            .map(|(i, (x, y))| {
                serde_json::json!({
                    "id": i,
                    "x": x,
                    "y": y,
                    "cluster_id": if i < self.clusters.len() {
                        self.clusters[i].name.clone()
                    } else {
                        format!("cluster_{}", i)
                    }
                })
            })
            .collect();

        let mut result = serde_json::json!({
            "points": points,
            "clusters": self.clusters.len(),
            "coordinates": self.coordinates.len(),
            "facet_data": self.facet_data.len()
        });

        if let Some(ref overlay) = self.active_overlay {
            result["active_overlay"] = serde_json::json!({
                "facet_name": overlay.facet_name,
                "enabled": overlay.enabled,
                "color_scheme": format!("{:?}", overlay.color_scheme),
                "aggregation": format!("{:?}", overlay.aggregation),
            });
        }

        result
    }
}
