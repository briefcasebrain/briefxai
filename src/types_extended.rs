use serde::{Deserialize, Serialize};

/// Extended cluster type for analysis features
/// This is used by the new Clio-inspired features for clustering analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCluster {
    pub conversation_ids: Vec<usize>,
    pub name: String,
    pub description: String,
    pub children: Vec<AnalysisCluster>,
}

impl AnalysisCluster {
    /// Convert from the original ConversationCluster type if needed
    pub fn from_conversation_cluster(
        cluster: &super::types::ConversationCluster,
        conversation_ids: Vec<usize>,
    ) -> Self {
        Self {
            conversation_ids,
            name: cluster.name.clone(),
            description: cluster.summary.clone(),
            children: cluster
                .children
                .as_ref()
                .map(|children| {
                    children
                        .iter()
                        .enumerate()
                        .map(|(i, child)| Self::from_conversation_cluster(child, vec![i]))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// Create a simple analysis cluster
    pub fn new(conversation_ids: Vec<usize>, name: String, description: String) -> Self {
        Self {
            conversation_ids,
            name,
            description,
            children: vec![],
        }
    }
}
