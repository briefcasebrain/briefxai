//! # BriefXAI
//!
//! High-performance conversation analysis platform for extracting insights from AI conversations.
//!
//! ## Architecture Overview
//!
//! BriefXAI is organized into several core modules:
//! - **Analysis Pipeline**: Orchestrates the complete analysis workflow
//! - **Data Processing**: Handles preprocessing, validation, and transformation
//! - **Machine Learning**: Clustering, embeddings, and pattern detection
//! - **Integration Layer**: LLM providers and external services
//! - **Web Interface**: REST API and WebSocket endpoints
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use briefx::{BriefXAIConfig, types::ConversationData};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = BriefXAIConfig::default();
//!     // Application logic here
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod embeddings;
pub mod types;
pub mod types_extended;
// pub mod clio_core;
// pub mod clio_api;
pub mod clustering;
pub mod examples;
pub mod facets;
pub mod llm;
pub mod persistence;
pub mod persistence_v2;
pub mod prompts;
pub mod umap;
pub mod utils;
pub mod web;
pub mod web_clio;

/// Analysis module containing session management and streaming capabilities
pub mod analysis {
    pub mod session_manager;
    pub mod streaming;
}

/// Preprocessing module for data validation and cleaning
pub mod preprocessing;

/// Visualization module for interactive data exploration
pub mod visualization {
    pub mod interactive_map;
}

/// Privacy module for PII detection and data protection
pub mod privacy {
    pub mod threshold_protection;
}

/// Investigation module for targeted search capabilities
pub mod investigation {
    pub mod targeted_search;
}

/// Discovery module for serendipitous insights
pub mod discovery {
    pub mod serendipitous;
}

pub mod analysis_integration;
pub mod error_recovery;
pub mod logging;
pub mod monitoring;

use anyhow::Result;
use std::path::Path;

// Re-export main types for convenience
pub use config::BriefXAIConfig;
pub use types::{BriefXAIResults, ConversationCluster, ConversationData, Facet, FacetValue};

/// Analyzes conversations from a file and returns results
///
/// # Arguments
///
/// * `file_path` - Path to the JSON file containing conversations
/// * `config` - Configuration for the analysis
///
/// # Returns
///
/// Returns `BriefXAIResults` containing analysis insights
///
/// # Examples
///
/// ```rust,no_run
/// use briefx::{analyze_conversations, BriefXAIConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = BriefXAIConfig::default();
///     let results = analyze_conversations("data.json", &config).await?;
///     Ok(())
/// }
/// ```
pub async fn analyze_conversations(
    _file_path: impl AsRef<Path>,
    _config: &BriefXAIConfig,
) -> Result<BriefXAIResults> {
    // Implementation would go here
    // This is a placeholder that would integrate with the actual analysis pipeline
    todo!("Implement main analysis function")
}

/// Creates a new analysis session
///
/// # Arguments
///
/// * `name` - Name for the analysis session
/// * `config` - Configuration for the session
///
/// # Returns
///
/// Returns a session ID that can be used to track the analysis
pub async fn create_session(_name: &str, _config: &BriefXAIConfig) -> Result<String> {
    // Implementation would go here
    todo!("Implement session creation")
}

/// Resumes a paused analysis session
///
/// # Arguments
///
/// * `session_id` - ID of the session to resume
///
/// # Returns
///
/// Returns Ok(()) if the session was successfully resumed
pub async fn resume_session(_session_id: &str) -> Result<()> {
    // Implementation would go here
    todo!("Implement session resumption")
}
