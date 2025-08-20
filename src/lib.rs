//! # BriefXAI
//!
//! BriefXAI is a production-ready platform for analyzing AI conversations at scale,
//! extracting insights, identifying patterns, and visualizing conversation clusters.
//!
//! ## Features
//!
//! - **Smart Analysis** - Extract facets, sentiments, and patterns automatically
//! - **High Performance** - Process thousands of conversations concurrently
//! - **Pause/Resume** - Stop and resume long-running analyses
//! - **Multi-Provider** - Support for OpenAI, Ollama, and other LLM providers
//! - **Real-time Results** - Stream insights via WebSocket
//! - **Data Privacy** - PII detection and local processing options
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use briefxai::{BriefXAIConfig, analyze_conversations};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load configuration
//!     let config = BriefXAIConfig::from_file("config.toml")?;
//!     
//!     // Analyze conversations
//!     let results = analyze_conversations("conversations.json", &config).await?;
//!     
//!     // Process results
//!     println!("Found {} clusters", results.clusters.len());
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - [`types`] - Core data types and structures
//! - [`config`] - Configuration management
//! - [`analysis`] - Analysis engine and session management
//! - [`preprocessing`] - Data validation and preprocessing
//! - [`llm`] - LLM provider integrations
//! - [`web`] - Web server and API endpoints
//! - [`clustering`] - Clustering algorithms
//! - [`embeddings`] - Embedding generation
//! - [`facets`] - Facet extraction

pub mod types;
pub mod types_extended;
pub mod config;
pub mod embeddings;
pub mod clustering;
pub mod facets;
pub mod llm;
pub mod umap;
pub mod web;
pub mod web_clio;
pub mod utils;
pub mod prompts;
pub mod examples;
pub mod persistence;
pub mod persistence_v2;

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
pub mod monitoring;
pub mod error_recovery;
pub mod logging;

use anyhow::Result;
use std::path::Path;

// Re-export main types for convenience
pub use config::BriefXAIConfig;
pub use types::{Facet, FacetValue, ConversationData, ConversationCluster, BriefXAIResults};

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
/// use briefxai::{analyze_conversations, BriefXAIConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = BriefXAIConfig::default();
///     let results = analyze_conversations("data.json", &config).await?;
///     Ok(())
/// }
/// ```
pub async fn analyze_conversations(
    file_path: impl AsRef<Path>,
    config: &BriefXAIConfig,
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
pub async fn create_session(
    name: &str,
    config: &BriefXAIConfig,
) -> Result<String> {
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
pub async fn resume_session(session_id: &str) -> Result<()> {
    // Implementation would go here
    todo!("Implement session resumption")
}