use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefXAIConfig {
    // General params
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_verbose")]
    pub verbose: bool,
    #[serde(default = "default_llm_batch_size")]
    pub llm_batch_size: usize,
    #[serde(default = "default_embed_batch_size")]
    pub embed_batch_size: usize,
    #[serde(default = "default_dedup_data")]
    pub dedup_data: bool,

    // Generate Base Clusters params
    #[serde(default = "default_max_conversation_tokens")]
    pub max_conversation_tokens: usize,
    #[serde(default = "default_max_points_to_sample_inside_cluster")]
    pub max_points_to_sample_inside_cluster: usize,
    #[serde(default = "default_max_points_to_sample_outside_cluster")]
    pub max_points_to_sample_outside_cluster: usize,
    #[serde(default = "default_n_name_description_samples_per_cluster")]
    pub n_name_description_samples_per_cluster: usize,

    // Hierarchy params
    #[serde(default = "default_min_top_level_size")]
    pub min_top_level_size: usize,
    #[serde(default = "default_n_samples_outside_neighborhood")]
    pub n_samples_outside_neighborhood: usize,
    #[serde(default = "default_n_categorize_samples")]
    pub n_categorize_samples: usize,
    #[serde(default = "default_max_children_for_renaming")]
    pub max_children_for_renaming: usize,
    #[serde(default = "default_n_rename_samples")]
    pub n_rename_samples: usize,

    // LLM settings
    #[serde(default)]
    pub llm_provider: LlmProvider,
    #[serde(default)]
    pub llm_model: String,
    #[serde(default)]
    pub llm_api_key: Option<String>,
    #[serde(default)]
    pub llm_base_url: Option<String>,
    #[serde(default)]
    pub llm_extra_params: HashMap<String, serde_json::Value>,

    // Embedding settings
    #[serde(default)]
    pub embedding_provider: EmbeddingProvider,
    #[serde(default)]
    pub embedding_model: String,
    #[serde(default)]
    pub embedding_api_key: Option<String>,

    // UMAP settings
    #[serde(default = "default_umap_n_neighbors")]
    pub umap_n_neighbors: usize,
    #[serde(default = "default_umap_min_dist")]
    pub umap_min_dist: f32,
    #[serde(default = "default_umap_n_components")]
    pub umap_n_components: usize,

    // K-means settings
    #[serde(default = "default_kmeans_approximate")]
    pub kmeans_approximate: bool,
    #[serde(default = "default_kmeans_verbose")]
    pub kmeans_verbose: bool,

    // Website settings
    #[serde(default)]
    pub website_password: Option<String>,
    #[serde(default = "default_html_max_size_per_file")]
    pub html_max_size_per_file: usize,
    #[serde(default = "default_website_port")]
    pub website_port: u16,
    
    // Cache settings
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    
    // Batch size for analysis (not LLM batching)
    #[serde(default = "default_batch_size")]
    pub batch_size: Option<usize>,
    
    // Clio Features Configuration
    #[serde(default = "default_enable_clio_features")]
    pub enable_clio_features: bool,
    
    #[serde(default = "default_clio_privacy_min_cluster_size")]
    pub clio_privacy_min_cluster_size: usize,
    
    #[serde(default = "default_clio_privacy_merge_small")]
    pub clio_privacy_merge_small_clusters: bool,
    
    #[serde(default = "default_clio_privacy_facet_threshold")]
    pub clio_privacy_facet_threshold: f32,
    
    #[serde(default = "default_enable_interactive_map")]
    pub enable_interactive_map: bool,
    
    #[serde(default = "default_enable_investigation")]
    pub enable_investigation: bool,
    
    #[serde(default = "default_enable_discovery")]
    pub enable_discovery: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmProvider {
    OpenAI,
    Ollama,
    VLLM,
    HuggingFace,
}

impl Default for LlmProvider {
    fn default() -> Self {
        LlmProvider::OpenAI
    }
}

impl LlmProvider {
    pub fn as_str(&self) -> &str {
        match self {
            LlmProvider::OpenAI => "openai",
            LlmProvider::Ollama => "ollama",
            LlmProvider::VLLM => "vllm",
            LlmProvider::HuggingFace => "huggingface",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    OpenAI,
    HuggingFace,
    ONNX,
    SentenceTransformers,
}

impl Default for EmbeddingProvider {
    fn default() -> Self {
        EmbeddingProvider::OpenAI
    }
}

impl std::fmt::Display for EmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingProvider::OpenAI => write!(f, "OpenAI"),
            EmbeddingProvider::HuggingFace => write!(f, "HuggingFace"),
            EmbeddingProvider::ONNX => write!(f, "ONNX"),
            EmbeddingProvider::SentenceTransformers => write!(f, "SentenceTransformers"),
        }
    }
}

impl Default for BriefXAIConfig {
    fn default() -> Self {
        Self {
            seed: default_seed(),
            verbose: default_verbose(),
            llm_batch_size: default_llm_batch_size(),
            embed_batch_size: default_embed_batch_size(),
            dedup_data: default_dedup_data(),
            max_conversation_tokens: default_max_conversation_tokens(),
            max_points_to_sample_inside_cluster: default_max_points_to_sample_inside_cluster(),
            max_points_to_sample_outside_cluster: default_max_points_to_sample_outside_cluster(),
            n_name_description_samples_per_cluster: default_n_name_description_samples_per_cluster(),
            min_top_level_size: default_min_top_level_size(),
            n_samples_outside_neighborhood: default_n_samples_outside_neighborhood(),
            n_categorize_samples: default_n_categorize_samples(),
            max_children_for_renaming: default_max_children_for_renaming(),
            n_rename_samples: default_n_rename_samples(),
            llm_provider: LlmProvider::default(),
            llm_model: "gpt-4o-mini".to_string(),
            llm_api_key: None,
            llm_base_url: None,
            llm_extra_params: HashMap::new(),
            embedding_provider: EmbeddingProvider::default(),
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_api_key: None,
            umap_n_neighbors: default_umap_n_neighbors(),
            umap_min_dist: default_umap_min_dist(),
            umap_n_components: default_umap_n_components(),
            kmeans_approximate: default_kmeans_approximate(),
            kmeans_verbose: default_kmeans_verbose(),
            website_password: None,
            html_max_size_per_file: default_html_max_size_per_file(),
            website_port: default_website_port(),
            cache_dir: default_cache_dir(),
            batch_size: default_batch_size(),
            enable_clio_features: default_enable_clio_features(),
            clio_privacy_min_cluster_size: default_clio_privacy_min_cluster_size(),
            clio_privacy_merge_small_clusters: default_clio_privacy_merge_small(),
            clio_privacy_facet_threshold: default_clio_privacy_facet_threshold(),
            enable_interactive_map: default_enable_interactive_map(),
            enable_investigation: default_enable_investigation(),
            enable_discovery: default_enable_discovery(),
        }
    }
}

impl BriefXAIConfig {
    pub fn n_base_clusters(&self, n: usize) -> usize {
        n / 10
    }

    pub fn n_average_clusters_per_neighborhood(&self, n: usize) -> usize {
        (n / 10).max(1)
    }

    pub fn n_desired_higher_level_names_per_cluster(&self, n: usize) -> usize {
        (n / 3).max(1)
    }
}

// Default value functions
fn default_seed() -> u64 { 27 }
fn default_verbose() -> bool { true }
fn default_llm_batch_size() -> usize { 1000 }
fn default_embed_batch_size() -> usize { 1000 }
fn default_dedup_data() -> bool { true }
fn default_max_conversation_tokens() -> usize { 8000 }
fn default_max_points_to_sample_inside_cluster() -> usize { 10 }
fn default_max_points_to_sample_outside_cluster() -> usize { 10 }
fn default_n_name_description_samples_per_cluster() -> usize { 5 }
fn default_min_top_level_size() -> usize { 5 }
fn default_n_samples_outside_neighborhood() -> usize { 5 }
fn default_n_categorize_samples() -> usize { 5 }
fn default_max_children_for_renaming() -> usize { 10 }
fn default_n_rename_samples() -> usize { 5 }
fn default_umap_n_neighbors() -> usize { 15 }
fn default_umap_min_dist() -> f32 { 0.1 }
fn default_umap_n_components() -> usize { 2 }
fn default_kmeans_approximate() -> bool { true }
fn default_kmeans_verbose() -> bool { true }
fn default_html_max_size_per_file() -> usize { 10_000_000 }
fn default_website_port() -> u16 { 8080 }
fn default_cache_dir() -> PathBuf { 
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("briefxai")
}

fn default_batch_size() -> Option<usize> {
    Some(100)
}

// Clio feature defaults
fn default_enable_clio_features() -> bool {
    true
}

fn default_clio_privacy_min_cluster_size() -> usize {
    10
}

fn default_clio_privacy_merge_small() -> bool {
    true
}

fn default_clio_privacy_facet_threshold() -> f32 {
    0.05
}

fn default_enable_interactive_map() -> bool {
    true
}

fn default_enable_investigation() -> bool {
    true
}

fn default_enable_discovery() -> bool {
    true
}