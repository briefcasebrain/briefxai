use anyhow::{Result, Context};
use reqwest;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn};
use dashmap::DashMap;
use ollama_rs::{Ollama, generation::embeddings::request::{GenerateEmbeddingsRequest, EmbeddingsInput}};

use crate::config::{BriefXAIConfig, EmbeddingProvider};
use crate::types::ConversationData;
use crate::prompts::conversation_to_string;

/// Cache for embeddings to avoid recomputation
type EmbeddingCache = Arc<DashMap<String, Vec<f32>>>;

#[derive(Clone)]
pub struct EmbeddingGenerator {
    config: BriefXAIConfig,
    provider: Arc<dyn EmbeddingProviderTrait>,
    cache: EmbeddingCache,
    semaphore: Arc<Semaphore>,
}

#[async_trait::async_trait]
pub trait EmbeddingProviderTrait: Send + Sync {
    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn name(&self) -> &str;
}

impl EmbeddingGenerator {
    pub async fn new(config: BriefXAIConfig) -> Result<Self> {
        let provider: Arc<dyn EmbeddingProviderTrait> = match config.embedding_provider {
            EmbeddingProvider::OpenAI => {
                Arc::new(OpenAIEmbeddings::new(
                    config.embedding_model.clone(),
                    config.embedding_api_key.clone(),
                )?)
            }
            EmbeddingProvider::HuggingFace => {
                Arc::new(HuggingFaceEmbeddings::new(
                    config.embedding_model.clone(),
                    config.embedding_api_key.clone(),
                ).await?)
            }
            EmbeddingProvider::SentenceTransformers => {
                Arc::new(OllamaEmbeddings::new(
                    config.embedding_model.clone(),
                ).await?)
            }
            _ => {
                Arc::new(OllamaEmbeddings::new(
                    config.embedding_model.clone(),
                ).await?)
            }
        };

        info!("Initialized {} embedding provider with model {}", 
              provider.name(), config.embedding_model);

        Ok(Self {
            config: config.clone(),
            provider,
            cache: Arc::new(DashMap::new()),
            semaphore: Arc::new(Semaphore::new(config.embed_batch_size.min(10))),
        })
    }

    pub async fn generate_batch(&self, conversations: &[ConversationData]) -> Result<Vec<Vec<f32>>> {
        info!("Generating embeddings for {} conversations", conversations.len());
        
        let mut embeddings = Vec::with_capacity(conversations.len());
        let batch_size = self.config.embed_batch_size;
        
        // Process in batches with progress tracking
        let pb = if self.config.verbose {
            Some(indicatif::ProgressBar::new(conversations.len() as u64))
        } else {
            None
        };
        
        for chunk in conversations.chunks(batch_size) {
            let texts: Vec<String> = chunk
                .iter()
                .map(conversation_to_string)
                .collect();
            
            // Check cache first
            let mut cached_embeddings = Vec::new();
            let mut texts_to_compute = Vec::new();
            let mut indices_to_compute = Vec::new();
            
            for (i, text) in texts.iter().enumerate() {
                let cache_key = self.compute_cache_key(text);
                if let Some(embedding) = self.cache.get(&cache_key) {
                    cached_embeddings.push((i, embedding.clone()));
                } else {
                    texts_to_compute.push(text.clone());
                    indices_to_compute.push(i);
                }
            }
            
            // Generate new embeddings
            let mut new_embeddings = Vec::new();
            if !texts_to_compute.is_empty() {
                let _permit = self.semaphore.acquire().await?;
                new_embeddings = self.provider.generate(texts_to_compute.clone()).await
                    .with_context(|| format!("Failed to generate embeddings for batch of {}", texts_to_compute.len()))?;
                
                // Cache the new embeddings
                for (text, embedding) in texts_to_compute.iter().zip(new_embeddings.iter()) {
                    let cache_key = self.compute_cache_key(text);
                    self.cache.insert(cache_key, embedding.clone());
                }
            }
            
            // Combine cached and new embeddings in correct order
            let mut batch_embeddings = vec![vec![]; texts.len()];
            for (idx, embedding) in cached_embeddings {
                batch_embeddings[idx] = embedding;
            }
            for (idx, embedding) in indices_to_compute.iter().zip(new_embeddings.iter()) {
                batch_embeddings[*idx] = embedding.clone();
            }
            
            embeddings.extend(batch_embeddings);
            
            if let Some(ref pb) = pb {
                pb.inc(chunk.len() as u64);
            }
        }
        
        if let Some(pb) = pb {
            pb.finish_with_message("Embeddings generated");
        }
        
        // Normalize embeddings
        let mut normalized = embeddings.clone();
        normalize_embeddings(&mut normalized);
        
        Ok(normalized)
    }
    
    fn compute_cache_key(&self, text: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.update(self.config.embedding_model.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

// OpenAI Provider
#[derive(Clone)]
#[derive(Debug)]
struct OpenAIEmbeddings {
    model: String,
    api_key: String,
    client: reqwest::Client,
}

impl OpenAIEmbeddings {
    fn new(model: String, api_key: Option<String>) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .context("OpenAI API key not provided")?;
        
        Ok(Self {
            model,
            api_key,
            client: reqwest::Client::new(),
        })
    }
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[async_trait::async_trait]
impl EmbeddingProviderTrait for OpenAIEmbeddings {
    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = OpenAIEmbeddingRequest {
            model: self.model.clone(),
            input: texts,
        };
        
        let response = self.client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<OpenAIEmbeddingResponse>()
            .await?;
        
        Ok(response.data.into_iter().map(|e| e.embedding).collect())
    }
    
    fn dimension(&self) -> usize {
        match self.model.as_str() {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => 1536,
        }
    }
    
    fn name(&self) -> &str {
        "OpenAI"
    }
}

// Ollama Provider (for local embeddings)
#[derive(Clone)]
struct OllamaEmbeddings {
    model: String,
    ollama: Arc<Ollama>,
}

impl OllamaEmbeddings {
    async fn new(model: String) -> Result<Self> {
        let ollama = Ollama::default();
        
        // Check if model is available
        let models = ollama.list_local_models().await
            .context("Failed to connect to Ollama. Is it running?")?;
        
        let model_exists = models.iter().any(|m| m.name == model);
        if !model_exists {
            warn!("Model {} not found locally. Attempting to pull...", model);
            // You might want to pull the model here
        }
        
        Ok(Self {
            model,
            ollama: Arc::new(ollama),
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingProviderTrait for OllamaEmbeddings {
    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        for text in texts {
            let request = GenerateEmbeddingsRequest::new(self.model.clone(), EmbeddingsInput::Single(text));
            let response = self.ollama.generate_embeddings(request).await?;
            // Handle the case where embeddings might be nested
            if let Some(first_embedding) = response.embeddings.first() {
                embeddings.push(first_embedding.clone());
            }
        }
        
        Ok(embeddings)
    }
    
    fn dimension(&self) -> usize {
        // This depends on the model, but most are 768 or 1536
        match self.model.as_str() {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            _ => 768,
        }
    }
    
    fn name(&self) -> &str {
        "Ollama"
    }
}

// HuggingFace Provider
#[derive(Clone)]
struct HuggingFaceEmbeddings {
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl HuggingFaceEmbeddings {
    async fn new(model: String, api_key: Option<String>) -> Result<Self> {
        Ok(Self {
            model,
            api_key,
            client: reqwest::Client::new(),
        })
    }
}

#[derive(Serialize)]
struct HFEmbeddingRequest {
    inputs: Vec<String>,
}

#[derive(Deserialize)]
struct HFEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait::async_trait]
impl EmbeddingProviderTrait for HuggingFaceEmbeddings {
    async fn generate(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "https://api-inference.huggingface.co/models/{}",
            self.model
        );
        
        let mut request = self.client
            .post(&url)
            .json(&HFEmbeddingRequest { inputs: texts });
        
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        
        let response = request
            .send()
            .await?
            .error_for_status()?
            .json::<HFEmbeddingResponse>()
            .await?;
        
        Ok(response.embeddings)
    }
    
    fn dimension(&self) -> usize {
        match self.model.as_str() {
            "sentence-transformers/all-mpnet-base-v2" => 768,
            "sentence-transformers/all-MiniLM-L6-v2" => 384,
            _ => 768,
        }
    }
    
    fn name(&self) -> &str {
        "HuggingFace"
    }
}

// Utility functions
pub async fn generate_embeddings(
    config: &BriefXAIConfig,
    data: &[ConversationData],
) -> Result<Vec<Vec<f32>>> {
    let generator = EmbeddingGenerator::new(config.clone()).await?;
    generator.generate_batch(data).await
}

pub fn normalize_embeddings(embeddings: &mut Vec<Vec<f32>>) {
    for embedding in embeddings.iter_mut() {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in embedding.iter_mut() {
                *value /= norm;
            }
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ConversationData;
    use tokio_test;
    
    fn create_test_config() -> BriefXAIConfig {
        BriefXAIConfig {
            embedding_provider: EmbeddingProvider::OpenAI,
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_api_key: Some("test-key".to_string()),
            embed_batch_size: 5,
            verbose: false,
            ..Default::default()
        }
    }
    
    fn create_test_conversations() -> Vec<ConversationData> {
        vec![
            ConversationData {
                messages: vec![
                    crate::types::Message {
                        role: "user".to_string(),
                        content: "Hello, how are you?".to_string(),
                    },
                    crate::types::Message {
                        role: "assistant".to_string(),
                        content: "I'm doing well, thank you!".to_string(),
                    },
                ],
                metadata: std::collections::HashMap::new(),
            },
            ConversationData {
                messages: vec![
                    crate::types::Message {
                        role: "user".to_string(),
                        content: "What's the weather like?".to_string(),
                    },
                    crate::types::Message {
                        role: "assistant".to_string(),
                        content: "I don't have access to current weather data.".to_string(),
                    },
                ],
                metadata: std::collections::HashMap::new(),
            },
        ]
    }
    
    #[test]
    fn test_normalize_embeddings() {
        let mut embeddings = vec![
            vec![3.0, 4.0],
            vec![1.0, 0.0],
        ];
        
        normalize_embeddings(&mut embeddings);
        
        assert!((embeddings[0][0] - 0.6).abs() < 1e-6);
        assert!((embeddings[0][1] - 0.8).abs() < 1e-6);
        assert!((embeddings[1][0] - 1.0).abs() < 1e-6);
        assert!((embeddings[1][1] - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_normalize_embeddings_zero_vector() {
        let mut embeddings = vec![
            vec![0.0, 0.0],
            vec![5.0, 12.0],
        ];
        
        normalize_embeddings(&mut embeddings);
        
        // Zero vector should remain zero
        assert_eq!(embeddings[0], vec![0.0, 0.0]);
        
        // Non-zero vector should be normalized
        assert!((embeddings[1][0] - 5.0/13.0).abs() < 1e-6);
        assert!((embeddings[1][1] - 12.0/13.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0, 0.0];
        
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity_edge_cases() {
        let zero = vec![0.0, 0.0];
        let unit = vec![1.0, 0.0];
        let negative = vec![-1.0, 0.0];
        
        // Similarity with zero vector
        assert_eq!(cosine_similarity(&zero, &unit), 0.0);
        
        // Opposite vectors
        assert!((cosine_similarity(&unit, &negative) + 1.0).abs() < 1e-6);
        
        // Identical vectors
        assert!((cosine_similarity(&unit, &unit) - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0, 0.0];
        
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
        assert!((cosine_distance(&a, &c) - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let c = vec![1.0, 1.0];
        
        assert_eq!(euclidean_distance(&a, &a), 0.0);
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
        assert!((euclidean_distance(&a, &c) - std::f32::consts::SQRT_2).abs() < 1e-6);
    }
    
    #[test]
    fn test_embedding_generator_cache_key() {
        let config = create_test_config();
        tokio_test::block_on(async {
            let generator = EmbeddingGenerator::new(config).await.unwrap();
            
            let key1 = generator.compute_cache_key("hello world");
            let key2 = generator.compute_cache_key("hello world");
            let key3 = generator.compute_cache_key("different text");
            
            // Same text should produce same key
            assert_eq!(key1, key2);
            
            // Different text should produce different key
            assert_ne!(key1, key3);
            
            // Keys should be hex strings
            assert!(key1.chars().all(|c| c.is_ascii_hexdigit()));
            assert_eq!(key1.len(), 64); // SHA256 hex string length
        });
    }
    
    #[tokio::test]
    async fn test_openai_embeddings_dimensions() {
        let embeddings = OpenAIEmbeddings::new(
            "text-embedding-ada-002".to_string(),
            Some("test-key".to_string())
        ).unwrap();
        
        assert_eq!(embeddings.dimension(), 1536);
        assert_eq!(embeddings.name(), "OpenAI");
        
        let embeddings_3_small = OpenAIEmbeddings::new(
            "text-embedding-3-small".to_string(),
            Some("test-key".to_string())
        ).unwrap();
        
        assert_eq!(embeddings_3_small.dimension(), 1536);
        
        let embeddings_3_large = OpenAIEmbeddings::new(
            "text-embedding-3-large".to_string(),
            Some("test-key".to_string())
        ).unwrap();
        
        assert_eq!(embeddings_3_large.dimension(), 3072);
    }
    
    #[tokio::test]
    async fn test_openai_embeddings_no_key() {
        // Remove any existing env var for clean test
        std::env::remove_var("OPENAI_API_KEY");
        
        let result = OpenAIEmbeddings::new(
            "text-embedding-3-small".to_string(),
            None
        );
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not provided"));
    }
    
    #[tokio::test]
    async fn test_ollama_embeddings_dimensions() {
        // This test might fail if Ollama is not running, which is expected
        let result = OllamaEmbeddings::new("nomic-embed-text".to_string()).await;
        
        if let Ok(embeddings) = result {
            assert_eq!(embeddings.dimension(), 768);
            assert_eq!(embeddings.name(), "Ollama");
        }
        // If Ollama is not running, the test should not fail the entire suite
    }
    
    #[tokio::test]
    async fn test_huggingface_embeddings_dimensions() {
        let embeddings = HuggingFaceEmbeddings::new(
            "sentence-transformers/all-mpnet-base-v2".to_string(),
            None
        ).await.unwrap();
        
        assert_eq!(embeddings.dimension(), 768);
        assert_eq!(embeddings.name(), "HuggingFace");
        
        let embeddings_minilm = HuggingFaceEmbeddings::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            None
        ).await.unwrap();
        
        assert_eq!(embeddings_minilm.dimension(), 384);
    }
    
    #[tokio::test]
    async fn test_embedding_generator_initialization() {
        let config = create_test_config();
        let result = EmbeddingGenerator::new(config).await;
        
        assert!(result.is_ok());
        let generator = result.unwrap();
        assert_eq!(generator.provider.name(), "OpenAI");
    }
    
    #[test]
    fn test_embedding_provider_names() {
        assert_eq!(EmbeddingProvider::OpenAI.to_string(), "OpenAI");
        assert_eq!(EmbeddingProvider::HuggingFace.to_string(), "HuggingFace");
        assert_eq!(EmbeddingProvider::SentenceTransformers.to_string(), "SentenceTransformers");
        assert_eq!(EmbeddingProvider::ONNX.to_string(), "ONNX");
    }
    
    #[test]
    fn test_embedding_provider_default() {
        let default_provider = EmbeddingProvider::default();
        assert!(matches!(default_provider, EmbeddingProvider::OpenAI));
    }
    
    #[test]
    fn test_distance_metrics_properties() {
        // Use normalized vectors for cosine distance test
        let a_raw = vec![1.0, 2.0, 3.0];
        let b_raw = vec![4.0, 5.0, 6.0];
        let c_raw = vec![2.0, 3.0, 4.0];
        
        let mut vectors = vec![a_raw.clone(), b_raw.clone(), c_raw.clone()];
        normalize_embeddings(&mut vectors);
        let a = &vectors[0];
        let b = &vectors[1];
        let c = &vectors[2];
        
        // Distance should be symmetric
        assert!((euclidean_distance(&a, &b) - euclidean_distance(&b, &a)).abs() < 1e-6);
        assert!((cosine_distance(&a, &b) - cosine_distance(&b, &a)).abs() < 1e-6);
        
        // Distance to self should be zero (or very close to zero for floating point)
        assert!((euclidean_distance(&a, &a)).abs() < 1e-6);
        assert!((cosine_distance(&a, &a)).abs() < 1e-6);
        
        // Triangle inequality for Euclidean distance
        let ab = euclidean_distance(&a, &b);
        let ac = euclidean_distance(&a, &c);
        let bc = euclidean_distance(&b, &c);
        
        assert!(ab <= ac + bc + 1e-6); // Small epsilon for floating point
        assert!(ac <= ab + bc + 1e-6);
        assert!(bc <= ab + ac + 1e-6);
    }
    
    #[test]
    fn test_large_embedding_normalization() {
        // Test with larger embeddings
        let mut large_embedding = vec![vec![0.0; 1000]];
        large_embedding[0][0] = 1.0;
        large_embedding[0][500] = 1.0;
        
        normalize_embeddings(&mut large_embedding);
        
        // Check normalization
        let norm: f32 = large_embedding[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}