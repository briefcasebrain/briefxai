pub mod provider_manager;

use anyhow::{Result, Context};
use reqwest;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Semaphore;
use tracing::{info, debug, warn};
use ollama_rs::{Ollama, generation::chat::{ChatMessage, request::ChatMessageRequest}, generation::options::GenerationOptions};
use async_trait::async_trait;

use crate::config::{BriefXAIConfig, LlmProvider};

#[async_trait]
pub trait LlmProviderTrait: Send + Sync {
    async fn complete(&self, prompt: &str, params: &LlmParams) -> Result<String>;
    async fn complete_with_system(&self, system: &str, prompt: &str, params: &LlmParams) -> Result<String>;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Default)]
pub struct LlmParams {
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Clone)]
pub struct LlmClient {
    config: BriefXAIConfig,
    provider: Arc<dyn LlmProviderTrait>,
    semaphore: Arc<Semaphore>,
}

impl LlmClient {
    pub async fn new(config: BriefXAIConfig) -> Result<Self> {
        let provider: Arc<dyn LlmProviderTrait> = match config.llm_provider {
            LlmProvider::OpenAI => {
                Arc::new(OpenAIProvider::new(
                    config.llm_model.clone(),
                    config.llm_api_key.clone(),
                )?)
            }
            LlmProvider::Ollama => {
                Arc::new(OllamaProvider::new(
                    config.llm_model.clone(),
                ).await?)
            }
            LlmProvider::VLLM => {
                Arc::new(VLLMProvider::new(
                    config.llm_model.clone(),
                    config.llm_base_url.clone().unwrap_or_else(|| "http://localhost:8000".to_string()),
                )?)
            }
            _ => {
                // Default to Ollama
                Arc::new(OllamaProvider::new(
                    config.llm_model.clone(),
                ).await?)
            }
        };

        info!("Initialized {} LLM provider with model {}", 
              provider.name(), config.llm_model);

        Ok(Self {
            config: config.clone(),
            provider,
            semaphore: Arc::new(Semaphore::new(2)), // Limit to 2 concurrent requests to avoid rate limits
        })
    }

    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let _permit = self.semaphore.acquire().await?;
        debug!("LLM completion requested: {} chars", prompt.len());
        
        // Use default parameters
        let params = LlmParams::default();
        
        // Retry logic for robustness
        let retry_config = crate::utils::RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_secs(2),
            max_delay: std::time::Duration::from_secs(30),
            exponential_base: 2.0,
            jitter: true,
        };
        
        crate::utils::retry_with_backoff(
            || self.provider.complete(prompt, &params),
            retry_config,
            "LLM completion"
        ).await
        .with_context(|| "Failed to complete LLM request after retries")
    }

    pub async fn complete_with_system(&self, system: &str, prompt: &str) -> Result<String> {
        let _permit = self.semaphore.acquire().await?;
        debug!("LLM completion with system prompt requested");
        
        // Use default parameters
        let params = LlmParams::default();
        
        // Retry logic for robustness
        let retry_config = crate::utils::RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_secs(2),
            max_delay: std::time::Duration::from_secs(30),
            exponential_base: 2.0,
            jitter: true,
        };
        
        crate::utils::retry_with_backoff(
            || self.provider.complete_with_system(system, prompt, &params),
            retry_config,
            "LLM completion with system"
        ).await
        .with_context(|| "Failed to complete LLM request with system prompt after retries")
    }

    pub async fn complete_batch(&self, prompts: Vec<String>) -> Result<Vec<String>> {
        let mut results = Vec::new();
        
        // Process in batches to respect rate limits
        let batch_size = self.config.llm_batch_size;
        for chunk in prompts.chunks(batch_size) {
            let mut batch_results = Vec::new();
            
            // Process batch concurrently but with semaphore limiting
            let futures: Vec<_> = chunk.iter()
                .map(|prompt| self.complete(prompt))
                .collect();
            
            for future in futures {
                batch_results.push(future.await?);
            }
            
            results.extend(batch_results);
            
            // Small delay between batches to be respectful to APIs
            if chunk.len() == batch_size {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
        
        Ok(results)
    }
}

// OpenAI Provider
#[derive(Clone)]
struct OpenAIProvider {
    model: String,
    api_key: String,
    client: reqwest::Client,
}

impl OpenAIProvider {
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
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    max_tokens: Option<u32>,
    temperature: f32,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    content: String,
}

#[async_trait::async_trait]
impl LlmProviderTrait for OpenAIProvider {
    async fn complete(&self, prompt: &str, params: &LlmParams) -> Result<String> {
        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: vec![
                OpenAIMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }
            ],
            max_tokens: params.max_tokens.map(|t| t as u32).or(Some(1000)),
            temperature: params.temperature.unwrap_or(0.7),
        };

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<OpenAIResponse>()
            .await?;

        response.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .context("No response from OpenAI")
    }

    async fn complete_with_system(&self, system: &str, prompt: &str, params: &LlmParams) -> Result<String> {
        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: vec![
                OpenAIMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                OpenAIMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }
            ],
            max_tokens: params.max_tokens.map(|t| t as u32).or(Some(1000)),
            temperature: params.temperature.unwrap_or(0.7),
        };

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<OpenAIResponse>()
            .await?;

        response.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .context("No response from OpenAI")
    }

    fn name(&self) -> &str {
        "OpenAI"
    }
}

// Ollama Provider
#[derive(Clone)]
struct OllamaProvider {
    model: String,
    ollama: Arc<Ollama>,
}

impl OllamaProvider {
    async fn new(model: String) -> Result<Self> {
        let ollama = Ollama::default();
        
        // Check if model is available
        let models = ollama.list_local_models().await
            .context("Failed to connect to Ollama. Is it running?")?;
        
        let model_exists = models.iter().any(|m| m.name == model);
        if !model_exists {
            warn!("Model {} not found locally. You may need to pull it first", model);
        }
        
        Ok(Self {
            model,
            ollama: Arc::new(ollama),
        })
    }
}

#[async_trait::async_trait]
impl LlmProviderTrait for OllamaProvider {
    async fn complete(&self, prompt: &str, params: &LlmParams) -> Result<String> {
        let messages = vec![
            ChatMessage::user(prompt.to_string()),
        ];
        
        let mut request = ChatMessageRequest::new(self.model.clone(), messages);
        
        // Apply parameters if provided
        if let Some(temperature) = params.temperature {
            let options = GenerationOptions::default().temperature(temperature);
            request = request.options(options);
        }
        
        let response = self.ollama.send_chat_messages(request).await?;
        
        Ok(response.message.content)
    }

    async fn complete_with_system(&self, system: &str, prompt: &str, params: &LlmParams) -> Result<String> {
        let messages = vec![
            ChatMessage::system(system.to_string()),
            ChatMessage::user(prompt.to_string()),
        ];
        
        let mut request = ChatMessageRequest::new(self.model.clone(), messages);
        
        // Apply parameters if provided
        if let Some(temperature) = params.temperature {
            let options = GenerationOptions::default().temperature(temperature);
            request = request.options(options);
        }
        
        let response = self.ollama.send_chat_messages(request).await?;
        
        Ok(response.message.content)
    }

    fn name(&self) -> &str {
        "Ollama"
    }
}

// VLLM Provider (for self-hosted models)
#[derive(Clone)]
struct VLLMProvider {
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl VLLMProvider {
    fn new(model: String, base_url: String) -> Result<Self> {
        Ok(Self {
            model,
            base_url,
            client: reqwest::Client::new(),
        })
    }
}

#[derive(Serialize)]
struct VLLMRequest {
    model: String,
    messages: Vec<VLLMMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct VLLMMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct VLLMResponse {
    choices: Vec<VLLMChoice>,
}

#[derive(Deserialize)]
struct VLLMChoice {
    message: VLLMResponseMessage,
}

#[derive(Deserialize)]
struct VLLMResponseMessage {
    content: String,
}

#[async_trait::async_trait]
impl LlmProviderTrait for VLLMProvider {
    async fn complete(&self, prompt: &str, params: &LlmParams) -> Result<String> {
        let request = VLLMRequest {
            model: self.model.clone(),
            messages: vec![
                VLLMMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }
            ],
            max_tokens: params.max_tokens.map(|t| t as u32).unwrap_or(1000),
            temperature: params.temperature.unwrap_or(0.7),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<VLLMResponse>()
            .await?;

        response.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .context("No response from VLLM")
    }

    async fn complete_with_system(&self, system: &str, prompt: &str, params: &LlmParams) -> Result<String> {
        let request = VLLMRequest {
            model: self.model.clone(),
            messages: vec![
                VLLMMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                VLLMMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }
            ],
            max_tokens: params.max_tokens.map(|t| t as u32).unwrap_or(1000),
            temperature: params.temperature.unwrap_or(0.7),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<VLLMResponse>()
            .await?;

        response.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .context("No response from VLLM")
    }

    fn name(&self) -> &str {
        "VLLM"
    }
}