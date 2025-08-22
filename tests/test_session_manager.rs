#[cfg(test)]
mod tests {
    use briefx::analysis::session_manager::*;
    use briefx::config::BriefXAIConfig;
    use briefx::persistence_v2::*;
    use briefx::types::{ConversationData, Message};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn setup_test_persistence() -> (Arc<EnhancedPersistenceLayer>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let persistence = Arc::new(EnhancedPersistenceLayer::new(db_path).await.unwrap());
        (persistence, temp_dir)
    }

    #[tokio::test]
    async fn test_session_creation() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        let config = BriefXAIConfig::default();
        let session = persistence
            .session_manager()
            .create_session(config)
            .await
            .unwrap();

        assert_eq!(session.status, SessionStatus::Pending);
        assert_eq!(session.current_batch, 0);
        assert_eq!(session.processed_conversations, 0);
    }

    #[tokio::test]
    async fn test_pause_resume_session() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        let config = BriefXAIConfig::default();
        let session = persistence
            .session_manager()
            .create_session(config)
            .await
            .unwrap();

        // Update to running
        persistence
            .session_manager()
            .update_session_status(&session.id, SessionStatus::Running)
            .await
            .unwrap();

        // Pause
        persistence
            .session_manager()
            .pause_session(&session.id)
            .await
            .unwrap();

        let paused = persistence
            .session_manager()
            .get_session(&session.id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(paused.status, SessionStatus::Paused);

        // Resume
        let resumed = persistence
            .session_manager()
            .resume_session(&session.id)
            .await
            .unwrap();

        assert_eq!(resumed.status, SessionStatus::Running);
    }

    #[tokio::test]
    async fn test_batch_progress_tracking() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        let config = BriefXAIConfig::default();
        let session = persistence
            .session_manager()
            .create_session(config)
            .await
            .unwrap();

        // Save batch progress
        persistence
            .session_manager()
            .save_batch_progress(
                &session.id,
                0,
                BatchStatus::Completed,
                Some(serde_json::json!({"result": "test"})),
                None,
            )
            .await
            .unwrap();

        persistence
            .session_manager()
            .save_batch_progress(
                &session.id,
                1,
                BatchStatus::Completed,
                Some(serde_json::json!({"result": "test2"})),
                None,
            )
            .await
            .unwrap();

        // Get last successful batch
        let last_batch = persistence
            .session_manager()
            .get_last_successful_batch(&session.id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(last_batch, 1);
    }

    #[tokio::test]
    async fn test_template_management() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        // Initialize default templates
        initialize_default_templates(&persistence).await.unwrap();

        // List templates
        let templates = persistence
            .template_repo()
            .list_templates(None)
            .await
            .unwrap();

        assert!(templates.len() >= 2);
        assert!(templates.iter().any(|t| t.id == "customer-support"));
        assert!(templates.iter().any(|t| t.id == "sales-conversations"));

        // Get specific template
        let support_template = persistence
            .template_repo()
            .get_template("customer-support")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(support_template.name, "Customer Support Analysis");
        assert_eq!(support_template.category, TemplateCategory::Support);
    }

    #[tokio::test]
    async fn test_provider_management() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        // Add providers
        let openai_provider = ProviderConfig {
            id: "openai-primary".to_string(),
            name: "OpenAI Primary".to_string(),
            provider_type: ProviderType::OpenAI,
            config: serde_json::json!({
                "api_key": "test-key",
                "model": "gpt-4o-mini"
            }),
            priority: 1,
            is_active: true,
            is_fallback: false,
            rate_limit: Some(serde_json::json!({
                "requests_per_minute": 60,
                "tokens_per_minute": 90000
            })),
            cost_per_token: Some(serde_json::json!({
                "input": 0.00015,
                "output": 0.0006
            })),
            created_at: 0,
            updated_at: 0,
        };

        persistence
            .provider_manager()
            .add_provider(openai_provider)
            .await
            .unwrap();

        let ollama_provider = ProviderConfig {
            id: "ollama-fallback".to_string(),
            name: "Ollama Fallback".to_string(),
            provider_type: ProviderType::Ollama,
            config: serde_json::json!({
                "base_url": "http://localhost:11434",
                "model": "llama2"
            }),
            priority: 10,
            is_active: true,
            is_fallback: true,
            rate_limit: None,
            cost_per_token: None,
            created_at: 0,
            updated_at: 0,
        };

        persistence
            .provider_manager()
            .add_provider(ollama_provider)
            .await
            .unwrap();

        // Get active providers
        let active = persistence
            .provider_manager()
            .get_active_providers()
            .await
            .unwrap();

        assert_eq!(active.len(), 2);
        assert_eq!(active[0].priority, 1); // Should be sorted by priority
        assert_eq!(active[1].priority, 10);

        // Get fallback providers
        let fallbacks = persistence
            .provider_manager()
            .get_fallback_providers()
            .await
            .unwrap();

        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0].id, "ollama-fallback");
    }

    #[tokio::test]
    async fn test_partial_results() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        let config = BriefXAIConfig::default();
        let session = persistence
            .session_manager()
            .create_session(config)
            .await
            .unwrap();

        // Store partial results
        persistence
            .store_partial_result(
                &session.id,
                Some(0),
                ResultType::Facet,
                serde_json::json!({
                    "facets": ["sentiment", "topic"],
                    "values": ["positive", "technical"]
                }),
            )
            .await
            .unwrap();

        persistence
            .store_partial_result(
                &session.id,
                Some(0),
                ResultType::Embedding,
                serde_json::json!({
                    "embeddings": [[0.1, 0.2, 0.3]]
                }),
            )
            .await
            .unwrap();

        // Retrieve partial results
        let facet_results = persistence
            .get_partial_results(&session.id, Some(ResultType::Facet))
            .await
            .unwrap();

        assert_eq!(facet_results.len(), 1);
        assert_eq!(facet_results[0].result_type, ResultType::Facet);

        let all_results = persistence
            .get_partial_results(&session.id, None)
            .await
            .unwrap();

        assert_eq!(all_results.len(), 2);
    }

    #[tokio::test]
    async fn test_response_cache() {
        let (persistence, _temp_dir) = setup_test_persistence().await;

        // Cache a response
        let cache_key = "test-prompt-hash";
        let output = serde_json::json!({
            "response": "This is a cached response",
            "tokens": 42
        });

        persistence
            .cache_response(
                cache_key,
                "llm",
                "openai",
                "input-hash-123",
                output.clone(),
                3600,
            )
            .await
            .unwrap();

        // Retrieve cached response
        let cached = persistence
            .get_cached_response(cache_key)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(cached, output);

        // Test cache miss
        let miss = persistence
            .get_cached_response("non-existent-key")
            .await
            .unwrap();

        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn test_batch_creation() {
        let conversations = vec![
            ConversationData {
                messages: vec![],
                metadata: Default::default(),
            },
            ConversationData {
                messages: vec![],
                metadata: Default::default(),
            },
            ConversationData {
                messages: vec![],
                metadata: Default::default(),
            },
            ConversationData {
                messages: vec![],
                metadata: Default::default(),
            },
            ConversationData {
                messages: vec![],
                metadata: Default::default(),
            },
        ];

        let batches = Batch::create_batches(&conversations, 2);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].number, 0);
        assert_eq!(batches[0].conversations.len(), 2);
        assert_eq!(batches[1].number, 1);
        assert_eq!(batches[1].conversations.len(), 2);
        assert_eq!(batches[2].number, 2);
        assert_eq!(batches[2].conversations.len(), 1);
    }
}
