//! Common test utilities and fixtures

use briefx::types::ConversationData;
use briefx::BriefXAIConfig;

/// Creates a test configuration with sensible defaults
pub fn test_config() -> BriefXAIConfig {
    let mut config = BriefXAIConfig::default();
    config.llm_batch_size = 2;
    config.embed_batch_size = 2;
    config.verbose = false;
    config
}

/// Creates sample conversation data for testing
pub fn sample_conversations() -> Vec<ConversationData> {
    vec![
        ConversationData {
            messages: vec![
                briefx::types::Message {
                    role: "user".to_string(),
                    content: "What is the weather like?".to_string(),
                },
                briefx::types::Message {
                    role: "assistant".to_string(),
                    content: "I don't have access to real-time weather data.".to_string(),
                },
            ],
            metadata: std::collections::HashMap::new(),
        },
        ConversationData {
            messages: vec![
                briefx::types::Message {
                    role: "user".to_string(),
                    content: "Help me write code".to_string(),
                },
                briefx::types::Message {
                    role: "assistant".to_string(),
                    content: "I'd be happy to help with coding.".to_string(),
                },
            ],
            metadata: std::collections::HashMap::new(),
        },
    ]
}

/// Creates a temporary directory for test outputs
pub fn temp_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("briefxai_test_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("Failed to create temp dir");
    dir
}

/// Cleanup function for test directories
pub fn cleanup_temp_dir(path: &std::path::Path) {
    if path.exists() {
        std::fs::remove_dir_all(path).ok();
    }
}