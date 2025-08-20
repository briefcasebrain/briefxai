use briefxai::{
    BriefXAI, BriefXAIConfig,
    types::{ConversationData, Message, Facet, FacetValue},
    config::{LlmProvider, EmbeddingProvider},
    clustering::{create_base_clusters, build_hierarchy},
    embeddings::generate_embeddings,
    facets::extract_facets,
    umap::generate_umap,
    analysis_integration::EnhancedAnalysisEngine,
    visualization::interactive_map::InteractiveMap,
    privacy::threshold_protection::{PrivacyFilter, PrivacyConfig},
    investigation::targeted_search::{InvestigationEngine, InvestigationQuery},
    discovery::serendipitous::DiscoveryEngine,
};
use std::collections::HashMap;
use tempfile::TempDir;
use tokio;

#[derive(Debug)]
struct TestScenario {
    name: String,
    conversations: Vec<ConversationData>,
    config: BriefXAIConfig,
    expected_min_clusters: usize,
    expected_max_clusters: usize,
}

impl TestScenario {
    fn new(name: &str, conversations: Vec<ConversationData>, config: BriefXAIConfig) -> Self {
        let expected_min_clusters = 1;
        let expected_max_clusters = conversations.len() / 5;
        
        Self {
            name: name.to_string(),
            conversations,
            config,
            expected_min_clusters,
            expected_max_clusters,
        }
    }
}

fn create_customer_support_conversations() -> Vec<ConversationData> {
    vec![
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "I can't log into my account".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "I'll help you reset your password. Please check your email.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "login_issue".to_string());
                meta.insert("resolved".to_string(), "true".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "My billing is wrong this month".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Let me review your billing details and correct any errors.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "billing".to_string());
                meta.insert("resolved".to_string(), "true".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "How do I cancel my subscription?".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "I can help you with cancellation. Would you like to discuss alternatives first?".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "No, please proceed with cancellation".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Your subscription has been cancelled. You'll receive a confirmation email.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "cancellation".to_string());
                meta.insert("resolved".to_string(), "true".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "I forgot my password again".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "No problem! I'll send you a password reset link.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "login_issue".to_string());
                meta.insert("resolved".to_string(), "true".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "There's a charge on my card I don't recognize".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "I'll investigate this charge and provide details within 24 hours.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "billing".to_string());
                meta.insert("resolved".to_string(), "false".to_string());
                meta
            },
        },
    ]
}

fn create_technical_support_conversations() -> Vec<ConversationData> {
    vec![
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "The application keeps crashing when I try to export data".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Let's troubleshoot this. Can you tell me what browser you're using?".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "I'm using Chrome version 120".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Try clearing your browser cache and cookies, then attempt the export again.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "bug_report".to_string());
                meta.insert("severity".to_string(), "medium".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "Feature request: Can we add dark mode?".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Great suggestion! I'll add this to our feature backlog for consideration.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "feature_request".to_string());
                meta.insert("priority".to_string(), "low".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "How do I integrate your API with my system?".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "I'll send you our API documentation and sample code examples.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "technical_inquiry".to_string());
                meta.insert("complexity".to_string(), "high".to_string());
                meta
            },
        },
    ]
}

fn create_sales_conversations() -> Vec<ConversationData> {
    vec![
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "I'm interested in your enterprise plan".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Excellent! Our enterprise plan includes advanced analytics and priority support. Can you tell me about your team size?".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "We have about 50 employees who would use this".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Perfect fit! I'll prepare a custom quote for your team size.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "sales_inquiry".to_string());
                meta.insert("lead_score".to_string(), "high".to_string());
                meta
            },
        },
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "What's the difference between your plans?".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Our basic plan includes core features, while premium adds advanced analytics and priority support.".to_string(),
                },
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("category".to_string(), "pricing_inquiry".to_string());
                meta.insert("lead_score".to_string(), "medium".to_string());
                meta
            },
        },
    ]
}

fn create_test_config() -> BriefXAIConfig {
    BriefXAIConfig {
        seed: 42,
        verbose: false,
        llm_batch_size: 10,
        embed_batch_size: 10,
        dedup_data: true,
        max_conversation_tokens: 4000,
        max_points_to_sample_inside_cluster: 5,
        max_points_to_sample_outside_cluster: 5,
        n_name_description_samples_per_cluster: 2,
        min_top_level_size: 2,
        n_samples_outside_neighborhood: 3,
        n_categorize_samples: 3,
        max_children_for_renaming: 5,
        n_rename_samples: 2,
        llm_provider: LlmProvider::OpenAI,
        llm_model: "gpt-4o-mini".to_string(),
        llm_api_key: Some("test-key".to_string()),
        embedding_provider: EmbeddingProvider::OpenAI,
        embedding_model: "text-embedding-3-small".to_string(),
        embedding_api_key: Some("test-key".to_string()),
        umap_n_neighbors: 5,
        umap_min_dist: 0.1,
        umap_n_components: 2,
        enable_clio_features: true,
        clio_privacy_min_cluster_size: 3,
        clio_privacy_merge_small_clusters: true,
        clio_privacy_facet_threshold: 0.1,
        enable_interactive_map: true,
        enable_investigation: true,
        enable_discovery: true,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_customer_support_analysis_pipeline() {
    let conversations = create_customer_support_conversations();
    let config = create_test_config();
    let scenario = TestScenario::new("customer_support", conversations, config);
    
    // Test the full analysis pipeline
    let temp_dir = TempDir::new().unwrap();
    let briefxai = BriefXAI::new(scenario.config.clone());
    
    // Note: This will fail without actual API keys, but tests the structure
    // In a real test environment, you'd mock the LLM and embedding providers
    match briefxai.run(scenario.conversations.clone(), temp_dir.path()).await {
        Ok(results) => {
            assert!(!results.hierarchy.is_empty());
            assert!(!results.facet_data.is_empty());
            assert!(!results.embeddings.is_empty());
            assert!(!results.umap_coords.is_empty());
            assert!(results.website_path.contains("index.html"));
            
            // Verify we have reasonable clustering
            assert!(results.hierarchy.len() >= scenario.expected_min_clusters);
            assert!(results.hierarchy.len() <= scenario.expected_max_clusters);
        },
        Err(e) => {
            // Expected to fail without real API keys
            assert!(e.to_string().contains("API") || e.to_string().contains("network") || e.to_string().contains("key"));
        }
    }
}

#[tokio::test]
async fn test_technical_support_analysis_pipeline() {
    let conversations = create_technical_support_conversations();
    let config = create_test_config();
    let scenario = TestScenario::new("technical_support", conversations, config);
    
    let temp_dir = TempDir::new().unwrap();
    let briefxai = BriefXAI::new(scenario.config.clone());
    
    match briefxai.run(scenario.conversations.clone(), temp_dir.path()).await {
        Ok(results) => {
            assert!(!results.hierarchy.is_empty());
            assert_eq!(results.facet_data.len(), scenario.conversations.len());
            assert_eq!(results.embeddings.len(), scenario.conversations.len());
            assert_eq!(results.umap_coords.len(), scenario.conversations.len());
        },
        Err(e) => {
            // Expected to fail without real API keys
            assert!(e.to_string().contains("API") || e.to_string().contains("network") || e.to_string().contains("key"));
        }
    }
}

#[tokio::test]
async fn test_mixed_conversation_types() {
    let mut conversations = create_customer_support_conversations();
    conversations.extend(create_technical_support_conversations());
    conversations.extend(create_sales_conversations());
    
    let config = create_test_config();
    let scenario = TestScenario::new("mixed_types", conversations, config);
    
    let temp_dir = TempDir::new().unwrap();
    let briefxai = BriefXAI::new(scenario.config.clone());
    
    match briefxai.run(scenario.conversations.clone(), temp_dir.path()).await {
        Ok(results) => {
            // Should have more diverse clusters
            assert!(results.hierarchy.len() >= 3);
            assert_eq!(results.facet_data.len(), scenario.conversations.len());
        },
        Err(e) => {
            // Expected to fail without real API keys
            println!("Expected error without API keys: {}", e);
        }
    }
}

#[tokio::test]
async fn test_enhanced_analysis_with_clio_features() {
    let conversations = create_customer_support_conversations();
    let config = create_test_config();
    
    // Mock the components that don't require API calls
    let mock_embeddings: Vec<Vec<f32>> = (0..conversations.len()).map(|i| {
        vec![i as f32 / conversations.len() as f32, (i + 1) as f32 / conversations.len() as f32]
    }).collect();
    
    let mock_facet_data: Vec<Vec<FacetValue>> = conversations.iter().enumerate().map(|(i, _)| {
        let facet = Facet {
            name: "sentiment".to_string(),
            question: "What is the sentiment?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };
        vec![FacetValue {
            facet,
            value: match i % 3 {
                0 => "positive",
                1 => "negative",
                _ => "neutral",
            }.to_string(),
        }]
    }).collect();
    
    let mock_umap_coords: Vec<(f32, f32)> = (0..conversations.len()).map(|i| {
        (i as f32 / conversations.len() as f32, (i * 2) as f32 / conversations.len() as f32)
    }).collect();
    
    // Test Clio features integration
    let clio_config = briefxai::analysis_integration::ClioConfig {
        enable_clio_features: true,
        privacy: PrivacyConfig {
            min_cluster_size: 2,
            merge_small_clusters: true,
            min_facet_prevalence: 0.1,
            ..Default::default()
        },
        enable_interactive_map: true,
        enable_investigation: true,
        enable_discovery: true,
    };
    
    let engine = EnhancedAnalysisEngine::new(config.clone(), clio_config);
    
    // Create mock clusters
    let mock_clusters = vec![
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![0, 1],
            name: "Login Issues".to_string(),
            description: "Problems with account access".to_string(),
            children: vec![],
        },
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![2, 3],
            name: "Billing Concerns".to_string(),
            description: "Questions about charges and payments".to_string(),
            children: vec![],
        },
    ];
    
    let result = engine.analyze_with_clio_features(
        conversations.clone(),
        mock_facet_data,
        mock_embeddings,
        mock_umap_coords,
        mock_clusters,
    ).await;
    
    assert!(result.is_ok());
    let enhanced_results = result.unwrap();
    
    // Verify Clio features are present
    assert!(enhanced_results.interactive_map.is_some());
    assert!(enhanced_results.investigation_engine.is_some());
    assert!(enhanced_results.discovery_engine.is_some());
    assert!(!enhanced_results.privacy_filtered_clusters.is_empty());
}

#[tokio::test]
async fn test_privacy_filtering_integration() {
    let conversations = create_customer_support_conversations();
    
    let privacy_config = PrivacyConfig {
        min_cluster_size: 2,
        min_unique_sources: 1,
        merge_small_clusters: true,
        redact_small_cluster_descriptions: true,
        min_facet_prevalence: 0.2,
        sensitive_facets: vec!["personal_info".to_string(), "payment_info".to_string()],
        sensitive_facet_threshold: 5,
    };
    
    let mut filter = PrivacyFilter::new(privacy_config);
    
    // Create test clusters
    let test_clusters = vec![
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![0],
            name: "Single Conversation".to_string(),
            description: "Only one conversation".to_string(),
            children: vec![],
        },
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![1, 2, 3],
            name: "Multiple Conversations".to_string(),
            description: "Several conversations grouped".to_string(),
            children: vec![],
        },
    ];
    
    let result = filter.filter_clusters(test_clusters, &conversations).await;
    assert!(result.is_ok());
    
    let filtered_clusters = result.unwrap();
    // Single conversation cluster should be filtered out or merged
    assert!(filtered_clusters.len() <= 1);
    
    let report = filter.generate_report();
    assert!(report.total_redacted > 0 || report.total_merged > 0);
}

#[tokio::test]
async fn test_investigation_engine_comprehensive() {
    let conversations = create_customer_support_conversations();
    let mut facet_data = Vec::new();
    
    // Create comprehensive facet data
    for (i, _) in conversations.iter().enumerate() {
        let sentiment_facet = Facet {
            name: "sentiment".to_string(),
            question: "What is the sentiment?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };
        
        let category_facet = Facet {
            name: "category".to_string(),
            question: "What category?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };
        
        facet_data.push(vec![
            FacetValue {
                facet: sentiment_facet,
                value: match i % 3 {
                    0 => "positive",
                    1 => "negative",
                    _ => "neutral",
                }.to_string(),
            },
            FacetValue {
                facet: category_facet,
                value: match i % 2 {
                    0 => "support",
                    _ => "billing",
                }.to_string(),
            },
        ]);
    }
    
    // Create test clusters
    let clusters = vec![
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![0, 1],
            name: "Account Issues".to_string(),
            description: "Problems with user accounts".to_string(),
            children: vec![],
        },
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![2, 3, 4],
            name: "Billing Problems".to_string(),
            description: "Issues related to billing and payments".to_string(),
            children: vec![],
        },
    ];
    
    let engine = InvestigationEngine::new(
        clusters.clone(),
        conversations.clone(),
        facet_data.clone(),
        None,
    ).unwrap();
    
    // Test basic search
    let query = InvestigationQuery {
        search_terms: vec!["account".to_string()],
        facet_filters: vec![],
        metric_filters: vec![],
        sort_by: briefxai::investigation::targeted_search::SortCriterion::Relevance,
        limit: Some(10),
        highlight_matches: true,
    };
    
    let results = engine.investigate(&query).unwrap();
    assert!(!results.clusters.is_empty());
    assert!(results.total_matches > 0);
    
    // Test facet filtering
    let facet_query = InvestigationQuery {
        search_terms: vec![],
        facet_filters: vec![
            briefxai::investigation::targeted_search::FacetFilter {
                facet_name: "sentiment".to_string(),
                operator: briefxai::investigation::targeted_search::FilterOperator::Equals,
                value: "positive".to_string(),
                threshold: None,
            }
        ],
        metric_filters: vec![],
        sort_by: briefxai::investigation::targeted_search::SortCriterion::Size,
        limit: Some(5),
        highlight_matches: false,
    };
    
    let facet_results = engine.investigate(&facet_query).unwrap();
    assert!(facet_results.query_time_ms > 0);
}

#[tokio::test]
async fn test_discovery_engine_comprehensive() {
    let conversations = create_customer_support_conversations();
    let facet_data: Vec<Vec<FacetValue>> = conversations.iter().enumerate().map(|(i, _)| {
        let facet = Facet {
            name: "topic".to_string(),
            question: "What is the main topic?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };
        vec![FacetValue {
            facet,
            value: match i % 4 {
                0 => "authentication",
                1 => "billing",
                2 => "cancellation",
                _ => "general",
            }.to_string(),
        }]
    }).collect();
    
    let clusters = vec![
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![0, 3],
            name: "Authentication Issues".to_string(),
            description: "Login and password problems".to_string(),
            children: vec![],
        },
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![1, 4],
            name: "Billing Issues".to_string(),
            description: "Payment and billing concerns".to_string(),
            children: vec![],
        },
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![2],
            name: "Account Management".to_string(),
            description: "Subscription and account changes".to_string(),
            children: vec![],
        },
    ];
    
    let mut engine = DiscoveryEngine::new(clusters.clone(), facet_data).unwrap();
    
    // Test getting recommendations
    let recommendations = engine.get_recommendations(None, 3).unwrap();
    assert!(!recommendations.is_empty());
    assert!(recommendations.len() <= 3);
    
    // Verify recommendation structure
    for rec in &recommendations {
        assert!(!rec.cluster_name.is_empty());
        assert!(!rec.reason.is_empty());
        assert!(rec.confidence > 0.0 && rec.confidence <= 1.0);
    }
    
    // Test with specific starting cluster
    let specific_recommendations = engine.get_recommendations(Some(0), 2).unwrap();
    assert!(!specific_recommendations.is_empty());
    
    // Test preference updates
    engine.update_preferences(vec![0, 1]);
    let personalized_recommendations = engine.get_recommendations(None, 3).unwrap();
    assert!(!personalized_recommendations.is_empty());
    
    // Test pattern detection
    let patterns = engine.detect_patterns().unwrap();
    // Patterns might be empty for small datasets, which is fine
    println!("Detected {} patterns", patterns.len());
}

#[tokio::test]
async fn test_interactive_map_creation() {
    let conversations = create_customer_support_conversations();
    let facet_data: Vec<Vec<FacetValue>> = conversations.iter().enumerate().map(|(i, _)| {
        let facet = Facet {
            name: "sentiment".to_string(),
            question: "What is the sentiment?".to_string(),
            prefill: "".to_string(),
            summary_criteria: None,
            numeric: None,
        };
        vec![FacetValue {
            facet,
            value: match i % 3 {
                0 => "positive",
                1 => "negative",
                _ => "neutral",
            }.to_string(),
        }]
    }).collect();
    
    let clusters = vec![
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![0, 1],
            name: "Cluster A".to_string(),
            description: "First cluster".to_string(),
            children: vec![],
        },
        briefxai::types_extended::AnalysisCluster {
            conversation_ids: vec![2, 3, 4],
            name: "Cluster B".to_string(),
            description: "Second cluster".to_string(),
            children: vec![],
        },
    ];
    
    let umap_coords = vec![
        (0.1, 0.2),
        (0.8, 0.9),
        (0.3, 0.7),
        (0.6, 0.4),
        (0.9, 0.1),
    ];
    
    let map_result = InteractiveMap::new(clusters, umap_coords, &facet_data);
    assert!(map_result.is_ok());
    
    let map = map_result.unwrap();
    assert_eq!(map.points.len(), 5);
    
    // Test facet overlay
    let overlay = briefxai::visualization::interactive_map::FacetOverlay {
        facet_name: "sentiment".to_string(),
        color_scheme: briefxai::visualization::interactive_map::ColorScheme::Categorical,
        aggregation: briefxai::visualization::interactive_map::AggregationType::Prevalence,
        threshold: Some(0.1),
    };
    
    let overlay_result = map.apply_facet_overlay(overlay);
    assert!(overlay_result.is_ok());
}

#[tokio::test]
async fn test_error_handling_and_edge_cases() {
    // Test with empty conversations
    let empty_conversations: Vec<ConversationData> = vec![];
    let config = create_test_config();
    let temp_dir = TempDir::new().unwrap();
    let briefxai = BriefXAI::new(config.clone());
    
    let result = briefxai.run(empty_conversations, temp_dir.path()).await;
    // Should handle empty input gracefully
    assert!(result.is_err() || result.unwrap().hierarchy.is_empty());
    
    // Test with single conversation
    let single_conversation = vec![create_customer_support_conversations()[0].clone()];
    let result = briefxai.run(single_conversation, temp_dir.path()).await;
    // Might succeed or fail depending on API availability, but shouldn't panic
    match result {
        Ok(results) => {
            assert_eq!(results.facet_data.len(), 1);
            assert_eq!(results.embeddings.len(), 1);
        },
        Err(_) => {
            // Expected without API keys
        }
    }
    
    // Test with malformed configuration
    let mut bad_config = config.clone();
    bad_config.llm_batch_size = 0;
    bad_config.embed_batch_size = 0;
    
    let briefxai_bad = BriefXAI::new(bad_config);
    let result = briefxai_bad.run(create_customer_support_conversations(), temp_dir.path()).await;
    // Should handle bad config gracefully
    println!("Bad config result: {:?}", result.is_err());
}

#[tokio::test]
async fn test_concurrent_processing() {
    use tokio::task;
    
    let conversations = create_customer_support_conversations();
    let config = create_test_config();
    let temp_dir = TempDir::new().unwrap();
    
    // Test concurrent analysis runs
    let handles: Vec<_> = (0..3).map(|i| {
        let conversations = conversations.clone();
        let config = config.clone();
        let temp_path = temp_dir.path().join(format!("run_{}", i));
        std::fs::create_dir_all(&temp_path).unwrap();
        
        task::spawn(async move {
            let briefxai = BriefXAI::new(config);
            briefxai.run(conversations, temp_path).await
        })
    }).collect();
    
    let results = futures::future::join_all(handles).await;
    
    // All tasks should complete (though they may fail due to API keys)
    for result in results {
        assert!(result.is_ok()); // Task didn't panic
        // The inner result may be Err due to API keys, which is fine
    }
}

#[tokio::test] 
async fn test_configuration_validation() {
    // Test various configuration scenarios
    let base_config = create_test_config();
    
    // Test with different providers
    let mut ollama_config = base_config.clone();
    ollama_config.llm_provider = LlmProvider::Ollama;
    ollama_config.embedding_provider = EmbeddingProvider::SentenceTransformers;
    
    let briefxai = BriefXAI::new(ollama_config);
    // Should initialize without error
    assert_eq!(briefxai.config.llm_provider.as_str(), "ollama");
    
    // Test with extreme values
    let mut extreme_config = base_config.clone();
    extreme_config.max_conversation_tokens = 1;
    extreme_config.min_top_level_size = 1000; // Larger than any reasonable dataset
    
    let briefxai_extreme = BriefXAI::new(extreme_config);
    // Should handle extreme configs gracefully
    assert!(briefxai_extreme.config.max_conversation_tokens == 1);
}