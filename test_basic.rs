// Basic test to verify core types and modules compile
use briefxai::{BriefXAIConfig, Facet, FacetValue, ConversationData};

fn main() {
    println!("Testing basic types...");
    
    // Test configuration
    let config = BriefXAIConfig::default();
    println!("Config created: {:?}", config.server_port);
    
    // Test facet types
    let facet = Facet {
        id: "test".to_string(),
        description: "Test facet".to_string(),
        evidence: vec!["evidence".to_string()],
        confidence: 0.9,
        metadata: Default::default(),
    };
    println!("Facet created: {}", facet.id);
    
    // Test conversation data
    let conversation = ConversationData {
        id: "conv1".to_string(),
        messages: vec![],
        metadata: Default::default(),
        embeddings: None,
        facets: vec![],
        cluster_id: None,
    };
    println!("Conversation created: {}", conversation.id);
    
    println!("Basic types test passed!");
}