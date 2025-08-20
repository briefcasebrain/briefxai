use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationData {
    pub messages: Vec<Message>,
    pub metadata: HashMap<String, String>,
}

impl ConversationData {
    pub fn len(&self) -> usize {
        self.messages.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Facet {
    pub name: String,
    pub question: String,
    #[serde(default)]
    pub prefill: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_criteria: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric: Option<(i32, i32)>,
}

impl Hash for Facet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.question.hash(state);
    }
}

impl PartialEq for Facet {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.question == other.question
    }
}

impl Eq for Facet {}

impl Default for Facet {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            question: "Default question".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: None,
        }
    }
}

impl Facet {
    pub fn should_make_clusters(&self) -> bool {
        self.summary_criteria.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetValue {
    pub facet: Facet,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFacetData {
    pub conversation: ConversationData,
    pub facet_values: Vec<FacetValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEmbedding {
    pub conversation: ConversationData,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationCluster {
    pub facet: Facet,
    pub summary: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<ConversationCluster>>,
    #[serde(skip)]
    pub parent: Option<Box<ConversationCluster>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indices: Option<Vec<usize>>,
}

impl Hash for ConversationCluster {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.summary.hash(state);
        self.name.hash(state);
        self.facet.hash(state);
    }
}

impl PartialEq for ConversationCluster {
    fn eq(&self, other: &Self) -> bool {
        self.summary == other.summary && self.name == other.name && self.facet == other.facet
    }
}

impl Eq for ConversationCluster {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefXAIResults {
    pub facet_data: Vec<Vec<FacetValue>>,
    pub hierarchy: Vec<ConversationCluster>,
    pub embeddings: Vec<Vec<f32>>,
    pub umap_coords: Vec<(f32, f32)>,
    pub website_path: String,
}

pub fn get_main_facets() -> Vec<Facet> {
    vec![
        Facet {
            name: "Request".to_string(),
            question: "What is the user's overall request for the assistant?".to_string(),
            prefill: "The user's overall request for the assistant is to".to_string(),
            summary_criteria: Some("The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job.".to_string()),
            numeric: None,
        },
        Facet {
            name: "Language".to_string(),
            question: "What are the main languages of the conversation? Do not mention programming languages and do not provide only language codes; only the full names for human languages, like 'English' or 'Arabic; Indonesian'. Do not include language families (just provide the general language, e.g., 'Arabic' and not 'Tripolitanian Arabic'; something that a standard langcodes package would be able to identify). Only consider languages in or after the human's first request. Output each language as a single full word with no other commentary.".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: None,
        },
        Facet {
            name: "Task".to_string(),
            question: "What task is the model being asked to perform in this conversation?".to_string(),
            prefill: "The task is to".to_string(),
            summary_criteria: Some("The cluster name should be a clear single sentence that captures the specific action or task the model is being asked to perform in the conversation. It should use an active verb and be written from the model's perspective. For example, 'Generate a creative writing story about space exploration' or 'Calculate the probability of winning a card game'".to_string()),
            numeric: None,
        },
        Facet {
            name: "Concerning".to_string(),
            question: r#"On a scale of 1 to 5, how concerning or potentially harmful is this conversation from a safety perspective? Use the following scale:
1: Not concerning at all. Completely benign conversation.
Example: A friendly chat about favorite books or movies.

2: Slightly concerning, but likely harmless.
Example: A heated but civil debate about politics.

3: Moderately concerning. May warrant a closer look.
Example: Discussion of controversial topics that could escalate.

4: Very concerning. Likely needs review.
Example: Conversations involving potential self-harm or harmful ideation.

5: Extremely concerning. Immediate review needed.
Example: Explicit threats of violence or illegal activities.
Answer with only a single number from 1 to 5."#.to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((1, 5)),
        },
    ]
}

pub fn get_generic_summary_facets() -> Vec<Facet> {
    vec![
        Facet {
            name: "Summary".to_string(),
            question: "Provide a concise summary of this data.".to_string(),
            prefill: String::new(),
            summary_criteria: Some("The cluster name should be a clear single sentence that accurately captures the examples.".to_string()),
            numeric: None,
        }
    ]
}