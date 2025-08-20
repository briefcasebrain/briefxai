use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use tracing::info;
use tiktoken_rs::{cl100k_base, CoreBPE};
use unicode_normalization::UnicodeNormalization;

use crate::types::{ConversationData, Message};
use crate::preprocessing::validators::{CompositeValidator, ValidationResult, IssueSeverity};
use crate::preprocessing::language_detector::{LanguageDetector, LanguageStats};

// ============================================================================
// Preprocessing Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityReport {
    pub total_conversations: usize,
    pub valid_conversations: usize,
    pub validation_results: ValidationResult,
    pub language_stats: LanguageStats,
    pub token_stats: TokenStats,
    pub recommendations: Vec<String>,
    pub auto_fixable_issues: usize,
    pub estimated_processing_time: f32,
    pub overall_quality_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStats {
    pub total_tokens: usize,
    pub avg_tokens_per_conversation: f32,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub conversations_over_limit: Vec<usize>,
    pub token_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingStrategy {
    pub batch_size: usize,
    pub batching_method: BatchingMethod,
    pub estimated_batches: usize,
    pub token_budget_per_batch: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchingMethod {
    FixedSize,
    TokenBudget,
    Similarity,
    Language,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingOptions {
    pub remove_duplicates: bool,
    pub remove_pii: bool,
    pub fix_encoding: bool,
    pub normalize_unicode: bool,
    pub truncate_long: bool,
    pub remove_empty: bool,
    pub min_quality_score: f32,
    pub target_language: Option<String>,
    pub max_tokens_per_conversation: usize,
}

impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            remove_duplicates: true,
            remove_pii: false,
            fix_encoding: true,
            normalize_unicode: true,
            truncate_long: true,
            remove_empty: true,
            min_quality_score: 0.3,
            target_language: None,
            max_tokens_per_conversation: 4000,
        }
    }
}

// ============================================================================
// Normalizers
// ============================================================================

pub trait Normalizer: Send + Sync {
    fn normalize(&self, conversation: ConversationData) -> ConversationData;
    fn name(&self) -> &str;
}

pub struct UnicodeNormalizer;

impl Normalizer for UnicodeNormalizer {
    fn normalize(&self, conversation: ConversationData) -> ConversationData {
        let normalized_messages = conversation
            .messages
            .into_iter()
            .map(|mut message| {
                message.content = message.content.nfc().collect::<String>();
                message
            })
            .collect();
        
        ConversationData {
            messages: normalized_messages,
            metadata: conversation.metadata,
        }
    }
    
    fn name(&self) -> &str {
        "UnicodeNormalizer"
    }
}

pub struct WhitespaceNormalizer;

impl Normalizer for WhitespaceNormalizer {
    fn normalize(&self, conversation: ConversationData) -> ConversationData {
        let normalized_messages = conversation
            .messages
            .into_iter()
            .map(|mut message| {
                // Normalize whitespace
                message.content = message.content
                    .lines()
                    .map(|line| line.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                
                // Remove multiple consecutive spaces
                while message.content.contains("  ") {
                    message.content = message.content.replace("  ", " ");
                }
                
                message
            })
            .collect();
        
        ConversationData {
            messages: normalized_messages,
            metadata: conversation.metadata,
        }
    }
    
    fn name(&self) -> &str {
        "WhitespaceNormalizer"
    }
}

pub struct EncodingFixer;

impl EncodingFixer {
    fn fix_mojibake(&self, text: &str) -> String {
        // Common mojibake patterns - simplified to avoid encoding issues
        let replacements = [
            ("Ã©", "é"),
            ("Ã¨", "è"),
            ("Ã ", "à"),
            ("Ã§", "ç"),
            ("Ã±", "ñ"),
            ("Ã¼", "ü"),
            ("Ã¶", "ö"),
            ("Ã¤", "ä"),
            ("â€™", "'"),
            ("â€œ", "\""),
            ("â€", "\""),
            ("...", "…"),
        ];
        
        let mut fixed = text.to_string();
        for (pattern, replacement) in &replacements {
            fixed = fixed.replace(pattern, replacement);
        }
        
        fixed
    }
}

impl Normalizer for EncodingFixer {
    fn normalize(&self, conversation: ConversationData) -> ConversationData {
        let normalized_messages = conversation
            .messages
            .into_iter()
            .map(|mut message| {
                message.content = self.fix_mojibake(&message.content);
                message
            })
            .collect();
        
        ConversationData {
            messages: normalized_messages,
            metadata: conversation.metadata,
        }
    }
    
    fn name(&self) -> &str {
        "EncodingFixer"
    }
}

// ============================================================================
// Token Counter
// ============================================================================

pub struct TokenCounter {
    encoder: CoreBPE,
}

impl TokenCounter {
    pub fn new() -> Result<Self> {
        let encoder = cl100k_base()?;
        Ok(Self { encoder })
    }
    
    pub fn count_tokens(&self, text: &str) -> usize {
        self.encoder.encode_with_special_tokens(text).len()
    }
    
    pub fn count_conversation(&self, conversation: &ConversationData) -> usize {
        conversation
            .messages.iter()
            .map(|m| self.count_tokens(&m.content) + self.count_tokens(&m.role) + 4)
            .sum()
    }
    
    pub fn analyze_dataset(&self, conversations: &[ConversationData]) -> TokenStats {
        let mut total_tokens = 0;
        let mut max_tokens = 0;
        let mut min_tokens = usize::MAX;
        let mut conversations_over_limit = Vec::new();
        let mut token_distribution = HashMap::new();
        
        for (i, conversation) in conversations.iter().enumerate() {
            let tokens = self.count_conversation(conversation);
            total_tokens += tokens;
            max_tokens = max_tokens.max(tokens);
            min_tokens = min_tokens.min(tokens);
            
            if tokens > 4000 {
                conversations_over_limit.push(i);
            }
            
            // Bucket tokens for distribution
            let bucket = match tokens {
                0..=100 => "0-100",
                101..=500 => "101-500",
                501..=1000 => "501-1000",
                1001..=2000 => "1001-2000",
                2001..=4000 => "2001-4000",
                _ => "4000+",
            };
            
            *token_distribution.entry(bucket.to_string()).or_insert(0) += 1;
        }
        
        let avg_tokens_per_conversation = if conversations.is_empty() {
            0.0
        } else {
            total_tokens as f32 / conversations.len() as f32
        };
        
        if conversations.is_empty() {
            min_tokens = 0;
        }
        
        TokenStats {
            total_tokens,
            avg_tokens_per_conversation,
            max_tokens,
            min_tokens,
            conversations_over_limit,
            token_distribution,
        }
    }
    
    pub fn truncate_to_limit(&self, conversation: ConversationData, max_tokens: usize) -> ConversationData {
        let mut truncated = Vec::new();
        let mut total_tokens = 0;
        
        for message in &conversation.messages {
            let message_tokens = self.count_tokens(&message.content) + self.count_tokens(&message.role) + 4;
            
            if total_tokens + message_tokens <= max_tokens {
                truncated.push(message.clone());
                total_tokens += message_tokens;
            } else if total_tokens < max_tokens {
                // Partially include this message
                let remaining_tokens = max_tokens - total_tokens - self.count_tokens(&message.role) - 4;
                if remaining_tokens > 10 {
                    let truncated_content = self.truncate_text(&message.content, remaining_tokens);
                    truncated.push(Message {
                        role: message.role.clone(),
                        content: format!("{}... [truncated]", truncated_content),
                    });
                }
                break;
            } else {
                break;
            }
        }
        
        ConversationData {
            messages: truncated,
            metadata: conversation.metadata,
        }
    }
    
    fn truncate_text(&self, text: &str, max_tokens: usize) -> String {
        let tokens = self.encoder.encode_with_special_tokens(text);
        if tokens.len() <= max_tokens {
            return text.to_string();
        }
        
        let truncated_tokens = &tokens[..max_tokens];
        self.encoder.decode(truncated_tokens.to_vec()).unwrap_or_else(|_| {
            // Fallback to character truncation
            text.chars().take(max_tokens * 4).collect()
        })
    }
}

// ============================================================================
// Smart Preprocessor
// ============================================================================

pub struct SmartPreprocessor {
    validators: CompositeValidator,
    normalizers: Vec<Box<dyn Normalizer>>,
    language_detector: LanguageDetector,
    token_counter: TokenCounter,
    options: PreprocessingOptions,
}

impl SmartPreprocessor {
    pub fn new(options: PreprocessingOptions) -> Result<Self> {
        Ok(Self {
            validators: CompositeValidator::new(),
            normalizers: vec![
                Box::new(UnicodeNormalizer),
                Box::new(WhitespaceNormalizer),
                Box::new(EncodingFixer),
            ],
            language_detector: LanguageDetector::new(),
            token_counter: TokenCounter::new()?,
            options,
        })
    }
    
    pub async fn analyze_data(&self, data: &[ConversationData]) -> DataQualityReport {
        info!("Analyzing {} conversations", data.len());
        
        // Run validators
        let validation_results = self.validators.aggregate_results(
            self.validators.validate_all(data)
        );
        
        // Analyze languages
        let language_stats = self.language_detector.analyze_dataset(data);
        
        // Count tokens
        let token_stats = self.token_counter.analyze_dataset(data);
        
        // Count valid conversations
        let valid_conversations = data.len() - validation_results.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .map(|i| i.conversation_index.unwrap_or(0))
            .collect::<HashSet<_>>()
            .len();
        
        // Count auto-fixable issues
        let auto_fixable_issues = validation_results.issues
            .iter()
            .filter(|i| matches!(i.issue_type, 
                crate::preprocessing::validators::IssueType::Empty |
                crate::preprocessing::validators::IssueType::Duplicate |
                crate::preprocessing::validators::IssueType::EncodingIssue
            ))
            .count();
        
        // Estimate processing time (rough estimate)
        let estimated_processing_time = (token_stats.total_tokens as f32 / 1000.0) * 0.5;
        
        // Calculate overall quality score
        let overall_quality_score = validation_results.quality_score * 0.4 +
            language_stats.confidence * 0.2 +
            (valid_conversations as f32 / data.len().max(1) as f32) * 0.4;
        
        // Generate recommendations
        let mut recommendations = validation_results.suggestions.clone();
        
        if language_stats.languages.len() > 1 {
            recommendations.push(format!(
                "Dataset contains {} different languages. Consider separating by language for better results.",
                language_stats.languages.len()
            ));
        }
        
        if token_stats.conversations_over_limit.len() > 10 {
            recommendations.push(format!(
                "{} conversations exceed token limit. Enable truncation or increase limit.",
                token_stats.conversations_over_limit.len()
            ));
        }
        
        if overall_quality_score < 0.5 {
            recommendations.push("Data quality is low. Consider cleaning data before analysis.".to_string());
        }
        
        DataQualityReport {
            total_conversations: data.len(),
            valid_conversations,
            validation_results,
            language_stats,
            token_stats,
            recommendations,
            auto_fixable_issues,
            estimated_processing_time,
            overall_quality_score,
        }
    }
    
    pub async fn auto_fix(&self, mut data: Vec<ConversationData>) -> Vec<ConversationData> {
        info!("Auto-fixing {} conversations", data.len());
        
        // Remove empty conversations
        if self.options.remove_empty {
            data.retain(|conv| !conv.is_empty());
        }
        
        // Remove duplicates
        if self.options.remove_duplicates {
            let mut seen = HashSet::new();
            data.retain(|conv| {
                let hash = self.hash_conversation(conv);
                seen.insert(hash)
            });
        }
        
        // Apply normalizers
        data = data
            .into_iter()
            .map(|conv| {
                let mut normalized = conv;
                for normalizer in &self.normalizers {
                    if self.should_apply_normalizer(normalizer.name()) {
                        normalized = normalizer.normalize(normalized);
                    }
                }
                normalized
            })
            .collect();
        
        // Truncate long conversations
        if self.options.truncate_long {
            data = data
                .into_iter()
                .map(|conv| {
                    self.token_counter.truncate_to_limit(conv, self.options.max_tokens_per_conversation)
                })
                .collect();
        }
        
        // Filter by language if specified
        if let Some(target_lang) = &self.options.target_language {
            data.retain(|conv| {
                let detected = self.language_detector.detect_conversation(conv);
                detected.language == *target_lang || detected.confidence < 0.5
            });
        }
        
        // Filter by quality score
        data.retain(|conv| {
            self.calculate_conversation_quality(conv) >= self.options.min_quality_score
        });
        
        info!("Auto-fix complete. {} conversations remaining", data.len());
        
        data
    }
    
    pub async fn suggest_batching(&self, data: &[ConversationData]) -> BatchingStrategy {
        let token_stats = self.token_counter.analyze_dataset(data);
        let language_stats = self.language_detector.analyze_dataset(data);
        
        // Determine best batching method
        let batching_method = if language_stats.languages.len() > 1 {
            BatchingMethod::Language
        } else if token_stats.max_tokens > 8000 {
            BatchingMethod::TokenBudget
        } else {
            BatchingMethod::FixedSize
        };
        
        // Calculate batch size
        let (batch_size, token_budget) = match batching_method {
            BatchingMethod::TokenBudget => {
                let budget = 100000;  // Typical LLM context limit
                let avg_tokens = token_stats.avg_tokens_per_conversation as usize;
                let size = (budget / avg_tokens.max(1)).min(100);
                (size, budget)
            }
            BatchingMethod::Language => {
                // Batch by language, aim for ~50 conversations per batch
                (50, 100000)
            }
            _ => {
                // Fixed size batching
                (100, 100000)
            }
        };
        
        let estimated_batches = (data.len() + batch_size - 1) / batch_size;
        
        BatchingStrategy {
            batch_size,
            batching_method,
            estimated_batches,
            token_budget_per_batch: token_budget,
        }
    }
    
    fn hash_conversation(&self, conversation: &ConversationData) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        for message in &conversation.messages {
            message.role.hash(&mut hasher);
            message.content.hash(&mut hasher);
        }
        hasher.finish()
    }
    
    fn should_apply_normalizer(&self, name: &str) -> bool {
        match name {
            "UnicodeNormalizer" => self.options.normalize_unicode,
            "EncodingFixer" => self.options.fix_encoding,
            _ => true,
        }
    }
    
    fn calculate_conversation_quality(&self, conversation: &ConversationData) -> f32 {
        if conversation.is_empty() {
            return 0.0;
        }
        
        let mut score = 1.0;
        
        // Penalize very short conversations
        if conversation.len() < 2 {
            score *= 0.5;
        }
        
        // Penalize conversations with very short messages
        let avg_message_length: usize = conversation
            .messages.iter()
            .map(|m| m.content.len())
            .sum::<usize>() / conversation.len().max(1);
        
        if avg_message_length < 20 {
            score *= 0.7;
        }
        
        // Penalize if no user/assistant messages
        let has_user = conversation.messages.iter().any(|m| m.role == "user");
        let has_assistant = conversation.messages.iter().any(|m| m.role == "assistant");
        
        if !has_user || !has_assistant {
            score *= 0.6;
        }
        
        score
    }
    
    pub fn create_sample_preview(
        &self,
        data: &[ConversationData],
        max_samples: usize,
    ) -> Vec<(usize, ConversationData, String)> {
        let mut samples = Vec::new();
        
        for (i, conversation) in data.iter().enumerate().take(max_samples) {
            let preview = if conversation.len() > 3 {
                ConversationData {
                    messages: conversation.messages[..3].to_vec(),
                    metadata: conversation.metadata.clone(),
                }
            } else {
                conversation.clone()
            };
            
            let detected_language = self.language_detector.detect_conversation(conversation);
            let quality = self.calculate_conversation_quality(conversation);
            let tokens = self.token_counter.count_conversation(conversation);
            
            let summary = format!(
                "Conv #{}: {} messages, {} tokens, {} (conf: {:.1}), quality: {:.1}",
                i + 1,
                conversation.len(),
                tokens,
                detected_language.language,
                detected_language.confidence,
                quality
            );
            
            samples.push((i, preview, summary));
        }
        
        samples
    }
}