use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

use crate::types::ConversationData;

// ============================================================================
// Validation Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub suggestions: Vec<String>,
    pub quality_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub issue_type: IssueType,
    pub message: String,
    pub conversation_index: Option<usize>,
    pub message_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueType {
    Empty,
    TooShort,
    TooLong,
    Duplicate,
    MissingRole,
    InvalidFormat,
    PII,
    Profanity,
    LowQuality,
    Truncated,
    MixedLanguages,
    EncodingIssue,
}

// ============================================================================
// Base Validator Trait
// ============================================================================

pub trait Validator: Send + Sync {
    fn validate(&self, conversations: &[ConversationData]) -> ValidationResult;
    fn name(&self) -> &str;
}

// ============================================================================
// Format Validator
// ============================================================================

pub struct FormatValidator {
    min_messages: usize,
    max_messages: usize,
    required_roles: HashSet<String>,
}

impl FormatValidator {
    pub fn new() -> Self {
        let mut required_roles = HashSet::new();
        required_roles.insert("user".to_string());
        required_roles.insert("assistant".to_string());

        Self {
            min_messages: 2,
            max_messages: 1000,
            required_roles,
        }
    }
}

impl Validator for FormatValidator {
    fn validate(&self, conversations: &[ConversationData]) -> ValidationResult {
        let mut issues = Vec::new();
        let mut valid_count = 0;

        for (i, conversation) in conversations.iter().enumerate() {
            // Check if empty
            if conversation.messages.is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::Empty,
                    message: "Conversation is empty".to_string(),
                    conversation_index: Some(i),
                    message_index: None,
                });
                continue;
            }

            // Check message count
            if conversation.len() < self.min_messages {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    issue_type: IssueType::TooShort,
                    message: format!("Conversation has only {} messages", conversation.len()),
                    conversation_index: Some(i),
                    message_index: None,
                });
            } else if conversation.len() > self.max_messages {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    issue_type: IssueType::TooLong,
                    message: format!(
                        "Conversation has {} messages, may be truncated",
                        conversation.len()
                    ),
                    conversation_index: Some(i),
                    message_index: None,
                });
            }

            // Check roles
            let roles: HashSet<String> = conversation
                .messages
                .iter()
                .map(|m| m.role.clone())
                .collect();

            for required_role in &self.required_roles {
                if !roles.contains(required_role) {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::MissingRole,
                        message: format!("Missing '{}' role in conversation", required_role),
                        conversation_index: Some(i),
                        message_index: None,
                    });
                }
            }

            // Check for alternating roles
            let mut prev_role = "";
            for (j, message) in conversation.messages.iter().enumerate() {
                if message.role == prev_role && message.role != "system" {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Info,
                        issue_type: IssueType::InvalidFormat,
                        message: format!("Consecutive messages from same role: {}", message.role),
                        conversation_index: Some(i),
                        message_index: Some(j),
                    });
                }
                prev_role = &message.role;
            }

            if issues
                .iter()
                .filter(|issue| {
                    issue.conversation_index == Some(i) && issue.severity == IssueSeverity::Error
                })
                .count()
                == 0
            {
                valid_count += 1;
            }
        }

        let quality_score = valid_count as f32 / conversations.len().max(1) as f32;

        let mut suggestions = Vec::new();
        if issues.iter().any(|i| i.issue_type == IssueType::Empty) {
            suggestions.push("Remove empty conversations before analysis".to_string());
        }
        if issues.iter().any(|i| i.issue_type == IssueType::TooShort) {
            suggestions.push("Consider filtering out very short conversations".to_string());
        }

        ValidationResult {
            is_valid: issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Error)
                .count()
                == 0,
            issues,
            suggestions,
            quality_score,
        }
    }

    fn name(&self) -> &str {
        "FormatValidator"
    }
}

// ============================================================================
// Duplicate Detector
// ============================================================================

pub struct DuplicateDetector {
    similarity_threshold: f32,
}

impl DuplicateDetector {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.95,
        }
    }

    fn hash_conversation(conversation: &ConversationData) -> String {
        let mut hasher = Sha256::new();
        for message in &conversation.messages {
            hasher.update(message.role.as_bytes());
            hasher.update(message.content.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn calculate_similarity(conv1: &ConversationData, conv2: &ConversationData) -> f32 {
        if conv1.len() != conv2.len() {
            return 0.0;
        }

        let mut matching = 0;
        let mut total = 0;

        for (m1, m2) in conv1.messages.iter().zip(conv2.messages.iter()) {
            total += 1;
            if m1.role == m2.role && m1.content == m2.content {
                matching += 1;
            }
        }

        matching as f32 / total.max(1) as f32
    }
}

impl Validator for DuplicateDetector {
    fn validate(&self, conversations: &[ConversationData]) -> ValidationResult {
        let mut issues = Vec::new();
        let mut seen_hashes = HashSet::new();
        let mut duplicates = Vec::new();

        // Check for exact duplicates via hash
        for (i, conversation) in conversations.iter().enumerate() {
            let hash = Self::hash_conversation(conversation);
            if seen_hashes.contains(&hash) {
                duplicates.push(i);
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    issue_type: IssueType::Duplicate,
                    message: "Exact duplicate conversation found".to_string(),
                    conversation_index: Some(i),
                    message_index: None,
                });
            } else {
                seen_hashes.insert(hash);
            }
        }

        // Check for near-duplicates (more expensive)
        if conversations.len() < 1000 {
            // Only for smaller datasets
            for i in 0..conversations.len() {
                if duplicates.contains(&i) {
                    continue;
                }
                for j in (i + 1)..conversations.len() {
                    if duplicates.contains(&j) {
                        continue;
                    }
                    let similarity =
                        Self::calculate_similarity(&conversations[i], &conversations[j]);
                    if similarity >= self.similarity_threshold {
                        issues.push(ValidationIssue {
                            severity: IssueSeverity::Info,
                            issue_type: IssueType::Duplicate,
                            message: format!(
                                "Near-duplicate found ({}% similar)",
                                (similarity * 100.0) as i32
                            ),
                            conversation_index: Some(j),
                            message_index: None,
                        });
                    }
                }
            }
        }

        let duplicate_count = duplicates.len();
        let quality_score = 1.0 - (duplicate_count as f32 / conversations.len().max(1) as f32);

        let mut suggestions = Vec::new();
        if duplicate_count > 0 {
            suggestions.push(format!(
                "Found {} duplicate conversations. Consider deduplication.",
                duplicate_count
            ));
        }

        ValidationResult {
            is_valid: true, // Duplicates are warnings, not errors
            issues,
            suggestions,
            quality_score,
        }
    }

    fn name(&self) -> &str {
        "DuplicateDetector"
    }
}

// ============================================================================
// PII Detector
// ============================================================================

static EMAIL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap());

static PHONE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b").unwrap());

static SSN_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap());

static CREDIT_CARD_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap());

pub struct PIIDetector {
    check_emails: bool,
    check_phones: bool,
    check_ssn: bool,
    check_credit_cards: bool,
}

impl PIIDetector {
    pub fn new() -> Self {
        Self {
            check_emails: true,
            check_phones: true,
            check_ssn: true,
            check_credit_cards: true,
        }
    }

    fn check_text_for_pii(&self, text: &str) -> Vec<String> {
        let mut found = Vec::new();

        if self.check_emails && EMAIL_REGEX.is_match(text) {
            found.push("email address".to_string());
        }

        if self.check_phones && PHONE_REGEX.is_match(text) {
            found.push("phone number".to_string());
        }

        if self.check_ssn && SSN_REGEX.is_match(text) {
            found.push("SSN".to_string());
        }

        if self.check_credit_cards && CREDIT_CARD_REGEX.is_match(text) {
            found.push("credit card number".to_string());
        }

        found
    }
}

impl Validator for PIIDetector {
    fn validate(&self, conversations: &[ConversationData]) -> ValidationResult {
        let mut issues = Vec::new();
        let mut pii_count = 0;

        for (i, conversation) in conversations.iter().enumerate() {
            for (j, message) in conversation.messages.iter().enumerate() {
                let pii_found = self.check_text_for_pii(&message.content);

                if !pii_found.is_empty() {
                    pii_count += 1;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::PII,
                        message: format!("Potential PII detected: {}", pii_found.join(", ")),
                        conversation_index: Some(i),
                        message_index: Some(j),
                    });
                }
            }
        }

        let quality_score = if pii_count > 0 { 0.8 } else { 1.0 };

        let mut suggestions = Vec::new();
        if pii_count > 0 {
            suggestions.push("Consider redacting or removing PII before analysis".to_string());
            suggestions
                .push("You can use automated PII removal tools or manual review".to_string());
        }

        ValidationResult {
            is_valid: true, // PII is a warning, not an error
            issues,
            suggestions,
            quality_score,
        }
    }

    fn name(&self) -> &str {
        "PIIDetector"
    }
}

// ============================================================================
// Content Quality Validator
// ============================================================================

pub struct ContentQualityValidator {
    min_content_length: usize,
    max_content_length: usize,
    min_word_count: usize,
}

impl ContentQualityValidator {
    pub fn new() -> Self {
        Self {
            min_content_length: 10,
            max_content_length: 10000,
            min_word_count: 3,
        }
    }

    fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn check_truncation(text: &str) -> bool {
        text.ends_with("...")
            || text.ends_with("…")
            || text.contains("[truncated]")
            || text.contains("[cut off]")
    }
}

impl Validator for ContentQualityValidator {
    fn validate(&self, conversations: &[ConversationData]) -> ValidationResult {
        let mut issues = Vec::new();
        let mut low_quality_count = 0;
        let mut truncated_count = 0;

        for (i, conversation) in conversations.iter().enumerate() {
            for (j, message) in conversation.messages.iter().enumerate() {
                let content_len = message.content.len();
                let word_count = Self::count_words(&message.content);

                // Check content length
                if content_len < self.min_content_length {
                    low_quality_count += 1;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Info,
                        issue_type: IssueType::LowQuality,
                        message: format!("Very short message ({} chars)", content_len),
                        conversation_index: Some(i),
                        message_index: Some(j),
                    });
                } else if content_len > self.max_content_length {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::TooLong,
                        message: format!(
                            "Very long message ({} chars), may affect processing",
                            content_len
                        ),
                        conversation_index: Some(i),
                        message_index: Some(j),
                    });
                }

                // Check word count
                if word_count < self.min_word_count && content_len > 0 {
                    low_quality_count += 1;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Info,
                        issue_type: IssueType::LowQuality,
                        message: format!("Message has only {} words", word_count),
                        conversation_index: Some(i),
                        message_index: Some(j),
                    });
                }

                // Check for truncation
                if Self::check_truncation(&message.content) {
                    truncated_count += 1;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::Truncated,
                        message: "Message appears to be truncated".to_string(),
                        conversation_index: Some(i),
                        message_index: Some(j),
                    });
                }
            }
        }

        let total_messages: usize = conversations.iter().map(|c| c.len()).sum();
        let quality_score =
            1.0 - ((low_quality_count + truncated_count) as f32 / total_messages.max(1) as f32);

        let mut suggestions = Vec::new();
        if low_quality_count > 10 {
            suggestions.push(
                "Many low-quality messages detected. Consider filtering or improving data quality."
                    .to_string(),
            );
        }
        if truncated_count > 0 {
            suggestions.push(
                "Some messages appear truncated. Check your data export process.".to_string(),
            );
        }

        ValidationResult {
            is_valid: true,
            issues,
            suggestions,
            quality_score,
        }
    }

    fn name(&self) -> &str {
        "ContentQualityValidator"
    }
}

// ============================================================================
// Composite Validator
// ============================================================================

pub struct CompositeValidator {
    validators: Vec<Box<dyn Validator>>,
}

impl CompositeValidator {
    pub fn new() -> Self {
        Self {
            validators: vec![
                Box::new(FormatValidator::new()),
                Box::new(DuplicateDetector::new()),
                Box::new(PIIDetector::new()),
                Box::new(ContentQualityValidator::new()),
            ],
        }
    }

    pub fn add_validator(&mut self, validator: Box<dyn Validator>) {
        self.validators.push(validator);
    }

    pub fn validate_all(&self, conversations: &[ConversationData]) -> Vec<ValidationResult> {
        self.validators
            .iter()
            .map(|v| v.validate(conversations))
            .collect()
    }

    pub fn aggregate_results(&self, results: Vec<ValidationResult>) -> ValidationResult {
        let mut all_issues = Vec::new();
        let mut all_suggestions = HashSet::new();
        let mut total_score = 0.0;
        let mut is_valid = true;

        for result in results {
            all_issues.extend(result.issues);
            for suggestion in result.suggestions {
                all_suggestions.insert(suggestion);
            }
            total_score += result.quality_score;
            is_valid = is_valid && result.is_valid;
        }

        let quality_score = total_score / self.validators.len().max(1) as f32;

        ValidationResult {
            is_valid,
            issues: all_issues,
            suggestions: all_suggestions.into_iter().collect(),
            quality_score,
        }
    }
}
