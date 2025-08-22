use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::ConversationData;

// ============================================================================
// Language Detection Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedLanguage {
    pub language: String,
    pub confidence: f32,
    pub script: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    pub primary_language: String,
    pub languages: HashMap<String, usize>,
    pub mixed_language_conversations: Vec<usize>,
    pub confidence: f32,
}

// ============================================================================
// Simple Language Detector
// ============================================================================

pub struct LanguageDetector {
    min_text_length: usize,
    confidence_threshold: f32,
}

impl LanguageDetector {
    pub fn new() -> Self {
        Self {
            min_text_length: 20,
            confidence_threshold: 0.7,
        }
    }

    pub fn detect(&self, text: &str) -> DetectedLanguage {
        // Simple heuristic-based detection
        // In production, you'd use a proper library like whatlang or lingua

        let text_lower = text.to_lowercase();

        // Check for common English patterns
        let english_indicators = [
            "the", "is", "are", "was", "were", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "shall", "to", "of",
            "in", "for", "on", "with", "at", "from", "by", "about", "into", "through", "during",
            "before", "after",
        ];

        let english_count = english_indicators
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();

        // Check for Chinese characters
        let chinese_count = text
            .chars()
            .filter(|c| {
                (*c >= '\u{4E00}' && *c <= '\u{9FFF}') || (*c >= '\u{3400}' && *c <= '\u{4DBF}')
            })
            .count();

        // Check for Japanese characters (Hiragana, Katakana)
        let japanese_count = text
            .chars()
            .filter(|c| {
                (*c >= '\u{3040}' && *c <= '\u{309F}') || (*c >= '\u{30A0}' && *c <= '\u{30FF}')
            })
            .count();

        // Check for Korean characters
        let korean_count = text
            .chars()
            .filter(|c| *c >= '\u{AC00}' && *c <= '\u{D7AF}')
            .count();

        // Check for Arabic script
        let arabic_count = text
            .chars()
            .filter(|c| *c >= '\u{0600}' && *c <= '\u{06FF}')
            .count();

        // Check for Cyrillic script (Russian, etc.)
        let cyrillic_count = text
            .chars()
            .filter(|c| *c >= '\u{0400}' && *c <= '\u{04FF}')
            .count();

        // Check for Spanish/Portuguese indicators
        let spanish_indicators = [
            "el", "la", "de", "que", "es", "en", "un", "por", "con", "para",
        ];
        let spanish_count = spanish_indicators
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();

        // Check for French indicators
        let french_indicators = [
            "le", "de", "un", "être", "et", "à", "il", "avoir", "ne", "je",
        ];
        let french_count = french_indicators
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();

        // Check for German indicators
        let german_indicators = [
            "der", "die", "das", "ist", "ich", "nicht", "ein", "zu", "haben", "werden",
        ];
        let german_count = german_indicators
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();

        // Determine language based on counts
        let text_len = text.len().max(1);

        if chinese_count > 0 && chinese_count as f32 / text_len as f32 > 0.1 {
            DetectedLanguage {
                language: "zh".to_string(),
                confidence: (chinese_count as f32 / text_len as f32).min(1.0),
                script: Some("Chinese".to_string()),
            }
        } else if japanese_count > 0 {
            DetectedLanguage {
                language: "ja".to_string(),
                confidence: (japanese_count as f32 / text_len as f32 * 10.0).min(1.0),
                script: Some("Japanese".to_string()),
            }
        } else if korean_count > 0 {
            DetectedLanguage {
                language: "ko".to_string(),
                confidence: (korean_count as f32 / text_len as f32 * 10.0).min(1.0),
                script: Some("Korean".to_string()),
            }
        } else if arabic_count > 0 {
            DetectedLanguage {
                language: "ar".to_string(),
                confidence: (arabic_count as f32 / text_len as f32 * 5.0).min(1.0),
                script: Some("Arabic".to_string()),
            }
        } else if cyrillic_count > 0 {
            DetectedLanguage {
                language: "ru".to_string(),
                confidence: (cyrillic_count as f32 / text_len as f32 * 5.0).min(1.0),
                script: Some("Cyrillic".to_string()),
            }
        } else if german_count >= 3 && german_count > spanish_count && german_count > french_count {
            DetectedLanguage {
                language: "de".to_string(),
                confidence: (german_count as f32 / 10.0).min(0.9),
                script: Some("Latin".to_string()),
            }
        } else if french_count >= 3 && french_count > spanish_count {
            DetectedLanguage {
                language: "fr".to_string(),
                confidence: (french_count as f32 / 10.0).min(0.9),
                script: Some("Latin".to_string()),
            }
        } else if spanish_count >= 3 && spanish_count > english_count / 2 {
            DetectedLanguage {
                language: "es".to_string(),
                confidence: (spanish_count as f32 / 10.0).min(0.9),
                script: Some("Latin".to_string()),
            }
        } else if english_count >= 5 {
            DetectedLanguage {
                language: "en".to_string(),
                confidence: (english_count as f32 / 28.0).min(0.95),
                script: Some("Latin".to_string()),
            }
        } else {
            DetectedLanguage {
                language: "unknown".to_string(),
                confidence: 0.0,
                script: None,
            }
        }
    }

    pub fn detect_conversation(&self, conversation: &ConversationData) -> DetectedLanguage {
        let mut language_counts: HashMap<String, usize> = HashMap::new();
        let mut total_confidence = 0.0;
        let mut count = 0;

        for message in &conversation.messages {
            if message.content.len() >= self.min_text_length {
                let detected = self.detect(&message.content);
                if detected.confidence >= self.confidence_threshold {
                    *language_counts
                        .entry(detected.language.clone())
                        .or_insert(0) += 1;
                    total_confidence += detected.confidence;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return DetectedLanguage {
                language: "unknown".to_string(),
                confidence: 0.0,
                script: None,
            };
        }

        // Find the most common language
        let primary_language = language_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, _)| lang.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Check if mixed languages
        let is_mixed = language_counts.len() > 1;
        let confidence = if is_mixed {
            (total_confidence / count as f32) * 0.8 // Reduce confidence for mixed
        } else {
            total_confidence / count as f32
        };

        DetectedLanguage {
            language: primary_language.clone(),
            confidence,
            script: self.get_script_for_language(&primary_language),
        }
    }

    pub fn analyze_dataset(&self, conversations: &[ConversationData]) -> LanguageStats {
        let mut language_counts: HashMap<String, usize> = HashMap::new();
        let mut mixed_language_conversations = Vec::new();
        let mut total_confidence = 0.0;

        for (i, conversation) in conversations.iter().enumerate() {
            let detected = self.detect_conversation(conversation);

            // Always count the language, even if unknown or low confidence
            let lang = if detected.language != "unknown" {
                detected.language.clone()
            } else {
                "en".to_string() // Default to English if unknown
            };

            *language_counts.entry(lang).or_insert(0) += 1;
            total_confidence += detected.confidence.max(0.1); // Minimum confidence

            // Check if conversation has mixed languages
            let mut message_languages = HashMap::new();
            for message in &conversation.messages {
                if message.content.len() >= self.min_text_length {
                    let msg_lang = self.detect(&message.content);
                    if msg_lang.confidence >= self.confidence_threshold {
                        *message_languages.entry(msg_lang.language).or_insert(0) += 1;
                    }
                }
            }

            if message_languages.len() > 1 {
                mixed_language_conversations.push(i);
            }
        }

        let primary_language = language_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, _)| lang.clone())
            .unwrap_or_else(|| "en".to_string());

        let confidence = total_confidence / conversations.len().max(1) as f32;

        LanguageStats {
            primary_language,
            languages: language_counts,
            mixed_language_conversations,
            confidence,
        }
    }

    fn get_script_for_language(&self, language: &str) -> Option<String> {
        match language {
            "zh" => Some("Chinese".to_string()),
            "ja" => Some("Japanese".to_string()),
            "ko" => Some("Korean".to_string()),
            "ar" => Some("Arabic".to_string()),
            "ru" | "uk" | "bg" => Some("Cyrillic".to_string()),
            "hi" | "mr" | "ne" => Some("Devanagari".to_string()),
            "th" => Some("Thai".to_string()),
            "he" => Some("Hebrew".to_string()),
            "en" | "es" | "fr" | "de" | "it" | "pt" | "nl" | "sv" | "no" | "da" | "fi" => {
                Some("Latin".to_string())
            }
            _ => None,
        }
    }
}

// ============================================================================
// Language Normalizer
// ============================================================================

pub struct LanguageNormalizer;

impl LanguageNormalizer {
    pub fn normalize_language_code(code: &str) -> String {
        // Convert to ISO 639-1 codes
        match code.to_lowercase().as_str() {
            "english" | "eng" => "en",
            "spanish" | "spa" | "español" => "es",
            "french" | "fra" | "français" => "fr",
            "german" | "deu" | "deutsch" => "de",
            "chinese" | "chi" | "zho" | "中文" => "zh",
            "japanese" | "jpn" | "日本語" => "ja",
            "korean" | "kor" | "한국어" => "ko",
            "russian" | "rus" | "русский" => "ru",
            "arabic" | "ara" | "العربية" => "ar",
            "portuguese" | "por" | "português" => "pt",
            "italian" | "ita" | "italiano" => "it",
            "dutch" | "nld" | "nederlands" => "nl",
            _ => code,
        }
        .to_string()
    }

    pub fn get_language_name(code: &str) -> String {
        match code {
            "en" => "English",
            "es" => "Spanish",
            "fr" => "French",
            "de" => "German",
            "zh" => "Chinese",
            "ja" => "Japanese",
            "ko" => "Korean",
            "ru" => "Russian",
            "ar" => "Arabic",
            "pt" => "Portuguese",
            "it" => "Italian",
            "nl" => "Dutch",
            _ => code,
        }
        .to_string()
    }
}
