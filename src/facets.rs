use anyhow::Result;
use tracing::info;

use crate::config::BriefXAIConfig;
use crate::llm::LlmClient;
use crate::prompts::get_facet_prompt;
use crate::types::{ConversationData, Facet, FacetValue};
use crate::utils::most_common;

pub async fn extract_facets(
    config: &BriefXAIConfig,
    data: &[ConversationData],
) -> Result<Vec<Vec<FacetValue>>> {
    info!("Extracting facets from {} conversations", data.len());

    let llm_client = LlmClient::new(config.clone()).await?;
    let facets = crate::types::get_main_facets();

    let mut all_facet_values = Vec::new();

    for conversation in data {
        let mut conversation_facets = Vec::new();

        for facet in &facets {
            let facet_value =
                extract_single_facet(&llm_client, conversation, facet, config).await?;

            conversation_facets.push(facet_value);
        }

        all_facet_values.push(conversation_facets);
    }

    info!("Extracted facets for all conversations");
    Ok(all_facet_values)
}

async fn extract_single_facet(
    llm_client: &LlmClient,
    conversation: &ConversationData,
    facet: &Facet,
    _config: &BriefXAIConfig,
) -> Result<FacetValue> {
    let prompt = get_facet_prompt(conversation, facet);

    // Sample multiple times if configured
    let n_samples = if facet.numeric.is_some() { 1 } else { 3 };
    let mut samples = Vec::new();

    for _ in 0..n_samples {
        let response = llm_client.complete(&prompt).await?;
        let value = process_facet_response(&response, facet)?;
        samples.push(value);
    }

    // Take most common response
    let final_value = most_common(samples).unwrap_or_default();

    Ok(FacetValue {
        facet: facet.clone(),
        value: final_value,
    })
}

pub fn process_facet_response(response: &str, facet: &Facet) -> Result<String> {
    let cleaned = response.trim();

    // Handle numeric facets
    if let Some((min, max)) = facet.numeric {
        if let Ok(num) = cleaned.parse::<i32>() {
            let clamped = num.clamp(min, max);
            return Ok(clamped.to_string());
        }
        // Try to extract number from text
        if let Some(num_str) = extract_number(cleaned) {
            if let Ok(num) = num_str.parse::<i32>() {
                let clamped = num.clamp(min, max);
                return Ok(clamped.to_string());
            }
        }
        // Default to middle value if parsing fails
        return Ok(((min + max) / 2).to_string());
    }

    // Handle prefilled responses
    if !facet.prefill.is_empty() && cleaned.starts_with(&facet.prefill) {
        return Ok(cleaned[facet.prefill.len()..].trim().to_string());
    }

    Ok(cleaned.to_string())
}

fn extract_number(text: &str) -> Option<&str> {
    // Simple regex-like extraction for numbers
    let chars: Vec<char> = text.chars().collect();
    let mut start = None;
    let mut end = None;

    for (i, ch) in chars.iter().enumerate() {
        if ch.is_ascii_digit() {
            if start.is_none() {
                start = Some(i);
            }
            end = Some(i + 1);
        } else if start.is_some() {
            break;
        }
    }

    if let (Some(s), Some(e)) = (start, end) {
        Some(&text[s..e])
    } else {
        None
    }
}

pub async fn extract_facets_batch(
    config: &BriefXAIConfig,
    data: &[ConversationData],
    facets: &[Facet],
) -> Result<Vec<Vec<FacetValue>>> {
    info!(
        "Batch extracting {} facets from {} conversations",
        facets.len(),
        data.len()
    );

    let llm_client = LlmClient::new(config.clone()).await?;

    // Process in batches for efficiency
    let mut results = Vec::new();
    for conversation in data {
        let mut facet_values = Vec::new();

        for facet in facets {
            let value = extract_single_facet(&llm_client, conversation, facet, config).await?;
            facet_values.push(value);
        }

        results.push(facet_values);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ConversationData, Message};
    use std::collections::HashMap;

    fn create_test_conversation() -> ConversationData {
        ConversationData {
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "I'm having trouble with my account".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "I'd be happy to help you with your account issue.".to_string(),
                },
            ],
            metadata: HashMap::new(),
        }
    }

    fn create_test_config() -> BriefXAIConfig {
        BriefXAIConfig {
            llm_provider: crate::config::LlmProvider::OpenAI,
            llm_model: "gpt-4o-mini".to_string(),
            llm_api_key: Some("test-key".to_string()),
            verbose: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("The answer is 42"), Some("42"));
        assert_eq!(extract_number("Score: 5/10"), Some("5"));
        assert_eq!(extract_number("No numbers here"), None);
        assert_eq!(extract_number("123abc456"), Some("123"));
        assert_eq!(extract_number("Price is $99.50"), Some("99"));
        assert_eq!(extract_number("Call 555-1234"), Some("555"));
        assert_eq!(extract_number(""), None);
    }

    #[test]
    fn test_extract_number_edge_cases() {
        assert_eq!(extract_number("0"), Some("0"));
        assert_eq!(extract_number("00123"), Some("00123"));
        assert_eq!(extract_number("abc123def456"), Some("123"));
        assert_eq!(extract_number("!@#$%^&*()"), None);
        assert_eq!(extract_number("3.14159"), Some("3"));
    }

    #[test]
    fn test_process_facet_response_numeric() {
        let facet = Facet {
            name: "Score".to_string(),
            question: "Rate this".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((1, 5)),
        };

        assert_eq!(process_facet_response("3", &facet).unwrap(), "3");
        assert_eq!(
            process_facet_response("The score is 4", &facet).unwrap(),
            "4"
        );
        assert_eq!(
            process_facet_response("10", &facet).unwrap(),
            "5" // Clamped to max
        );
        assert_eq!(
            process_facet_response("0", &facet).unwrap(),
            "1" // Clamped to min
        );
        assert_eq!(
            process_facet_response("invalid", &facet).unwrap(),
            "3" // Default to middle value
        );
    }

    #[test]
    fn test_process_facet_response_numeric_edge_cases() {
        let facet = Facet {
            name: "Binary".to_string(),
            question: "Yes or no?".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((0, 1)),
        };

        assert_eq!(process_facet_response("0", &facet).unwrap(), "0");
        assert_eq!(process_facet_response("1", &facet).unwrap(), "1");
        assert_eq!(
            process_facet_response("2", &facet).unwrap(),
            "1" // Clamped to max
        );
        assert_eq!(
            process_facet_response("-1", &facet).unwrap(),
            "0" // Clamped to min
        );
    }

    #[test]
    fn test_process_facet_response_text() {
        let facet = Facet {
            name: "Sentiment".to_string(),
            question: "What is the sentiment?".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: None,
        };

        assert_eq!(
            process_facet_response("positive", &facet).unwrap(),
            "positive"
        );
        assert_eq!(
            process_facet_response("  negative  ", &facet).unwrap(),
            "negative"
        );
        assert_eq!(process_facet_response("", &facet).unwrap(), "");
    }

    #[test]
    fn test_process_facet_response_prefill() {
        let facet = Facet {
            name: "Category".to_string(),
            question: "What category?".to_string(),
            prefill: "Category: ".to_string(),
            summary_criteria: None,
            numeric: None,
        };

        assert_eq!(
            process_facet_response("Category: Support", &facet).unwrap(),
            "Support"
        );
        assert_eq!(
            process_facet_response("Category:  Billing  ", &facet).unwrap(),
            "Billing"
        );
        assert_eq!(
            process_facet_response("Different format", &facet).unwrap(),
            "Different format"
        );
    }

    #[test]
    fn test_process_facet_response_complex_numeric() {
        let facet = Facet {
            name: "Satisfaction".to_string(),
            question: "Rate satisfaction 1-10".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((1, 10)),
        };

        assert_eq!(
            process_facet_response("I would rate this a 7 out of 10", &facet).unwrap(),
            "7"
        );
        assert_eq!(
            process_facet_response("Rating: 9/10 - excellent!", &facet).unwrap(),
            "9"
        );
        assert_eq!(
            process_facet_response("Completely unsatisfied", &facet).unwrap(),
            "5" // Default to middle
        );
    }

    #[test]
    fn test_process_facet_response_boundary_conditions() {
        let facet = Facet {
            name: "Range".to_string(),
            question: "Pick a number".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((10, 20)),
        };

        // Test exact boundaries
        assert_eq!(process_facet_response("10", &facet).unwrap(), "10");
        assert_eq!(process_facet_response("20", &facet).unwrap(), "20");

        // Test beyond boundaries
        assert_eq!(process_facet_response("5", &facet).unwrap(), "10");
        assert_eq!(process_facet_response("25", &facet).unwrap(), "20");

        // Test default
        assert_eq!(
            process_facet_response("no number", &facet).unwrap(),
            "15" // (10 + 20) / 2
        );
    }

    #[test]
    fn test_numeric_clamping_edge_cases() {
        // Single value range
        let single_range_facet = Facet {
            name: "Single".to_string(),
            question: "Only one option".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((5, 5)),
        };

        assert_eq!(
            process_facet_response("3", &single_range_facet).unwrap(),
            "5"
        );
        assert_eq!(
            process_facet_response("7", &single_range_facet).unwrap(),
            "5"
        );
        assert_eq!(
            process_facet_response("5", &single_range_facet).unwrap(),
            "5"
        );
    }

    #[test]
    fn test_process_facet_response_whitespace_handling() {
        let facet = Facet {
            name: "Test".to_string(),
            question: "Test question".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: None,
        };

        assert_eq!(
            process_facet_response("   whitespace   ", &facet).unwrap(),
            "whitespace"
        );
        assert_eq!(
            process_facet_response("\n\tlined\n\t", &facet).unwrap(),
            "lined"
        );
        assert_eq!(process_facet_response("", &facet).unwrap(), "");
    }

    #[test]
    fn test_facet_value_creation() {
        let facet = Facet {
            name: "test_facet".to_string(),
            question: "Test question".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: None,
        };

        let facet_value = FacetValue {
            facet: facet.clone(),
            value: "test_value".to_string(),
        };

        assert_eq!(facet_value.facet.name, "test_facet");
        assert_eq!(facet_value.value, "test_value");
    }

    #[test]
    fn test_main_facets_available() {
        let facets = crate::types::get_main_facets();
        assert!(!facets.is_empty());

        // Check that we have some expected facets
        let facet_names: Vec<String> = facets.iter().map(|f| f.name.clone()).collect();
        assert!(facet_names.len() > 0);

        // Verify facet structure
        for facet in &facets {
            assert!(!facet.name.is_empty());
            assert!(!facet.question.is_empty());
        }
    }

    #[test]
    fn test_number_extraction_comprehensive() {
        let test_cases = vec![
            ("123", Some("123")),
            ("abc123", Some("123")),
            ("123abc", Some("123")),
            ("12.34", Some("12")),
            ("no digits", None),
            ("multiple 123 numbers 456", Some("123")),
            ("0", Some("0")),
            ("000", Some("000")),
            ("1a2b3", Some("1")),
            ("!@#123$%^", Some("123")),
        ];

        for (input, expected) in test_cases {
            assert_eq!(
                extract_number(input),
                expected,
                "Failed for input: {}",
                input
            );
        }
    }

    #[test]
    fn test_facet_processing_consistency() {
        let facet = Facet {
            name: "Consistency".to_string(),
            question: "Rate consistency".to_string(),
            prefill: String::new(),
            summary_criteria: None,
            numeric: Some((1, 5)),
        };

        // Same input should produce same output
        let input = "The rating is 3";
        let result1 = process_facet_response(input, &facet).unwrap();
        let result2 = process_facet_response(input, &facet).unwrap();

        assert_eq!(result1, result2);
    }
}
