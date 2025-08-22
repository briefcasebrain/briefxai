//! Error types for BriefXAI
//!
//! This module defines the error types used throughout the application.

use std::fmt;
use thiserror::Error;

/// Main error type for BriefXAI operations
#[derive(Error, Debug)]
pub enum BriefXAIError {
    /// Configuration-related errors
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// LLM provider errors
    #[error("LLM provider error: {0}")]
    LlmProvider(String),
    
    /// Embedding generation errors
    #[error("Embedding error: {0}")]
    Embedding(String),
    
    /// Data validation errors
    #[error("Validation error: {0}")]
    Validation(String),
    
    /// Clustering errors
    #[error("Clustering error: {0}")]
    Clustering(String),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Database errors
    #[error("Database error: {0}")]
    Database(String),
    
    /// Network errors
    #[error("Network error: {0}")]
    Network(String),
    
    /// Session management errors
    #[error("Session error: {0}")]
    Session(String),
    
    /// Rate limiting errors
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),
    
    /// API key errors
    #[error("API key error: {0}")]
    ApiKey(String),
    
    /// Generic errors
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for BriefXAIError {
    fn from(err: anyhow::Error) -> Self {
        BriefXAIError::Other(err.to_string())
    }
}

impl From<reqwest::Error> for BriefXAIError {
    fn from(err: reqwest::Error) -> Self {
        BriefXAIError::Network(err.to_string())
    }
}

/// Result type alias for BriefXAI operations
pub type Result<T> = std::result::Result<T, BriefXAIError>;

/// Session-specific errors
#[derive(Error, Debug)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    NotFound(String),
    
    #[error("Session already exists: {0}")]
    AlreadyExists(String),
    
    #[error("Session is locked")]
    Locked,
    
    #[error("Session expired")]
    Expired,
}

/// Validation errors with field information
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },
    
    #[error("Invalid format for field {field}: {reason}")]
    InvalidFormat { field: String, reason: String },
    
    #[error("Value out of range for {field}: {value}")]
    OutOfRange { field: String, value: String },
    
    #[error("Data size exceeds limit: {size} > {limit}")]
    SizeLimit { size: usize, limit: usize },
}