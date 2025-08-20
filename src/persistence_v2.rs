use anyhow::{Result, Context, bail};
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions, Row};
use serde::{Serialize, Deserialize};
use serde_json;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;
use tokio::sync::RwLock;
use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use sha2::{Sha256, Digest};

use crate::config::BriefXAIConfig;

// ============================================================================
// Migration System
// ============================================================================

struct Migration {
    version: i32,
    name: String,
    sql: String,
}

impl Migration {
    fn new(version: i32, name: impl Into<String>, sql: impl Into<String>) -> Self {
        Self {
            version,
            name: name.into(),
            sql: sql.into(),
        }
    }
}

async fn get_migrations() -> Vec<Migration> {
    vec![
        Migration::new(
            1,
            "initial_schema",
            include_str!("../migrations/001_initial_schema.sql")
        ),
    ]
}

// ============================================================================
// Domain Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSession {
    pub id: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub status: SessionStatus,
    pub config: BriefXAIConfig,
    pub current_batch: i32,
    pub total_batches: Option<i32>,
    pub total_conversations: Option<i32>,
    pub processed_conversations: i32,
    pub error_message: Option<String>,
    pub results: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
}

impl ToString for SessionStatus {
    fn to_string(&self) -> String {
        match self {
            SessionStatus::Pending => "pending".to_string(),
            SessionStatus::Running => "running".to_string(),
            SessionStatus::Paused => "paused".to_string(),
            SessionStatus::Completed => "completed".to_string(),
            SessionStatus::Failed => "failed".to_string(),
        }
    }
}

impl From<String> for SessionStatus {
    fn from(s: String) -> Self {
        match s.as_str() {
            "pending" => SessionStatus::Pending,
            "running" => SessionStatus::Running,
            "paused" => SessionStatus::Paused,
            "completed" => SessionStatus::Completed,
            "failed" => SessionStatus::Failed,
            _ => SessionStatus::Pending,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgress {
    pub id: i32,
    pub session_id: String,
    pub batch_number: i32,
    pub status: BatchStatus,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub input_data: Option<serde_json::Value>,
    pub output_data: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub retry_count: i32,
    pub processing_time_ms: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BatchStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

impl ToString for BatchStatus {
    fn to_string(&self) -> String {
        match self {
            BatchStatus::Pending => "pending".to_string(),
            BatchStatus::Processing => "processing".to_string(),
            BatchStatus::Completed => "completed".to_string(),
            BatchStatus::Failed => "failed".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisTemplate {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: TemplateCategory,
    pub is_public: bool,
    pub config: serde_json::Value,
    pub custom_prompts: Option<serde_json::Value>,
    pub facet_definitions: Option<serde_json::Value>,
    pub created_at: i64,
    pub updated_at: i64,
    pub created_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TemplateCategory {
    General,
    Support,
    Sales,
    Medical,
    Education,
    Custom,
}

impl ToString for TemplateCategory {
    fn to_string(&self) -> String {
        match self {
            TemplateCategory::General => "general".to_string(),
            TemplateCategory::Support => "support".to_string(),
            TemplateCategory::Sales => "sales".to_string(),
            TemplateCategory::Medical => "medical".to_string(),
            TemplateCategory::Education => "education".to_string(),
            TemplateCategory::Custom => "custom".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub settings: Option<serde_json::Value>,
    pub tags: Option<Vec<String>>,
    pub is_archived: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub config: serde_json::Value,
    pub priority: i32,
    pub is_active: bool,
    pub is_fallback: bool,
    pub rate_limit: Option<serde_json::Value>,
    pub cost_per_token: Option<serde_json::Value>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    OpenAI,
    Ollama,
    VLLM,
    HuggingFace,
    Alternative,
}

impl ToString for ProviderType {
    fn to_string(&self) -> String {
        match self {
            ProviderType::OpenAI => "openai".to_string(),
            ProviderType::Ollama => "ollama".to_string(),
            ProviderType::VLLM => "vllm".to_string(),
            ProviderType::HuggingFace => "huggingface".to_string(),
            ProviderType::Alternative => "alternative".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialResult {
    pub id: i32,
    pub session_id: String,
    pub batch_number: Option<i32>,
    pub result_type: ResultType,
    pub data: serde_json::Value,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ResultType {
    Facet,
    Cluster,
    Insight,
    Embedding,
}

impl ToString for ResultType {
    fn to_string(&self) -> String {
        match self {
            ResultType::Facet => "facet".to_string(),
            ResultType::Cluster => "cluster".to_string(),
            ResultType::Insight => "insight".to_string(),
            ResultType::Embedding => "embedding".to_string(),
        }
    }
}

// ============================================================================
// Session Manager
// ============================================================================

pub struct SessionManager {
    pool: SqlitePool,
    current_session: Arc<RwLock<Option<AnalysisSession>>>,
}

impl SessionManager {
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            current_session: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn create_session(&self, config: BriefXAIConfig) -> Result<AnalysisSession> {
        let session = AnalysisSession {
            id: Uuid::new_v4().to_string(),
            created_at: current_timestamp(),
            updated_at: current_timestamp(),
            status: SessionStatus::Pending,
            config,
            current_batch: 0,
            total_batches: None,
            total_conversations: None,
            processed_conversations: 0,
            error_message: None,
            results: None,
            metadata: None,
        };

        let config_json = serde_json::to_string(&session.config)?;
        
        sqlx::query(r#"
            INSERT INTO analysis_sessions (
                id, created_at, updated_at, status, config, 
                current_batch, processed_conversations
            )
            VALUES (?, datetime('now'), datetime('now'), ?, ?, ?, ?)
        "#)
        .bind(&session.id)
        .bind(session.status.to_string())
        .bind(config_json)
        .bind(session.current_batch)
        .bind(session.processed_conversations)
        .execute(&self.pool)
        .await?;

        *self.current_session.write().await = Some(session.clone());
        Ok(session)
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<AnalysisSession>> {
        let row = sqlx::query(r#"
            SELECT * FROM analysis_sessions WHERE id = ?
        "#)
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let session = AnalysisSession {
                id: row.get("id"),
                created_at: 0, // Will be parsed from timestamp
                updated_at: 0,
                status: SessionStatus::from(row.get::<String, _>("status")),
                config: serde_json::from_str(&row.get::<String, _>("config"))?,
                current_batch: row.get("current_batch"),
                total_batches: row.get("total_batches"),
                total_conversations: row.get("total_conversations"),
                processed_conversations: row.get("processed_conversations"),
                error_message: row.get("error_message"),
                results: row.get::<Option<String>, _>("results")
                    .and_then(|s| serde_json::from_str(&s).ok()),
                metadata: row.get::<Option<String>, _>("metadata")
                    .and_then(|s| serde_json::from_str(&s).ok()),
            };
            Ok(Some(session))
        } else {
            Ok(None)
        }
    }

    pub async fn update_session_status(&self, session_id: &str, status: SessionStatus) -> Result<()> {
        sqlx::query(r#"
            UPDATE analysis_sessions 
            SET status = ?, updated_at = datetime('now')
            WHERE id = ?
        "#)
        .bind(status.to_string())
        .bind(session_id)
        .execute(&self.pool)
        .await?;

        if let Some(ref mut session) = *self.current_session.write().await {
            if session.id == session_id {
                session.status = status;
                session.updated_at = current_timestamp();
            }
        }

        Ok(())
    }

    pub async fn pause_session(&self, session_id: &str) -> Result<()> {
        self.update_session_status(session_id, SessionStatus::Paused).await
    }

    pub async fn resume_session(&self, session_id: &str) -> Result<AnalysisSession> {
        let mut session = self.get_session(session_id).await?
            .context("Session not found")?;
        
        if session.status != SessionStatus::Paused {
            bail!("Session is not paused");
        }

        self.update_session_status(session_id, SessionStatus::Running).await?;
        session.status = SessionStatus::Running;
        *self.current_session.write().await = Some(session.clone());
        
        Ok(session)
    }

    pub async fn save_batch_progress(
        &self,
        session_id: &str,
        batch_number: i32,
        status: BatchStatus,
        output_data: Option<serde_json::Value>,
        error_message: Option<String>,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        // Update or insert batch progress
        sqlx::query(r#"
            INSERT INTO batch_progress (
                session_id, batch_number, status, completed_at, 
                output_data, error_message
            )
            VALUES (?, ?, ?, datetime('now'), ?, ?)
            ON CONFLICT(session_id, batch_number) DO UPDATE SET
                status = excluded.status,
                completed_at = excluded.completed_at,
                output_data = excluded.output_data,
                error_message = excluded.error_message,
                retry_count = retry_count + 1
        "#)
        .bind(session_id)
        .bind(batch_number)
        .bind(status.to_string())
        .bind(output_data.map(|d| d.to_string()))
        .bind(error_message)
        .execute(&mut *tx)
        .await?;

        // Update session progress
        if matches!(status, BatchStatus::Completed) {
            sqlx::query(r#"
                UPDATE analysis_sessions 
                SET current_batch = ?, updated_at = datetime('now')
                WHERE id = ?
            "#)
            .bind(batch_number + 1)
            .bind(session_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    pub async fn get_last_successful_batch(&self, session_id: &str) -> Result<Option<i32>> {
        let row = sqlx::query(r#"
            SELECT MAX(batch_number) as last_batch
            FROM batch_progress
            WHERE session_id = ? AND status = 'completed'
        "#)
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(row.get("last_batch"))
        } else {
            Ok(None)
        }
    }

    pub async fn recover_session(&self, session_id: &str) -> Result<AnalysisSession> {
        let mut session = self.get_session(session_id).await?
            .context("Session not found")?;
        
        // Find last successful batch
        if let Some(last_batch) = self.get_last_successful_batch(session_id).await? {
            session.current_batch = last_batch + 1;
            info!("Recovering session {} from batch {}", session_id, session.current_batch);
        } else {
            session.current_batch = 0;
            info!("Starting session {} from beginning", session_id);
        }

        self.update_session_status(session_id, SessionStatus::Running).await?;
        *self.current_session.write().await = Some(session.clone());
        
        Ok(session)
    }
}

// ============================================================================
// Template Repository
// ============================================================================

pub struct TemplateRepository {
    pool: SqlitePool,
}

impl TemplateRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub async fn create_template(&self, template: AnalysisTemplate) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO analysis_templates (
                id, name, description, category, is_public, config,
                custom_prompts, facet_definitions, created_at, updated_at, created_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)
        "#)
        .bind(&template.id)
        .bind(&template.name)
        .bind(&template.description)
        .bind(template.category.to_string())
        .bind(template.is_public)
        .bind(template.config.to_string())
        .bind(template.custom_prompts.map(|p| p.to_string()))
        .bind(template.facet_definitions.map(|f| f.to_string()))
        .bind(&template.created_by)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    pub async fn get_template(&self, template_id: &str) -> Result<Option<AnalysisTemplate>> {
        let row = sqlx::query(r#"
            SELECT * FROM analysis_templates WHERE id = ?
        "#)
        .bind(template_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(self.row_to_template(row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_templates(&self, category: Option<TemplateCategory>) -> Result<Vec<AnalysisTemplate>> {
        let query = if let Some(cat) = category {
            sqlx::query(r#"
                SELECT * FROM analysis_templates 
                WHERE category = ? AND is_public = true
                ORDER BY name
            "#)
            .bind(cat.to_string())
        } else {
            sqlx::query(r#"
                SELECT * FROM analysis_templates 
                WHERE is_public = true
                ORDER BY name
            "#)
        };

        let rows = query.fetch_all(&self.pool).await?;
        
        rows.into_iter()
            .map(|row| self.row_to_template(row))
            .collect()
    }

    fn row_to_template(&self, row: sqlx::sqlite::SqliteRow) -> Result<AnalysisTemplate> {
        Ok(AnalysisTemplate {
            id: row.get("id"),
            name: row.get("name"),
            description: row.get("description"),
            category: match row.get::<String, _>("category").as_str() {
                "general" => TemplateCategory::General,
                "support" => TemplateCategory::Support,
                "sales" => TemplateCategory::Sales,
                "medical" => TemplateCategory::Medical,
                "education" => TemplateCategory::Education,
                _ => TemplateCategory::Custom,
            },
            is_public: row.get("is_public"),
            config: serde_json::from_str(&row.get::<String, _>("config"))?,
            custom_prompts: row.get::<Option<String>, _>("custom_prompts")
                .and_then(|s| serde_json::from_str(&s).ok()),
            facet_definitions: row.get::<Option<String>, _>("facet_definitions")
                .and_then(|s| serde_json::from_str(&s).ok()),
            created_at: 0,
            updated_at: 0,
            created_by: row.get("created_by"),
        })
    }
}

// ============================================================================
// Provider Manager
// ============================================================================

pub struct ProviderManager {
    pool: SqlitePool,
}

impl ProviderManager {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub async fn add_provider(&self, provider: ProviderConfig) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO provider_configs (
                id, name, provider_type, config, priority, is_active,
                is_fallback, rate_limit, cost_per_token, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        "#)
        .bind(&provider.id)
        .bind(&provider.name)
        .bind(provider.provider_type.to_string())
        .bind(provider.config.to_string())
        .bind(provider.priority)
        .bind(provider.is_active)
        .bind(provider.is_fallback)
        .bind(provider.rate_limit.map(|r| r.to_string()))
        .bind(provider.cost_per_token.map(|c| c.to_string()))
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    pub async fn get_active_providers(&self) -> Result<Vec<ProviderConfig>> {
        let rows = sqlx::query(r#"
            SELECT * FROM provider_configs 
            WHERE is_active = true
            ORDER BY priority ASC
        "#)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|row| self.row_to_provider(row))
            .collect()
    }

    pub async fn get_fallback_providers(&self) -> Result<Vec<ProviderConfig>> {
        let rows = sqlx::query(r#"
            SELECT * FROM provider_configs 
            WHERE is_active = true AND is_fallback = true
            ORDER BY priority ASC
        "#)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|row| self.row_to_provider(row))
            .collect()
    }

    pub async fn record_usage(
        &self,
        provider_id: &str,
        session_id: Option<&str>,
        tokens_used: i32,
        cost: Option<f64>,
        success: bool,
        error_message: Option<&str>,
        response_time_ms: i32,
    ) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO provider_usage (
                provider_id, session_id, timestamp, tokens_used, cost,
                success, error_message, response_time_ms
            )
            VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?)
        "#)
        .bind(provider_id)
        .bind(session_id)
        .bind(tokens_used)
        .bind(cost)
        .bind(success)
        .bind(error_message)
        .bind(response_time_ms)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    fn row_to_provider(&self, row: sqlx::sqlite::SqliteRow) -> Result<ProviderConfig> {
        Ok(ProviderConfig {
            id: row.get("id"),
            name: row.get("name"),
            provider_type: match row.get::<String, _>("provider_type").as_str() {
                "openai" => ProviderType::OpenAI,
                "ollama" => ProviderType::Ollama,
                "vllm" => ProviderType::VLLM,
                "huggingface" => ProviderType::HuggingFace,
                "alternative" => ProviderType::Alternative,
                _ => ProviderType::OpenAI,
            },
            config: serde_json::from_str(&row.get::<String, _>("config"))?,
            priority: row.get("priority"),
            is_active: row.get("is_active"),
            is_fallback: row.get("is_fallback"),
            rate_limit: row.get::<Option<String>, _>("rate_limit")
                .and_then(|s| serde_json::from_str(&s).ok()),
            cost_per_token: row.get::<Option<String>, _>("cost_per_token")
                .and_then(|s| serde_json::from_str(&s).ok()),
            created_at: 0,
            updated_at: 0,
        })
    }
}

// ============================================================================
// Enhanced Persistence Layer
// ============================================================================

pub struct EnhancedPersistenceLayer {
    pool: SqlitePool,
    session_manager: Arc<SessionManager>,
    template_repo: Arc<TemplateRepository>,
    provider_manager: Arc<ProviderManager>,
    memory_cache: Arc<DashMap<String, (Vec<u8>, i64)>>,
}

impl EnhancedPersistenceLayer {
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        // Create directory if needed
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        info!("Initializing enhanced persistence at {}", db_path.display());

        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect(&db_url)
            .await?;

        // Run migrations
        Self::run_migrations(&pool).await?;

        let session_manager = Arc::new(SessionManager::new(pool.clone()));
        let template_repo = Arc::new(TemplateRepository::new(pool.clone()));
        let provider_manager = Arc::new(ProviderManager::new(pool.clone()));

        Ok(Self {
            pool,
            session_manager,
            template_repo,
            provider_manager,
            memory_cache: Arc::new(DashMap::new()),
        })
    }

    async fn run_migrations(pool: &SqlitePool) -> Result<()> {
        // Create migrations table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(pool)
        .await?;

        // Get applied migrations
        let applied: Vec<i32> = sqlx::query("SELECT version FROM schema_migrations")
            .fetch_all(pool)
            .await?
            .into_iter()
            .map(|row| row.get("version"))
            .collect();

        // Apply new migrations
        for migration in get_migrations().await {
            if !applied.contains(&migration.version) {
                info!("Applying migration {}: {}", migration.version, migration.name);
                
                let mut tx = pool.begin().await?;
                
                // Execute migration SQL
                sqlx::raw_sql(&migration.sql)
                    .execute(&mut *tx)
                    .await?;
                
                // Record migration
                sqlx::query(r#"
                    INSERT INTO schema_migrations (version, name)
                    VALUES (?, ?)
                "#)
                .bind(migration.version)
                .bind(&migration.name)
                .execute(&mut *tx)
                .await?;
                
                tx.commit().await?;
                
                info!("Migration {} applied successfully", migration.version);
            }
        }

        Ok(())
    }

    pub fn session_manager(&self) -> Arc<SessionManager> {
        self.session_manager.clone()
    }

    pub fn template_repo(&self) -> Arc<TemplateRepository> {
        self.template_repo.clone()
    }

    pub fn provider_manager(&self) -> Arc<ProviderManager> {
        self.provider_manager.clone()
    }

    pub async fn store_partial_result(
        &self,
        session_id: &str,
        batch_number: Option<i32>,
        result_type: ResultType,
        data: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO partial_results (
                session_id, batch_number, result_type, data, created_at
            )
            VALUES (?, ?, ?, ?, datetime('now'))
        "#)
        .bind(session_id)
        .bind(batch_number)
        .bind(result_type.to_string())
        .bind(data.to_string())
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    pub async fn get_partial_results(
        &self,
        session_id: &str,
        result_type: Option<ResultType>,
    ) -> Result<Vec<PartialResult>> {
        let query = if let Some(rt) = result_type {
            sqlx::query(r#"
                SELECT * FROM partial_results
                WHERE session_id = ? AND result_type = ?
                ORDER BY created_at ASC
            "#)
            .bind(session_id)
            .bind(rt.to_string())
        } else {
            sqlx::query(r#"
                SELECT * FROM partial_results
                WHERE session_id = ?
                ORDER BY created_at ASC
            "#)
            .bind(session_id)
        };

        let rows = query.fetch_all(&self.pool).await?;
        
        rows.into_iter()
            .map(|row| {
                Ok(PartialResult {
                    id: row.get("id"),
                    session_id: row.get("session_id"),
                    batch_number: row.get("batch_number"),
                    result_type: match row.get::<String, _>("result_type").as_str() {
                        "facet" => ResultType::Facet,
                        "cluster" => ResultType::Cluster,
                        "insight" => ResultType::Insight,
                        "embedding" => ResultType::Embedding,
                        _ => ResultType::Facet,
                    },
                    data: serde_json::from_str(&row.get::<String, _>("data"))?,
                    created_at: 0,
                })
            })
            .collect()
    }

    pub async fn cache_response(
        &self,
        cache_key: &str,
        cache_type: &str,
        provider: &str,
        input_hash: &str,
        output: serde_json::Value,
        ttl_seconds: i32,
    ) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO response_cache (
                id, cache_key, cache_type, provider, input_hash, output,
                created_at, accessed_at, access_count, ttl_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 1, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                output = excluded.output,
                accessed_at = datetime('now'),
                access_count = access_count + 1
        "#)
        .bind(Uuid::new_v4().to_string())
        .bind(cache_key)
        .bind(cache_type)
        .bind(provider)
        .bind(input_hash)
        .bind(output.to_string())
        .bind(ttl_seconds)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    pub async fn get_cached_response(
        &self,
        cache_key: &str,
    ) -> Result<Option<serde_json::Value>> {
        let row = sqlx::query(r#"
            SELECT output FROM response_cache
            WHERE cache_key = ? 
            AND datetime('now') < datetime(created_at, '+' || ttl_seconds || ' seconds')
        "#)
        .bind(cache_key)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let output: String = row.get("output");
            Ok(Some(serde_json::from_str(&output)?))
        } else {
            Ok(None)
        }
    }

    pub async fn cleanup_expired_cache(&self) -> Result<()> {
        sqlx::query(r#"
            DELETE FROM response_cache
            WHERE datetime('now') > datetime(created_at, '+' || ttl_seconds || ' seconds')
        "#)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

pub fn compute_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ============================================================================
// Initialize Default Templates
// ============================================================================

pub async fn initialize_default_templates(persistence: &EnhancedPersistenceLayer) -> Result<()> {
    let templates = vec![
        AnalysisTemplate {
            id: "customer-support".to_string(),
            name: "Customer Support Analysis".to_string(),
            description: Some("Analyze customer support conversations for common issues and sentiment".to_string()),
            category: TemplateCategory::Support,
            is_public: true,
            config: serde_json::json!({
                "facets": ["sentiment", "issue_type", "resolution_status", "urgency"],
                "clustering": {
                    "min_cluster_size": 5,
                    "max_clusters": 20
                }
            }),
            custom_prompts: Some(serde_json::json!({
                "sentiment": "Analyze the overall sentiment of this conversation (positive, neutral, negative)",
                "issue_type": "What is the main issue or topic being discussed?",
                "resolution_status": "Was the issue resolved? (resolved, unresolved, partial)",
                "urgency": "How urgent is this issue? (low, medium, high, critical)"
            })),
            facet_definitions: None,
            created_at: current_timestamp(),
            updated_at: current_timestamp(),
            created_by: Some("system".to_string()),
        },
        AnalysisTemplate {
            id: "sales-conversations".to_string(),
            name: "Sales Conversation Analysis".to_string(),
            description: Some("Analyze sales conversations for opportunities and objections".to_string()),
            category: TemplateCategory::Sales,
            is_public: true,
            config: serde_json::json!({
                "facets": ["deal_stage", "objections", "competitor_mentions", "budget_discussed"],
                "clustering": {
                    "min_cluster_size": 3,
                    "max_clusters": 15
                }
            }),
            custom_prompts: None,
            facet_definitions: None,
            created_at: current_timestamp(),
            updated_at: current_timestamp(),
            created_by: Some("system".to_string()),
        },
    ];

    for template in templates {
        // Check if template already exists
        if persistence.template_repo().get_template(&template.id).await?.is_none() {
            persistence.template_repo().create_template(template).await?;
        }
    }

    Ok(())
}