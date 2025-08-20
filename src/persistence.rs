use anyhow::{Result, Context};
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions, Row};
use serde_json;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, debug};
use tokio::sync::RwLock;
use std::sync::Arc;
use dashmap::DashMap;

use crate::types::{ConversationData, ConversationCluster, FacetValue};
use crate::config::BriefXAIConfig;

// Cache types
pub type CacheKey = String;
pub type CacheValue = Vec<u8>;

// In-memory cache with TTL
pub struct MemoryCache {
    data: Arc<DashMap<CacheKey, (CacheValue, u64)>>, // value, expiry_timestamp
    max_size: usize,
}

impl MemoryCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: Arc::new(DashMap::new()),
            max_size,
        }
    }
    
    pub fn get(&self, key: &str) -> Option<CacheValue> {
        if let Some(entry) = self.data.get(key) {
            let (value, expiry) = entry.value();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            if *expiry > now {
                Some(value.clone())
            } else {
                self.data.remove(key);
                None
            }
        } else {
            None
        }
    }
    
    pub fn set(&self, key: String, value: CacheValue, ttl_seconds: u64) {
        let expiry = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() + ttl_seconds;
        
        // Evict old entries if cache is full
        if self.data.len() >= self.max_size {
            self.evict_oldest();
        }
        
        self.data.insert(key, (value, expiry));
    }
    
    fn evict_oldest(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // First try to remove expired entries
        let expired: Vec<_> = self.data
            .iter()
            .filter(|entry| entry.value().1 <= now)
            .map(|entry| entry.key().clone())
            .collect();
        
        for key in expired {
            self.data.remove(&key);
        }
        
        // If still full, remove oldest entries
        if self.data.len() >= self.max_size {
            let mut entries: Vec<_> = self.data
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().1))
                .collect();
            
            entries.sort_by_key(|e| e.1);
            
            for (key, _) in entries.iter().take(self.max_size / 4) {
                self.data.remove(key);
            }
        }
    }
    
    pub fn clear(&self) {
        self.data.clear();
    }
}

// Database persistence layer
pub struct PersistenceLayer {
    pool: SqlitePool,
    memory_cache: MemoryCache,
    config: BriefXAIConfig,
}

impl PersistenceLayer {
    pub async fn new(config: BriefXAIConfig) -> Result<Self> {
        let db_path = config.cache_dir.join("briefxai.db");
        
        // Create cache directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create cache directory")?;
        }
        
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        
        info!("Initializing persistence layer at {}", db_path.display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await
            .context("Failed to connect to database")?;
        
        // Run migrations
        Self::run_migrations(&pool).await?;
        
        Ok(Self {
            pool,
            memory_cache: MemoryCache::new(1000),
            config,
        })
    }
    
    async fn run_migrations(pool: &SqlitePool) -> Result<()> {
        debug!("Running database migrations");
        
        // Create tables
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                metadata TEXT
            )
        "#)
        .execute(pool)
        .await?;
        
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        "#)
        .execute(pool)
        .await?;
        
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL
            )
        "#)
        .execute(pool)
        .await?;
        
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS analysis_results (
                session_id TEXT PRIMARY KEY,
                clusters TEXT NOT NULL,
                facets TEXT NOT NULL,
                umap_coords TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        "#)
        .execute(pool)
        .await?;
        
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL
            )
        "#)
        .execute(pool)
        .await?;
        
        // Create indices
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
            .execute(pool)
            .await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
            .execute(pool)
            .await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings_cache(model)")
            .execute(pool)
            .await?;
        
        info!("Database migrations completed");
        Ok(())
    }
    
    // Session management
    pub async fn create_session(&self, session_id: &str, config: &BriefXAIConfig) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        let config_json = serde_json::to_string(config)?;
        
        sqlx::query(r#"
            INSERT INTO sessions (id, created_at, updated_at, status, config)
            VALUES (?, ?, ?, 'created', ?)
        "#)
        .bind(session_id)
        .bind(now)
        .bind(now)
        .bind(config_json)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn update_session_status(&self, session_id: &str, status: &str) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        sqlx::query(r#"
            UPDATE sessions 
            SET status = ?, updated_at = ?
            WHERE id = ?
        "#)
        .bind(status)
        .bind(now)
        .bind(session_id)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionInfo>> {
        let row = sqlx::query(r#"
            SELECT id, created_at, updated_at, status, config, metadata
            FROM sessions
            WHERE id = ?
        "#)
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(row) = row {
            Ok(Some(SessionInfo {
                id: row.get("id"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                status: row.get("status"),
                config: serde_json::from_str(row.get("config"))?,
                metadata: row.get::<Option<String>, _>("metadata")
                    .and_then(|m| serde_json::from_str(&m).ok()),
            }))
        } else {
            Ok(None)
        }
    }
    
    // Conversation storage
    pub async fn store_conversations(
        &self,
        session_id: &str,
        conversations: &[ConversationData],
    ) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        for (i, conv) in conversations.iter().enumerate() {
            let conv_id = format!("{}-{}", session_id, i);
            let data = serde_json::to_string(conv)?;
            
            sqlx::query(r#"
                INSERT OR REPLACE INTO conversations (id, session_id, data, created_at)
                VALUES (?, ?, ?, ?)
            "#)
            .bind(&conv_id)
            .bind(session_id)
            .bind(data)
            .bind(now)
            .execute(&self.pool)
            .await?;
        }
        
        Ok(())
    }
    
    pub async fn get_conversations(&self, session_id: &str) -> Result<Vec<ConversationData>> {
        let rows = sqlx::query(r#"
            SELECT data FROM conversations
            WHERE session_id = ?
            ORDER BY id
        "#)
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        
        let mut conversations = Vec::new();
        for row in rows {
            let data: String = row.get("data");
            conversations.push(serde_json::from_str(&data)?);
        }
        
        Ok(conversations)
    }
    
    // Embedding cache
    pub async fn get_cached_embedding(&self, hash: &str, model: &str) -> Result<Option<Vec<f32>>> {
        // Check memory cache first
        let cache_key = format!("emb:{}:{}", model, hash);
        if let Some(cached) = self.memory_cache.get(&cache_key) {
            let embedding: Vec<f32> = bincode::deserialize(&cached)?;
            return Ok(Some(embedding));
        }
        
        // Check database
        let row = sqlx::query(r#"
            SELECT embedding FROM embeddings_cache
            WHERE hash = ? AND model = ?
        "#)
        .bind(hash)
        .bind(model)
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(row) = row {
            let blob: Vec<u8> = row.get("embedding");
            let embedding: Vec<f32> = bincode::deserialize(&blob)?;
            
            // Update accessed time
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64;
            
            sqlx::query("UPDATE embeddings_cache SET accessed_at = ? WHERE hash = ?")
                .bind(now)
                .bind(hash)
                .execute(&self.pool)
                .await?;
            
            // Store in memory cache
            self.memory_cache.set(cache_key, blob, 3600); // 1 hour TTL
            
            Ok(Some(embedding))
        } else {
            Ok(None)
        }
    }
    
    pub async fn cache_embedding(&self, hash: &str, model: &str, embedding: &[f32]) -> Result<()> {
        let blob = bincode::serialize(embedding)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        sqlx::query(r#"
            INSERT OR REPLACE INTO embeddings_cache (hash, model, embedding, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?)
        "#)
        .bind(hash)
        .bind(model)
        .bind(&blob)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;
        
        // Store in memory cache
        let cache_key = format!("emb:{}:{}", model, hash);
        self.memory_cache.set(cache_key, blob, 3600);
        
        Ok(())
    }
    
    // LLM response cache
    pub async fn get_cached_llm_response(&self, prompt_hash: &str, model: &str) -> Result<Option<String>> {
        // Check memory cache first
        let cache_key = format!("llm:{}:{}", model, prompt_hash);
        if let Some(cached) = self.memory_cache.get(&cache_key) {
            let response = String::from_utf8(cached)?;
            return Ok(Some(response));
        }
        
        // Check database
        let row = sqlx::query(r#"
            SELECT response FROM llm_cache
            WHERE prompt_hash = ? AND model = ?
        "#)
        .bind(prompt_hash)
        .bind(model)
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(row) = row {
            let response: String = row.get("response");
            
            // Update accessed time
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64;
            
            sqlx::query("UPDATE llm_cache SET accessed_at = ? WHERE prompt_hash = ?")
                .bind(now)
                .bind(prompt_hash)
                .execute(&self.pool)
                .await?;
            
            // Store in memory cache
            self.memory_cache.set(cache_key, response.as_bytes().to_vec(), 3600);
            
            Ok(Some(response))
        } else {
            Ok(None)
        }
    }
    
    pub async fn cache_llm_response(&self, prompt_hash: &str, model: &str, response: &str) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        sqlx::query(r#"
            INSERT OR REPLACE INTO llm_cache (prompt_hash, model, response, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?)
        "#)
        .bind(prompt_hash)
        .bind(model)
        .bind(response)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;
        
        // Store in memory cache
        let cache_key = format!("llm:{}:{}", model, prompt_hash);
        self.memory_cache.set(cache_key, response.as_bytes().to_vec(), 3600);
        
        Ok(())
    }
    
    // Analysis results storage
    pub async fn store_analysis_results(
        &self,
        session_id: &str,
        clusters: &[ConversationCluster],
        facets: &[Vec<FacetValue>],
        umap_coords: &[(f32, f32)],
    ) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        let clusters_json = serde_json::to_string(clusters)?;
        let facets_json = serde_json::to_string(facets)?;
        let umap_json = serde_json::to_string(umap_coords)?;
        
        sqlx::query(r#"
            INSERT OR REPLACE INTO analysis_results (session_id, clusters, facets, umap_coords, created_at)
            VALUES (?, ?, ?, ?, ?)
        "#)
        .bind(session_id)
        .bind(clusters_json)
        .bind(facets_json)
        .bind(umap_json)
        .bind(now)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get_analysis_results(
        &self,
        session_id: &str,
    ) -> Result<Option<AnalysisResults>> {
        let row = sqlx::query(r#"
            SELECT clusters, facets, umap_coords FROM analysis_results
            WHERE session_id = ?
        "#)
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(row) = row {
            Ok(Some(AnalysisResults {
                clusters: serde_json::from_str(row.get("clusters"))?,
                facets: serde_json::from_str(row.get("facets"))?,
                umap_coords: serde_json::from_str(row.get("umap_coords"))?,
            }))
        } else {
            Ok(None)
        }
    }
    
    // Cleanup old data
    pub async fn cleanup_old_data(&self, days_to_keep: u64) -> Result<()> {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64 - (days_to_keep * 24 * 3600) as i64;
        
        info!("Cleaning up data older than {} days", days_to_keep);
        
        // Delete old sessions and related data
        sqlx::query("DELETE FROM conversations WHERE session_id IN (SELECT id FROM sessions WHERE created_at < ?)")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;
        
        sqlx::query("DELETE FROM analysis_results WHERE session_id IN (SELECT id FROM sessions WHERE created_at < ?)")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;
        
        sqlx::query("DELETE FROM sessions WHERE created_at < ?")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;
        
        // Clean up old cache entries
        sqlx::query("DELETE FROM embeddings_cache WHERE accessed_at < ?")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;
        
        sqlx::query("DELETE FROM llm_cache WHERE accessed_at < ?")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;
        
        // Vacuum database to reclaim space
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await?;
        
        info!("Cleanup completed");
        Ok(())
    }
    
    // Export/Import functionality
    pub async fn export_session(&self, session_id: &str) -> Result<SessionExport> {
        let session = self.get_session(session_id).await?
            .context("Session not found")?;
        let conversations = self.get_conversations(session_id).await?;
        let analysis = self.get_analysis_results(session_id).await?;
        
        Ok(SessionExport {
            session,
            conversations,
            analysis,
        })
    }
    
    pub async fn import_session(&self, export: SessionExport) -> Result<String> {
        // Generate new session ID to avoid conflicts
        let new_session_id = uuid::Uuid::new_v4().to_string();
        
        // Create session
        self.create_session(&new_session_id, &export.session.config).await?;
        
        // Store conversations
        self.store_conversations(&new_session_id, &export.conversations).await?;
        
        // Store analysis results if present
        if let Some(analysis) = export.analysis {
            self.store_analysis_results(
                &new_session_id,
                &analysis.clusters,
                &analysis.facets,
                &analysis.umap_coords,
            ).await?;
        }
        
        Ok(new_session_id)
    }
}

// Data structures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub status: String,
    pub config: BriefXAIConfig,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnalysisResults {
    pub clusters: Vec<ConversationCluster>,
    pub facets: Vec<Vec<FacetValue>>,
    pub umap_coords: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionExport {
    pub session: SessionInfo,
    pub conversations: Vec<ConversationData>,
    pub analysis: Option<AnalysisResults>,
}

// Global persistence instance
lazy_static::lazy_static! {
    static ref GLOBAL_PERSISTENCE: Arc<RwLock<Option<Arc<PersistenceLayer>>>> = Arc::new(RwLock::new(None));
}

pub async fn initialize_persistence(config: BriefXAIConfig) -> Result<()> {
    let persistence = Arc::new(PersistenceLayer::new(config).await?);
    let mut global = GLOBAL_PERSISTENCE.write().await;
    *global = Some(persistence);
    Ok(())
}

pub async fn get_persistence() -> Result<Arc<PersistenceLayer>> {
    let global = GLOBAL_PERSISTENCE.read().await;
    global.as_ref()
        .cloned()
        .context("Persistence layer not initialized")
}

// Helper function to compute hash for caching
pub fn compute_hash(data: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}