use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

use crate::clustering;
use crate::config::BriefXAIConfig;
use crate::embeddings;
use crate::facets;
use crate::persistence_v2::{
    AnalysisSession, BatchStatus, EnhancedPersistenceLayer, ResultType, SessionStatus,
};
use crate::types::{ConversationData, FacetValue};
use crate::umap;

// ============================================================================
// Session State and Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionEvent {
    Started {
        session_id: String,
    },
    Paused {
        session_id: String,
    },
    Resumed {
        session_id: String,
    },
    BatchStarted {
        session_id: String,
        batch_number: i32,
        total_batches: i32,
    },
    BatchCompleted {
        session_id: String,
        batch_number: i32,
    },
    BatchFailed {
        session_id: String,
        batch_number: i32,
        error: String,
    },
    ProgressUpdate {
        session_id: String,
        stage: String,
        progress: f32,
        message: String,
    },
    PartialResultAvailable {
        session_id: String,
        result_type: String,
    },
    Completed {
        session_id: String,
    },
    Failed {
        session_id: String,
        error: String,
    },
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub session: AnalysisSession,
    pub is_paused: Arc<RwLock<bool>>,
    pub should_stop: Arc<RwLock<bool>>,
    pub event_sender: broadcast::Sender<SessionEvent>,
    pub start_time: Instant,
    pub pause_duration: Duration,
}

// ============================================================================
// Batch Processing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    pub number: i32,
    pub conversations: Vec<ConversationData>,
    pub start_index: usize,
    pub end_index: usize,
}

impl Batch {
    pub fn create_batches(conversations: &[ConversationData], batch_size: usize) -> Vec<Batch> {
        conversations
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| Batch {
                number: i as i32,
                conversations: chunk.to_vec(),
                start_index: i * batch_size,
                end_index: std::cmp::min((i + 1) * batch_size, conversations.len()),
            })
            .collect()
    }
}

// ============================================================================
// Analysis Session Manager
// ============================================================================

pub struct AnalysisSessionManager {
    inner: Arc<AnalysisSessionManagerInner>,
}

struct AnalysisSessionManagerInner {
    persistence: Arc<EnhancedPersistenceLayer>,
    active_sessions: Arc<RwLock<std::collections::HashMap<String, SessionState>>>,
}

impl AnalysisSessionManager {
    pub fn new(persistence: Arc<EnhancedPersistenceLayer>) -> Self {
        Self {
            inner: Arc::new(AnalysisSessionManagerInner {
                persistence,
                active_sessions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            }),
        }
    }

    // ========================================================================
    // Session Lifecycle
    // ========================================================================

    pub async fn start_analysis(
        &self,
        config: BriefXAIConfig,
        conversations: Vec<ConversationData>,
    ) -> Result<(String, broadcast::Receiver<SessionEvent>)> {
        // Create new session
        let session = self
            .inner
            .persistence
            .session_manager()
            .create_session(config.clone())
            .await?;

        let session_id = session.id.clone();
        let (event_sender, event_receiver) = broadcast::channel(1000);

        // Create session state
        let state = SessionState {
            session: session.clone(),
            is_paused: Arc::new(RwLock::new(false)),
            should_stop: Arc::new(RwLock::new(false)),
            event_sender: event_sender.clone(),
            start_time: Instant::now(),
            pause_duration: Duration::from_secs(0),
        };

        // Store in active sessions
        self.inner
            .active_sessions
            .write()
            .await
            .insert(session_id.clone(), state.clone());

        // Send start event
        let _ = event_sender.send(SessionEvent::Started {
            session_id: session_id.clone(),
        });

        // Start analysis in background
        let inner = self.inner();
        let session_id_clone = session_id.clone();
        tokio::spawn(async move {
            if let Err(e) = AnalysisSessionManager::run_analysis_internal_static(
                inner.clone(),
                session_id_clone.clone(),
                config,
                conversations,
            )
            .await
            {
                error!("Analysis failed for session {}: {}", session_id_clone, e);
                let _ = AnalysisSessionManager::mark_session_failed_static(
                    &inner,
                    &session_id_clone,
                    &e.to_string(),
                )
                .await;
            }
        });

        Ok((session_id, event_receiver))
    }

    pub async fn pause_analysis(&self, session_id: &str) -> Result<()> {
        let sessions = self.inner.active_sessions.read().await;
        let state = sessions
            .get(session_id)
            .context("Session not found or not active")?;

        *state.is_paused.write().await = true;

        self.inner
            .persistence
            .session_manager()
            .pause_session(session_id)
            .await?;

        let _ = state.event_sender.send(SessionEvent::Paused {
            session_id: session_id.to_string(),
        });

        info!("Session {} paused", session_id);
        Ok(())
    }

    pub async fn resume_analysis(
        &self,
        session_id: &str,
    ) -> Result<broadcast::Receiver<SessionEvent>> {
        // Get session from persistence
        let session = self
            .inner
            .persistence
            .session_manager()
            .resume_session(session_id)
            .await?;

        // Check if already active
        if self
            .inner
            .active_sessions
            .read()
            .await
            .contains_key(session_id)
        {
            bail!("Session is already active");
        }

        let (event_sender, event_receiver) = broadcast::channel(1000);

        // Create session state
        let state = SessionState {
            session: session.clone(),
            is_paused: Arc::new(RwLock::new(false)),
            should_stop: Arc::new(RwLock::new(false)),
            event_sender: event_sender.clone(),
            start_time: Instant::now(),
            pause_duration: Duration::from_secs(0),
        };

        // Store in active sessions
        self.inner
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), state);

        // Send resume event
        let _ = event_sender.send(SessionEvent::Resumed {
            session_id: session_id.to_string(),
        });

        info!("Session {} resumed", session_id);

        // Resume analysis in background
        let _inner = self.inner();
        let _session_id_clone = session_id.to_string();
        let _config = session.config.clone();

        tokio::spawn(async move {
            // Reload conversations from persistence
            // For now, we'll need to handle this differently
            warn!("Resume functionality needs conversation reload implementation");
        });

        Ok(event_receiver)
    }

    pub async fn stop_analysis(&self, session_id: &str) -> Result<()> {
        let sessions = self.inner.active_sessions.read().await;
        let state = sessions
            .get(session_id)
            .context("Session not found or not active")?;

        *state.should_stop.write().await = true;

        info!("Stopping session {}", session_id);
        Ok(())
    }

    pub async fn get_session_status(&self, session_id: &str) -> Result<SessionStatus> {
        if let Some(state) = self.inner.active_sessions.read().await.get(session_id) {
            Ok(state.session.status.clone())
        } else {
            self.inner
                .persistence
                .session_manager()
                .get_session(session_id)
                .await?
                .map(|s| s.status)
                .context("Session not found")
        }
    }

    pub async fn list_sessions(&self) -> Result<Vec<AnalysisSession>> {
        // This would query the database for all sessions
        // For now, return active sessions
        let sessions = self.inner.active_sessions.read().await;
        Ok(sessions.values().map(|s| s.session.clone()).collect())
    }

    // ========================================================================
    // Internal Analysis Runner
    // ========================================================================

    async fn run_analysis_internal_static(
        inner: Arc<AnalysisSessionManagerInner>,
        session_id: String,
        config: BriefXAIConfig,
        conversations: Vec<ConversationData>,
    ) -> Result<()> {
        let state = inner
            .active_sessions
            .read()
            .await
            .get(&session_id)
            .cloned()
            .context("Session state not found")?;

        // Update session with total conversations
        inner
            .persistence
            .session_manager()
            .update_session_status(&session_id, SessionStatus::Running)
            .await?;

        // Create batches
        let batch_size = config.batch_size.unwrap_or(100);
        let batches = Batch::create_batches(&conversations, batch_size);
        let total_batches = batches.len() as i32;

        info!(
            "Starting analysis for session {} with {} batches",
            session_id, total_batches
        );

        // Process each batch
        for batch in batches {
            // Check if should stop
            if *state.should_stop.read().await {
                info!("Session {} stopped by user", session_id);
                break;
            }

            // Check if paused
            while *state.is_paused.read().await {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if *state.should_stop.read().await {
                    break;
                }
            }

            // Send batch started event
            let _ = state.event_sender.send(SessionEvent::BatchStarted {
                session_id: session_id.clone(),
                batch_number: batch.number,
                total_batches,
            });

            // Process batch
            match Self::process_batch_static(&inner, &session_id, &config, &batch, &state).await {
                Ok(()) => {
                    // Save batch progress
                    inner
                        .persistence
                        .session_manager()
                        .save_batch_progress(
                            &session_id,
                            batch.number,
                            BatchStatus::Completed,
                            None,
                            None,
                        )
                        .await?;

                    // Send batch completed event
                    let _ = state.event_sender.send(SessionEvent::BatchCompleted {
                        session_id: session_id.clone(),
                        batch_number: batch.number,
                    });
                }
                Err(e) => {
                    error!(
                        "Batch {} failed for session {}: {}",
                        batch.number, session_id, e
                    );

                    // Save batch failure
                    inner
                        .persistence
                        .session_manager()
                        .save_batch_progress(
                            &session_id,
                            batch.number,
                            BatchStatus::Failed,
                            None,
                            Some(e.to_string()),
                        )
                        .await?;

                    // Send batch failed event
                    let _ = state.event_sender.send(SessionEvent::BatchFailed {
                        session_id: session_id.clone(),
                        batch_number: batch.number,
                        error: e.to_string(),
                    });

                    // Continue with next batch (could make this configurable)
                }
            }
        }

        // Finalize analysis
        Self::finalize_analysis_static(&inner, &session_id, &config, &state).await?;

        // Mark session as completed
        inner
            .persistence
            .session_manager()
            .update_session_status(&session_id, SessionStatus::Completed)
            .await?;

        // Send completed event
        let _ = state.event_sender.send(SessionEvent::Completed {
            session_id: session_id.clone(),
        });

        // Remove from active sessions
        inner.active_sessions.write().await.remove(&session_id);

        info!("Analysis completed for session {}", session_id);
        Ok(())
    }

    async fn process_batch_static(
        inner: &Arc<AnalysisSessionManagerInner>,
        session_id: &str,
        config: &BriefXAIConfig,
        batch: &Batch,
        state: &SessionState,
    ) -> Result<()> {
        debug!(
            "Processing batch {} for session {}",
            batch.number, session_id
        );

        // Extract facets
        let _ = state.event_sender.send(SessionEvent::ProgressUpdate {
            session_id: session_id.to_string(),
            stage: "facet_extraction".to_string(),
            progress: 0.0,
            message: format!("Extracting facets for batch {}", batch.number),
        });

        let facets = facets::extract_facets(config, &batch.conversations).await?;

        // Store partial facet results
        inner
            .persistence
            .store_partial_result(
                session_id,
                Some(batch.number),
                ResultType::Facet,
                serde_json::to_value(&facets)?,
            )
            .await?;

        // Generate embeddings
        let _ = state.event_sender.send(SessionEvent::ProgressUpdate {
            session_id: session_id.to_string(),
            stage: "embedding_generation".to_string(),
            progress: 0.33,
            message: format!("Generating embeddings for batch {}", batch.number),
        });

        let embeddings = embeddings::generate_embeddings(config, &batch.conversations).await?;

        // Store partial embedding results
        inner
            .persistence
            .store_partial_result(
                session_id,
                Some(batch.number),
                ResultType::Embedding,
                serde_json::to_value(&embeddings)?,
            )
            .await?;

        // Notify partial results available
        let _ = state
            .event_sender
            .send(SessionEvent::PartialResultAvailable {
                session_id: session_id.to_string(),
                result_type: "batch".to_string(),
            });

        Ok(())
    }

    async fn finalize_analysis_static(
        inner: &Arc<AnalysisSessionManagerInner>,
        session_id: &str,
        config: &BriefXAIConfig,
        state: &SessionState,
    ) -> Result<()> {
        info!("Finalizing analysis for session {}", session_id);

        // Load all partial results
        let facet_results = inner
            .persistence
            .get_partial_results(session_id, Some(ResultType::Facet))
            .await?;

        let embedding_results = inner
            .persistence
            .get_partial_results(session_id, Some(ResultType::Embedding))
            .await?;

        // Combine all facets
        let mut all_facets = Vec::new();
        for result in facet_results {
            let facets: Vec<Vec<FacetValue>> = serde_json::from_value(result.data)?;
            all_facets.extend(facets);
        }

        // Combine all embeddings
        let mut all_embeddings = Vec::new();
        for result in embedding_results {
            let embeddings: Vec<Vec<f32>> = serde_json::from_value(result.data)?;
            all_embeddings.extend(embeddings);
        }

        // Perform clustering
        let _ = state.event_sender.send(SessionEvent::ProgressUpdate {
            session_id: session_id.to_string(),
            stage: "clustering".to_string(),
            progress: 0.7,
            message: "Performing clustering analysis".to_string(),
        });

        let base_clusters =
            clustering::create_base_clusters(config, &all_embeddings, &all_facets).await?;

        let hierarchy = clustering::build_hierarchy(config, base_clusters).await?;

        // Store clustering results
        inner
            .persistence
            .store_partial_result(
                session_id,
                None,
                ResultType::Cluster,
                serde_json::to_value(&hierarchy)?,
            )
            .await?;

        // Generate UMAP visualization
        let _ = state.event_sender.send(SessionEvent::ProgressUpdate {
            session_id: session_id.to_string(),
            stage: "visualization".to_string(),
            progress: 0.9,
            message: "Generating visualizations".to_string(),
        });

        let umap_coords = umap::generate_umap(config, &all_embeddings).await?;

        // Store final results
        let final_results = serde_json::json!({
            "clusters": hierarchy,
            "facets": all_facets,
            "umap_coords": umap_coords,
            "total_conversations": all_facets.len(),
        });

        // Update session with results
        // Note: This would need to be implemented through the persistence layer
        // For now, we'll store as partial result
        inner
            .persistence
            .store_partial_result(session_id, None, ResultType::Insight, final_results)
            .await?;

        Ok(())
    }

    async fn mark_session_failed_static(
        inner: &Arc<AnalysisSessionManagerInner>,
        session_id: &str,
        error: &str,
    ) -> Result<()> {
        inner
            .persistence
            .session_manager()
            .update_session_status(session_id, SessionStatus::Failed)
            .await?;

        if let Some(state) = inner.active_sessions.read().await.get(session_id) {
            let _ = state.event_sender.send(SessionEvent::Failed {
                session_id: session_id.to_string(),
                error: error.to_string(),
            });
        }

        inner.active_sessions.write().await.remove(session_id);

        Ok(())
    }

    // Helper to get inner Arc for spawning tasks
    fn inner(&self) -> Arc<AnalysisSessionManagerInner> {
        self.inner.clone()
    }
}

// ============================================================================
// Progress Tracker
// ============================================================================

pub struct ProgressTracker {
    session_id: String,
    event_sender: broadcast::Sender<SessionEvent>,
    current_stage: RwLock<String>,
    current_progress: RwLock<f32>,
}

impl ProgressTracker {
    pub fn new(session_id: String, event_sender: broadcast::Sender<SessionEvent>) -> Self {
        Self {
            session_id,
            event_sender,
            current_stage: RwLock::new("initializing".to_string()),
            current_progress: RwLock::new(0.0),
        }
    }

    pub async fn update(&self, stage: &str, progress: f32, message: &str) {
        *self.current_stage.write().await = stage.to_string();
        *self.current_progress.write().await = progress;

        let _ = self.event_sender.send(SessionEvent::ProgressUpdate {
            session_id: self.session_id.clone(),
            stage: stage.to_string(),
            progress,
            message: message.to_string(),
        });
    }

    pub async fn get_current(&self) -> (String, f32) {
        let stage = self.current_stage.read().await.clone();
        let progress = *self.current_progress.read().await;
        (stage, progress)
    }
}

// ============================================================================
// Time and Cost Estimation
// ============================================================================

pub struct EstimationService {
    #[allow(dead_code)]
    persistence: Arc<EnhancedPersistenceLayer>,
}

impl EstimationService {
    pub fn new(persistence: Arc<EnhancedPersistenceLayer>) -> Self {
        Self { persistence }
    }

    pub async fn estimate_time(
        &self,
        num_conversations: usize,
        config: &BriefXAIConfig,
    ) -> Duration {
        // Based on historical data, estimate processing time
        // For now, use rough estimates
        let per_conversation_ms = match config.llm_provider.as_str() {
            "openai" => 500,
            "ollama" => 1000,
            "vllm" => 300,
            _ => 750,
        };

        Duration::from_millis((num_conversations * per_conversation_ms) as u64)
    }

    pub async fn estimate_cost(&self, num_conversations: usize, config: &BriefXAIConfig) -> f64 {
        // Based on provider and model, estimate cost
        // This would need actual token counting and pricing data
        let per_conversation_cost = match (config.llm_provider.as_str(), config.llm_model.as_str())
        {
            ("openai", "gpt-4o") => 0.01,
            ("openai", "gpt-4o-mini") => 0.002,
            ("openai", "gpt-3.5-turbo") => 0.001,
            _ => 0.0,
        };

        num_conversations as f64 * per_conversation_cost
    }
}
