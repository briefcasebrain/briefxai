use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, Path as AxumPath, State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::stream::StreamExt;
use serde_json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{broadcast, RwLock};
use tower_http::services::ServeDir;
use tracing::{debug, info, warn};

use crate::config::{BriefXAIConfig, EmbeddingProvider, LlmProvider};
use crate::types::{ConversationCluster, FacetValue};
// use crate::clio_api::{create_clio_routes, create_clio_state};

// Progress tracking structures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressUpdate {
    pub step: String,
    pub progress: f32, // 0.0 to 100.0
    pub message: String,
    pub details: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ProgressMessage {
    Started { session_id: String },
    Update(ProgressUpdate),
    Completed { session_id: String, result: String },
    Error { session_id: String, error: String },
}

// Global progress tracker
pub type ProgressSender = broadcast::Sender<ProgressMessage>;
pub type ProgressReceiver = broadcast::Receiver<ProgressMessage>;

#[derive(Clone)]
pub struct ProgressTracker {
    sender: ProgressSender,
    sessions: Arc<RwLock<HashMap<String, String>>>, // session_id -> status
}

impl ProgressTracker {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self {
            sender,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start_session(&self, session_id: String) -> Result<()> {
        self.sessions
            .write()
            .await
            .insert(session_id.clone(), "started".to_string());
        let _ = self.sender.send(ProgressMessage::Started { session_id });
        Ok(())
    }

    pub async fn update_progress(
        &self,
        session_id: &str,
        step: &str,
        progress: f32,
        message: &str,
        details: Option<String>,
    ) -> Result<()> {
        let update = ProgressUpdate {
            step: step.to_string(),
            progress,
            message: message.to_string(),
            details,
            timestamp: chrono::Utc::now(),
        };

        debug!(
            "Progress update for {}: {} - {:.1}% - {}",
            session_id, step, progress, message
        );
        let _ = self.sender.send(ProgressMessage::Update(update));
        Ok(())
    }

    pub async fn complete_session(&self, session_id: String, result: String) -> Result<()> {
        self.sessions
            .write()
            .await
            .insert(session_id.clone(), "completed".to_string());
        let _ = self
            .sender
            .send(ProgressMessage::Completed { session_id, result });
        Ok(())
    }

    pub async fn error_session(&self, session_id: String, error: String) -> Result<()> {
        self.sessions
            .write()
            .await
            .insert(session_id.clone(), "error".to_string());
        let _ = self
            .sender
            .send(ProgressMessage::Error { session_id, error });
        Ok(())
    }

    pub fn subscribe(&self) -> ProgressReceiver {
        self.sender.subscribe()
    }

    pub async fn get_session_status(&self, session_id: &str) -> Option<String> {
        self.sessions.read().await.get(session_id).cloned()
    }
}

// Global progress tracker instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_PROGRESS: ProgressTracker = ProgressTracker::new();
}

pub async fn generate_website<P: AsRef<Path>>(
    config: &BriefXAIConfig,
    hierarchy: &[ConversationCluster],
    facet_data: &[Vec<FacetValue>],
    umap_coords: &[(f32, f32)],
    output_dir: P,
) -> Result<String> {
    let output_dir = output_dir.as_ref();
    info!("Generating static website in {:?}", output_dir);

    // Create output directories
    let static_dir = output_dir.join("static");
    let data_dir = output_dir.join("data");
    fs::create_dir_all(&static_dir).await?;
    fs::create_dir_all(&data_dir).await?;

    // Generate JSON data files
    let hierarchy_json = serde_json::to_string_pretty(hierarchy)?;
    let facet_json = serde_json::to_string_pretty(facet_data)?;
    let umap_json = serde_json::to_string_pretty(umap_coords)?;

    // Split data into chunks if needed
    let chunks = if hierarchy_json.len() > config.html_max_size_per_file {
        split_json_data(&hierarchy_json, config.html_max_size_per_file)
    } else {
        vec![hierarchy_json]
    };

    // Write data files
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_path = data_dir.join(format!("hierarchy_{}.json", i));
        fs::write(&chunk_path, chunk).await?;
    }

    fs::write(data_dir.join("facets.json"), facet_json).await?;
    fs::write(data_dir.join("umap.json"), umap_json).await?;

    // Generate HTML files
    generate_html_files(&static_dir, &chunks, config.website_password.is_some()).await?;

    // Copy static assets
    copy_static_assets(&static_dir).await?;

    // Start web server if configured
    if config.verbose {
        info!("Website generated at {:?}", output_dir);
        info!("To view, open {:?}/index.html", static_dir);
    }

    Ok(static_dir.to_string_lossy().to_string())
}

async fn generate_html_files(
    output_dir: &Path,
    data_chunks: &[String],
    password_protected: bool,
) -> Result<()> {
    // Generate main index.html
    let index_html = generate_index_html(data_chunks, password_protected)?;
    fs::write(output_dir.join("index.html"), index_html).await?;

    // Generate visualization page
    let viz_html = generate_visualization_html()?;
    fs::write(output_dir.join("visualization.html"), viz_html).await?;

    Ok(())
}

fn generate_index_html(_data_chunks: &[String], _password_protected: bool) -> Result<String> {
    let html = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BriefXAI Results</title>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div id="app">
        <header>
            <h1>BriefXAI Analysis Results</h1>
        </header>
        
        <nav>
            <ul>
                <li><a href="#hierarchy">Hierarchy View</a></li>
                <li><a href="#umap">UMAP Visualization</a></li>
                <li><a href="#conversations">Conversations</a></li>
            </ul>
        </nav>
        
        <main>
            <section id="hierarchy">
                <h2>Cluster Hierarchy</h2>
                <div id="hierarchy-tree"></div>
            </section>
            
            <section id="umap">
                <h2>UMAP Visualization</h2>
                <div id="umap-plot"></div>
            </section>
            
            <section id="conversations">
                <h2>Conversation Browser</h2>
                <div id="conversation-list"></div>
                <div id="conversation-detail"></div>
            </section>
        </main>
    </div>
    
    <script src="static/app.js"></script>
</body>
</html>"##;

    Ok(html.to_string())
}

fn generate_visualization_html() -> Result<String> {
    let html = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BriefXAI Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        .visualization {
            display: flex;
            gap: 20px;
        }
        
        .tree-view {
            flex: 1;
            min-height: 600px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            overflow: auto;
        }
        
        .umap-view {
            flex: 1;
            min-height: 600px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        
        .node {
            cursor: pointer;
        }
        
        .node circle {
            fill: #fff;
            stroke: steelblue;
            stroke-width: 3px;
        }
        
        .node text {
            font: 12px sans-serif;
        }
        
        .link {
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }
        
        .tooltip {
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 12px sans-serif;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>BriefXAI Analysis Visualization</h1>
        
        <div class="visualization">
            <div class="tree-view" id="tree"></div>
            <div class="umap-view" id="umap"></div>
        </div>
        
        <div id="details">
            <h2>Details</h2>
            <div id="detail-content"></div>
        </div>
    </div>
    
    <div class="tooltip"></div>
    
    <script>
        // Load data and initialize visualizations
        async function init() {
            const hierarchyData = await fetch('data/hierarchy_0.json').then(r => r.json());
            const umapData = await fetch('data/umap.json').then(r => r.json());
            
            drawTree(hierarchyData);
            drawUmap(umapData);
        }
        
        function drawTree(data) {
            // D3.js tree visualization
            const width = 600;
            const height = 600;
            
            const svg = d3.select("#tree")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            // Tree layout implementation
            // ... (simplified for brevity)
        }
        
        function drawUmap(data) {
            // D3.js UMAP scatter plot
            const width = 600;
            const height = 600;
            
            const svg = d3.select("#umap")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            // Scatter plot implementation
            // ... (simplified for brevity)
        }
        
        init();
    </script>
</body>
</html>"##;

    Ok(html.to_string())
}

async fn copy_static_assets(output_dir: &Path) -> Result<()> {
    // Create CSS file
    let css = r#" 
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 0;
    background: #f5f5f5;
}

header {
    background: #2c3e50;
    color: white;
    padding: 20px;
    text-align: center;
}

nav {
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 0;
}

nav ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
}

nav li {
    margin: 0;
}

nav a {
    display: block;
    padding: 15px 30px;
    color: #333;
    text-decoration: none;
    transition: background 0.3s;
}

nav a:hover {
    background: #f0f0f0;
}

main {
    max-width: 1400px;
    margin: 20px auto;
    padding: 0 20px;
}

section {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h2 {
    color: #2c3e50;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 10px;
}
"#;

    fs::write(output_dir.join("style.css"), css).await?;

    // Create JavaScript file
    let js = r#"// Main application JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('BriefXAI website loaded');
    
    // Load and display data
    loadData();
});

async function loadData() {
    try {
        const hierarchy = await fetch('data/hierarchy_0.json').then(r => r.json());
        const facets = await fetch('data/facets.json').then(r => r.json());
        const umap = await fetch('data/umap.json').then(r => r.json());
        
        displayHierarchy(hierarchy);
        displayUmap(umap);
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function displayHierarchy(data) {
    const container = document.getElementById('hierarchy-tree');
    // Implement tree visualization
}

function displayUmap(data) {
    const container = document.getElementById('umap-plot');
    // Implement UMAP visualization
}"#;

    fs::write(output_dir.join("app.js"), js).await?;

    Ok(())
}

fn split_json_data(data: &str, max_size: usize) -> Vec<String> {
    // Simple splitting strategy - in production you'd want smarter chunking
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for line in data.lines() {
        if current_chunk.len() + line.len() > max_size && !current_chunk.is_empty() {
            chunks.push(current_chunk);
            current_chunk = String::new();
        }
        current_chunk.push_str(line);
        current_chunk.push('\n');
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

pub async fn serve_website(config: BriefXAIConfig, output_dir: PathBuf) -> Result<()> {
    // Serve actual files from the directory instead of generating HTML
    let app = Router::new()
        .nest_service(
            "/",
            ServeDir::new(&output_dir).append_index_html_on_directories(true),
        )
        .nest_service("/static", ServeDir::new(output_dir.join("static")))
        .nest_service("/data", ServeDir::new(output_dir.join("data")));

    let addr = format!("127.0.0.1:{}", config.website_port);
    info!("Starting web server at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

pub async fn serve_interactive_ui(config: BriefXAIConfig, ui_dir: PathBuf) -> Result<()> {
    use axum::routing::post;

    let port = config.website_port;
    let app = Router::new()
        .nest_service("/", ServeDir::new(&ui_dir))
        .route("/api/health", get(health_endpoint))
        .route("/api/analyze", post(analyze_endpoint))
        .route("/api/upload", post(upload_endpoint))
        .route("/api/example-data", get(example_data_endpoint))
        .route("/api/example", get(example_endpoint))
        .route("/api/status", get(status_endpoint))
        .route("/api/start-server", post(start_server_endpoint))
        .route("/api/pull-model", post(pull_model_endpoint))
        .route("/api/ollama-status", get(check_ollama_status))
        .route("/api/progress/:session_id", get(progress_endpoint))
        .route("/ws/progress", get(websocket_handler))
        .route(
            "/api/session/:session_id/status",
            get(session_status_handler),
        )
        .nest_service("/static", ServeDir::new(ui_dir.join("static")))
        .nest_service("/data", ServeDir::new(ui_dir.join("data")))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(Arc::new(config));

    let addr = format!("127.0.0.1:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// API endpoints for interactive UI
use axum::extract::Json as AxumJson;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    #[serde(default)]
    conversations: Vec<crate::types::ConversationData>,
    #[serde(default)]
    data: Vec<crate::types::ConversationData>,
    #[serde(default)]
    config: AnalyzeConfig,
}

impl Default for AnalyzeConfig {
    fn default() -> Self {
        Self {
            llm_provider: default_llm_provider(),
            llm_model: default_llm_model(),
            embedding_provider: default_embedding_provider(),
            embedding_model: default_embedding_model(),
            dedup: default_dedup(),
            batch_size: default_batch_size(),
            api_key: None,
            embedding_api_key: None,
            base_url: None,
        }
    }
}

#[derive(Deserialize)]
struct AnalyzeConfig {
    #[serde(default = "default_llm_provider")]
    llm_provider: String,
    #[serde(default = "default_llm_model")]
    llm_model: String,
    #[serde(default = "default_embedding_provider")]
    embedding_provider: String,
    #[serde(default = "default_embedding_model")]
    embedding_model: String,
    #[serde(default = "default_dedup")]
    dedup: bool,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    embedding_api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
}

fn default_llm_provider() -> String {
    "openai".to_string()
}

fn default_llm_model() -> String {
    "gpt-4o-mini".to_string()
}

fn default_embedding_provider() -> String {
    "openai".to_string()
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_dedup() -> bool {
    true
}

fn default_batch_size() -> usize {
    100
}

async fn analyze_endpoint(
    State(base_config): State<Arc<BriefXAIConfig>>,
    AxumJson(request): AxumJson<AnalyzeRequest>,
) -> impl IntoResponse {
    // Generate session ID
    let session_id = uuid::Uuid::new_v4().to_string();

    // Start progress tracking
    if let Err(e) = GLOBAL_PROGRESS.start_session(session_id.clone()).await {
        warn!("Failed to start progress session: {}", e);
    }

    // Create config from request, merging with base config
    let mut config = (*base_config).clone();

    // Update config with values from request
    config.llm_provider = match request.config.llm_provider.as_str() {
        "openai" => LlmProvider::OpenAI,
        "ollama" => LlmProvider::Ollama,
        "huggingface" => LlmProvider::HuggingFace,
        "vllm" => LlmProvider::VLLM,
        "gemini" => LlmProvider::Gemini,
        _ => LlmProvider::OpenAI,
    };

    config.llm_model = request.config.llm_model;
    // Use provided API key, or fall back to environment variable
    config.llm_api_key = request
        .config
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .or_else(|| std::env::var("GEMINI_API_KEY").ok())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());
    config.llm_base_url = request.config.base_url.clone();

    config.embedding_provider = match request.config.embedding_provider.as_str() {
        "openai" => EmbeddingProvider::OpenAI,
        "sentence-transformers" => EmbeddingProvider::SentenceTransformers,
        "huggingface" => EmbeddingProvider::HuggingFace,
        "gemini" => EmbeddingProvider::Gemini,
        "cohere" => EmbeddingProvider::Cohere,
        "voyage" => EmbeddingProvider::Voyage,
        "mixedbread" => EmbeddingProvider::MixedBread,
        "jina" => EmbeddingProvider::Jina,
        "nomic" => EmbeddingProvider::Nomic,
        "onnx" => EmbeddingProvider::ONNX,
        _ => EmbeddingProvider::OpenAI,
    };

    config.embedding_model = request.config.embedding_model;
    // Use provided embedding API key, or fall back to main API key, or environment variable
    config.embedding_api_key = request
        .config
        .embedding_api_key
        .or(request.config.api_key.clone())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .or_else(|| std::env::var("GEMINI_API_KEY").ok());

    config.dedup_data = request.config.dedup;
    config.embed_batch_size = request.config.batch_size;
    config.llm_batch_size = request.config.batch_size;

    // Use either conversations or data field (conversations takes precedence)
    let data = if !request.conversations.is_empty() {
        request.conversations
    } else {
        request.data
    };

    // Check if we have data to analyze
    if data.is_empty() {
        return AxumJson(ApiResponse {
            success: false,
            data: Some(serde_json::json!({
                "session_id": session_id
            })),
            error: Some("No conversation data provided".to_string()),
        });
    }

    // Run analysis with progress updates
    let result = run_analysis_with_progress(config, data, session_id.clone()).await;

    match result {
        Ok(results) => {
            let _ = GLOBAL_PROGRESS
                .complete_session(
                    session_id.clone(),
                    "Analysis completed successfully".to_string(),
                )
                .await;
            AxumJson(ApiResponse {
                success: true,
                data: Some(serde_json::json!({
                    "session_id": session_id,
                    "results": results
                })),
                error: None,
            })
        }
        Err(e) => {
            let _ = GLOBAL_PROGRESS
                .error_session(session_id.clone(), e.to_string())
                .await;
            AxumJson(ApiResponse {
                success: false,
                data: Some(serde_json::json!({
                    "session_id": session_id
                })),
                error: Some(e.to_string()),
            })
        }
    }
}

#[derive(Deserialize)]
struct StartServerRequest {
    #[serde(rename = "serverType")]
    server_type: String,
}

#[derive(Deserialize)]
struct PullModelRequest {
    model: String,
}

async fn pull_model_endpoint(
    AxumJson(req): AxumJson<PullModelRequest>,
) -> AxumJson<ApiResponse<serde_json::Value>> {
    use std::process::Command;

    // Try to pull the model using ollama
    match Command::new("ollama").args(&["pull", &req.model]).output() {
        Ok(output) => {
            if output.status.success() {
                AxumJson(ApiResponse {
                    success: true,
                    data: Some(serde_json::json!({
                        "message": format!("Model {} pulled successfully", req.model)
                    })),
                    error: None,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                AxumJson(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Failed to pull model: {}", error)),
                })
            }
        }
        Err(e) => AxumJson(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Failed to run ollama pull: {}", e)),
        }),
    }
}

async fn check_ollama_status() -> AxumJson<ApiResponse<serde_json::Value>> {
    use std::process::Command;

    // Check if Ollama is running by trying to list models
    match Command::new("ollama").arg("list").output() {
        Ok(output) => {
            if output.status.success() {
                let models_output = String::from_utf8_lossy(&output.stdout);
                let models: Vec<String> = models_output
                    .lines()
                    .skip(1) // Skip header
                    .filter_map(|line| line.split_whitespace().next().map(|s| s.to_string()))
                    .collect();

                AxumJson(ApiResponse {
                    success: true,
                    data: Some(serde_json::json!({
                        "running": true,
                        "models": models
                    })),
                    error: None,
                })
            } else {
                AxumJson(ApiResponse {
                    success: true,
                    data: Some(serde_json::json!({
                        "running": false,
                        "models": []
                    })),
                    error: None,
                })
            }
        }
        Err(_) => AxumJson(ApiResponse {
            success: true,
            data: Some(serde_json::json!({
                "running": false,
                "installed": false,
                "models": []
            })),
            error: None,
        }),
    }
}

async fn start_server_endpoint(
    AxumJson(req): AxumJson<StartServerRequest>,
) -> AxumJson<ApiResponse<serde_json::Value>> {
    use std::process::Command;

    let result = match req.server_type.as_str() {
        "ollama" => {
            // Try to start Ollama server
            match Command::new("ollama").arg("serve").spawn() {
                Ok(_) => {
                    // Give it a moment to start
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    Ok(
                        "Ollama server started successfully. You may need to pull a model."
                            .to_string(),
                    )
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        Err(anyhow::anyhow!(
                            "Ollama is not installed on your system.\n\n                            Ollama is the easiest way to run LLMs locally.\n\n                            To install Ollama:\n                            1. Visit https://ollama.ai\n                            2. Download the installer for your system\n                            3. Run the installer (takes ~2 minutes)\n                            4. Start Ollama: ollama serve\n                            5. Pull a model: ollama pull llama3.2\n\n                            Or continue with OpenAI API (no local setup required)."
                        ))
                    } else {
                        Err(anyhow::anyhow!(
                            "Failed to start Ollama: {}. Is another instance already running?",
                            e
                        ))
                    }
                }
            }
        }
        "vllm" => {
            // Try python3 first, then python
            let python_cmd = if Command::new("python3").arg("--version").output().is_ok() {
                "python3"
            } else if Command::new("python").arg("--version").output().is_ok() {
                "python"
            } else {
                return AxumJson(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(
                        "Python not found. Please install Python 3.8 or later.".to_string(),
                    ),
                });
            };

            // Check if vLLM is installed
            match Command::new(python_cmd)
                .args(&["-c", "import vllm"])
                .output()
            {
                Ok(output) if output.status.success() => {
                    // vLLM is installed, try to start server
                    match Command::new(python_cmd)
                        .args(&[
                            "-m", "vllm.entrypoints.openai.api_server",
                            "--model", "microsoft/Phi-3.5-mini-instruct",
                            "--host", "0.0.0.0",
                            "--port", "8000",
                            "--max-model-len", "2048"
                        ])
                        .spawn()
                    {
                        Ok(_) => {
                            tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                            Ok("vLLM server starting with Phi-3.5-mini model. This may take a moment to download the model.".to_string())
                        }
                        Err(e) => {
                            Err(anyhow::anyhow!("Failed to start vLLM server: {}. Please start manually:\n\n{} -m vllm.entrypoints.openai.api_server --model microsoft/Phi-3.5-mini-instruct --host 0.0.0.0 --port 8000", e, python_cmd))
                        }
                    }
                }
                _ => {
                    // vLLM not installed - provide clear, actionable instructions
                    Err(anyhow::anyhow!(
                        "vLLM is not installed on your system.\n\n                        What is vLLM?\n                        vLLM is an optional high-performance inference server for large language models.\n                        It provides fast local LLM inference but requires a CUDA-capable GPU.\n\n                        To install and use vLLM:\n\n                        1. Check requirements:\n   \n                           - Python 3.8 or later (you have {})
   \n                           - CUDA-capable GPU (NVIDIA)
   \n                           - 8GB+ GPU memory recommended\n\n                        2. Install vLLM:\n   \n                           {} -m pip install vllm\n\n                        3. Start the server:\n   \n                           {} -m vllm.entrypoints.openai.api_server \\n     \n                           --model microsoft/Phi-3.5-mini-instruct \\n     \n                           --host 0.0.0.0 --port 8000\n\n                        Alternatives:\n                        - For CPU-only systems: Use Ollama (easier setup)\n\n                        - For cloud usage: Use OpenAI API (no local setup needed)\n\n                        - Both options are available in the dropdown above\n\n                        You can continue without vLLM by selecting a different provider.",
                        python_cmd, python_cmd, python_cmd
                    ))
                }
            }
        }
        _ => Err(anyhow::anyhow!("Unknown server type: {}", req.server_type)),
    };

    match result {
        Ok(message) => AxumJson(ApiResponse {
            success: true,
            data: Some(serde_json::json!({ "message": message })),
            error: None,
        }),
        Err(e) => AxumJson(ApiResponse {
            success: false,
            data: None,
            error: Some(e.to_string()),
        }),
    }
}

async fn run_analysis_with_progress(
    config: BriefXAIConfig,
    data: Vec<crate::types::ConversationData>,
    session_id: String,
) -> Result<serde_json::Value> {
    // Enhanced progress tracking with more detail

    // Step 1: Data validation and preprocessing (0-15%)
    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "validation",
            2.0,
            "Starting data validation",
            Some(format!(
                "Checking {} conversations for validity",
                data.len()
            )),
        )
        .await;

    if data.is_empty() {
        return Err(anyhow::anyhow!("No conversations provided"));
    }

    // Calculate statistics
    let total_messages: usize = data.iter().map(|c| c.len()).sum();
    let avg_messages = total_messages / data.len();
    let max_messages = data.iter().map(|c| c.len()).max().unwrap_or(0);
    let min_messages = data.iter().map(|c| c.len()).min().unwrap_or(0);

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "validation",
            8.0,
            "Analyzing conversation structure",
            Some(format!(
                "Messages per conversation: min={}, avg={}, max={}",
                min_messages, avg_messages, max_messages
            )),
        )
        .await;

    // Deduplication if enabled
    let processed_data = if config.dedup_data {
        let _ = GLOBAL_PROGRESS
            .update_progress(
                &session_id,
                "dedup",
                10.0,
                "Deduplicating conversations",
                Some("Removing duplicate entries".to_string()),
            )
            .await;

        let deduped = crate::utils::dedup_data(data.clone())?;
        let removed = data.len() - deduped.len();

        let _ = GLOBAL_PROGRESS
            .update_progress(
                &session_id,
                "dedup",
                15.0,
                "Deduplication complete",
                Some(if removed > 0 {
                    format!("Removed {} duplicate conversations", removed)
                } else {
                    "No duplicates found".to_string()
                }),
            )
            .await;

        deduped
    } else {
        let _ = GLOBAL_PROGRESS
            .update_progress(
                &session_id,
                "validation",
                15.0,
                "Validation complete",
                Some(format!(
                    "{} unique conversations ready for analysis",
                    data.len()
                )),
            )
            .await;
        data
    };

    // Step 2: Facet extraction (15-35%)
    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "facets",
            18.0,
            "Initializing facet extraction",
            Some(format!(
                "LLM Provider: {} ({})",
                match config.llm_provider {
                    crate::config::LlmProvider::OpenAI => "OpenAI",
                    crate::config::LlmProvider::Ollama => "Ollama",
                    crate::config::LlmProvider::HuggingFace => "HuggingFace",
                    crate::config::LlmProvider::VLLM => "vLLM",
                    crate::config::LlmProvider::Gemini => "Google Gemini",
                },
                config.llm_model
            )),
        )
        .await;

    // Process in smaller batches for better feedback and to avoid rate limits
    let batch_size = 3.min(processed_data.len()); // Reduced batch size for stability
    let total_batches = (processed_data.len() + batch_size - 1) / batch_size;
    let mut all_facets = Vec::new();

    for (i, chunk) in processed_data.chunks(batch_size).enumerate() {
        let progress = 20.0 + (i as f32 / total_batches as f32 * 12.0);

        let _ = GLOBAL_PROGRESS
            .update_progress(
                &session_id,
                "facets",
                progress,
                "Extracting conversation facets",
                Some(format!(
                    "Analyzing batch {}/{} ({} conversations)",
                    i + 1,
                    total_batches,
                    chunk.len()
                )),
            )
            .await;

        // Try to extract facets with error handling
        match crate::facets::extract_facets(&config, chunk).await {
            Ok(batch_facets) => {
                all_facets.extend(batch_facets);
            }
            Err(e) => {
                // Log the error but try to continue
                warn!(
                    "Error in batch {}: {}. Retrying with smaller batch...",
                    i + 1,
                    e
                );

                // Try processing one by one if batch fails
                for (j, single_conv) in chunk.iter().enumerate() {
                    let _ = GLOBAL_PROGRESS
                        .update_progress(
                            &session_id,
                            "facets",
                            progress,
                            "Extracting conversation facets (retry)",
                            Some(format!(
                                "Processing conversation {} of batch {}",
                                j + 1,
                                i + 1
                            )),
                        )
                        .await;

                    match crate::facets::extract_facets(&config, &[single_conv.clone()]).await {
                        Ok(single_facet) => all_facets.extend(single_facet),
                        Err(e) => {
                            warn!("Failed to process conversation in batch {}: {}", i + 1, e);
                            // Add empty facets for failed conversation
                            all_facets.push(vec![]);
                        }
                    }

                    // Small delay between individual requests
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                }
            }
        }

        // Add delay between batches to respect rate limits
        if i < total_batches - 1 {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    let facet_data = all_facets;
    let facet_count = facet_data.first().map(|f| f.len()).unwrap_or(0);

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "facets",
            35.0,
            "Facet extraction complete",
            Some(format!(
                "Extracted {} facet categories from conversations",
                facet_count
            )),
        )
        .await;

    // Step 3: Embedding generation (35-55%)
    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "embeddings",
            38.0,
            "Initializing embedding generation",
            Some(format!(
                "Embedding Provider: {} ({})",
                config.embedding_provider, config.embedding_model
            )),
        )
        .await;

    // Generate embeddings with progress
    let embed_batch_size = 10.min(processed_data.len());
    let total_embed_batches = (processed_data.len() + embed_batch_size - 1) / embed_batch_size;
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

    for (i, chunk) in processed_data.chunks(embed_batch_size).enumerate() {
        let progress = 40.0 + (i as f32 / total_embed_batches as f32 * 12.0);

        let _ = GLOBAL_PROGRESS
            .update_progress(
                &session_id,
                "embeddings",
                progress,
                "Generating conversation embeddings",
                Some(format!(
                    "Processing batch {}/{} ({} items)",
                    i + 1,
                    total_embed_batches,
                    chunk.len()
                )),
            )
            .await;

        let batch_embeddings = crate::embeddings::generate_embeddings(&config, chunk).await?;
        all_embeddings.extend(batch_embeddings);
    }

    let embeddings = all_embeddings;
    let embedding_dim = embeddings.first().map(|e| e.len()).unwrap_or(0);

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "embeddings",
            55.0,
            "Embedding generation complete",
            Some(format!(
                "Generated {} embeddings ({} dimensions)",
                embeddings.len(),
                embedding_dim
            )),
        )
        .await;

    // Step 4: Clustering (55-70%)
    let n_clusters = config.n_base_clusters(processed_data.len());

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "clustering",
            58.0,
            "Initializing K-means clustering",
            Some(format!(
                "Computing {} clusters for {} conversations",
                n_clusters,
                processed_data.len()
            )),
        )
        .await;

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "clustering",
            62.0,
            "Running clustering algorithm",
            Some("Finding optimal cluster centers...".to_string()),
        )
        .await;

    let clusters =
        crate::clustering::create_base_clusters(&config, &embeddings, &facet_data).await?;

    let cluster_sizes: Vec<usize> = vec![processed_data.len() / clusters.len(); clusters.len()];
    let avg_size = cluster_sizes.iter().sum::<usize>() / cluster_sizes.len();

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "clustering",
            70.0,
            "Clustering complete",
            Some(format!(
                "Created {} clusters (avg size: {} conversations)",
                clusters.len(),
                avg_size
            )),
        )
        .await;

    // Step 5: Building hierarchy (70-80%)
    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "hierarchy",
            72.0,
            "Building cluster hierarchy",
            Some(format!("Organizing {} base clusters", clusters.len())),
        )
        .await;

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "hierarchy",
            75.0,
            "Creating hierarchical relationships",
            Some("Analyzing cluster similarities...".to_string()),
        )
        .await;

    let hierarchy = crate::clustering::build_hierarchy(&config, clusters).await?;

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "hierarchy",
            80.0,
            "Hierarchy complete",
            Some(format!(
                "Built hierarchical structure with {} top-level clusters",
                hierarchy.len()
            )),
        )
        .await;

    // Step 6: UMAP visualization (80-95%)
    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "umap",
            82.0,
            "Starting dimensionality reduction",
            Some(format!(
                "Reducing {}-dimensional embeddings to 2D",
                embedding_dim
            )),
        )
        .await;

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "umap",
            88.0,
            "Computing UMAP projection",
            Some("Optimizing 2D layout for visualization...".to_string()),
        )
        .await;

    let umap_coords = crate::umap::generate_umap(&config, &embeddings).await?;

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "umap",
            95.0,
            "UMAP projection complete",
            Some(format!(
                "Projected {} points to 2D visualization space",
                umap_coords.len()
            )),
        )
        .await;

    // Step 7: Finalizing results (95-100%)
    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "finalize",
            97.0,
            "Preparing final results",
            Some("Organizing data for visualization...".to_string()),
        )
        .await;

    // Prepare results
    let results = serde_json::json!({
        "conversations": processed_data,
        "facets": facet_data,
        "hierarchy": hierarchy,
        "umap": umap_coords,
        "stats": {
            "total_conversations": processed_data.len(),
            "total_clusters": hierarchy.len(),
            "embedding_dimension": embedding_dim,
            "total_messages": total_messages,
            "avg_messages_per_conversation": avg_messages
        }
    });

    let _ = GLOBAL_PROGRESS
        .update_progress(
            &session_id,
            "complete",
            100.0,
            "Analysis complete!",
            Some(format!(
                "Successfully analyzed {} conversations",
                processed_data.len()
            )),
        )
        .await;

    Ok(results)
}

async fn upload_endpoint(mut multipart: Multipart) -> impl IntoResponse {
    let mut uploaded_files = Vec::new();
    let mut conversation_data = Vec::new();
    let mut errors = Vec::new();

    // Process each part of the multipart upload
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        // Get field name and filename
        let name = field.name().unwrap_or("unknown").to_string();
        let file_name = field.file_name().unwrap_or("unknown").to_string();

        debug!("Processing upload field: {} ({})", name, file_name);

        // Read the field data
        match field.bytes().await {
            Ok(bytes) => {
                // Try to parse as JSON
                match serde_json::from_slice::<Vec<crate::types::ConversationData>>(&bytes) {
                    Ok(data) => {
                        info!(
                            "Successfully parsed {} conversations from {}",
                            data.len(),
                            file_name
                        );
                        conversation_data.extend(data);
                        uploaded_files.push(FileUploadInfo {
                            name: file_name.clone(),
                            size: bytes.len(),
                            mime_type: "application/json".to_string(),
                            conversations: conversation_data.len(),
                        });
                    }
                    Err(e) => {
                        // Try parsing as single conversation
                        match serde_json::from_slice::<crate::types::ConversationData>(&bytes) {
                            Ok(data) => {
                                info!("Successfully parsed single conversation from {}", file_name);
                                conversation_data.push(data);
                                uploaded_files.push(FileUploadInfo {
                                    name: file_name.clone(),
                                    size: bytes.len(),
                                    mime_type: "application/json".to_string(),
                                    conversations: 1,
                                });
                            }
                            Err(_) => {
                                // Try as CSV
                                match parse_csv_conversations(&bytes) {
                                    Ok(data) => {
                                        info!(
                                            "Successfully parsed {} conversations from CSV {}",
                                            data.len(),
                                            file_name
                                        );
                                        conversation_data.extend(data);
                                        uploaded_files.push(FileUploadInfo {
                                            name: file_name.clone(),
                                            size: bytes.len(),
                                            mime_type: "text/csv".to_string(),
                                            conversations: conversation_data.len(),
                                        });
                                    }
                                    Err(_) => {
                                        // Try as plain text conversation
                                        match parse_text_conversation(&bytes) {
                                            Ok(data) => {
                                                info!(
                                                    "Successfully parsed text conversation from {}",
                                                    file_name
                                                );
                                                conversation_data.push(data);
                                                uploaded_files.push(FileUploadInfo {
                                                    name: file_name.clone(),
                                                    size: bytes.len(),
                                                    mime_type: "text/plain".to_string(),
                                                    conversations: 1,
                                                });
                                            }
                                            Err(_) => {
                                                let error_msg = format!(
                                                    "Could not parse file '{}': {}",
                                                    file_name, e
                                                );
                                                warn!("{}", error_msg);
                                                errors.push(error_msg);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to read file '{}': {}", file_name, e);
                warn!("{}", error_msg);
                errors.push(error_msg);
            }
        }
    }

    // Prepare response
    if !conversation_data.is_empty() {
        // Store uploaded data temporarily (in production, use proper session storage)
        let session_id = uuid::Uuid::new_v4().to_string();

        AxumJson(ApiResponse {
            success: true,
            data: Some(serde_json::json!({
                "session_id": session_id,
                "files": uploaded_files,
                "conversations": conversation_data,
                "total_conversations": conversation_data.len(),
                "warnings": errors,
            })),
            error: None,
        })
    } else if !errors.is_empty() {
        AxumJson(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Upload failed: {}", errors.join("; "))),
        })
    } else {
        AxumJson(ApiResponse {
            success: false,
            data: None,
            error: Some("No files were uploaded".to_string()),
        })
    }
}

#[derive(Serialize)]
struct FileUploadInfo {
    name: String,
    size: usize,
    mime_type: String,
    conversations: usize,
}

// Parse CSV format conversations
fn parse_csv_conversations(bytes: &[u8]) -> Result<Vec<crate::types::ConversationData>> {
    let content = std::str::from_utf8(bytes)?;
    let mut conversations = Vec::new();
    let mut current_conversation = Vec::new();
    let mut current_id = String::new();

    for line in content.lines().skip(1) {
        // Skip header if exists
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        if parts.len() >= 3 {
            let conv_id = parts[0].to_string();
            let role = parts[1].trim_matches('"').to_string();
            let content = parts[2..].join(",").trim_matches('"').to_string();

            // Check if we're starting a new conversation
            if !current_id.is_empty() && conv_id != current_id {
                if !current_conversation.is_empty() {
                    conversations.push(current_conversation);
                    current_conversation = Vec::new();
                }
            }

            current_id = conv_id;
            current_conversation.push(crate::types::Message { role, content });
        }
    }

    // Don't forget the last conversation
    if !current_conversation.is_empty() {
        conversations.push(current_conversation);
    }

    if conversations.is_empty() {
        Err(anyhow::anyhow!("No valid conversations found in CSV"))
    } else {
        // Convert Vec<Message> to ConversationData
        let conv_data: Vec<crate::types::ConversationData> = conversations
            .into_iter()
            .map(|messages| crate::types::ConversationData {
                messages,
                metadata: std::collections::HashMap::new(),
            })
            .collect();
        Ok(conv_data)
    }
}

// Parse plain text conversation format
fn parse_text_conversation(bytes: &[u8]) -> Result<crate::types::ConversationData> {
    let content = std::str::from_utf8(bytes)?;
    let mut messages = Vec::new();
    let mut current_role = String::new();
    let mut current_content = String::new();

    for line in content.lines() {
        // Look for role markers
        if line.starts_with("User:") || line.starts_with("user:") || line.starts_with("Human:") {
            if !current_content.is_empty() {
                messages.push(crate::types::Message {
                    role: current_role,
                    content: current_content.trim().to_string(),
                });
            }
            current_role = "user".to_string();
            current_content = line
                .split_once(':')
                .map(|(_, c)| c.trim().to_string())
                .unwrap_or_default();
        } else if line.starts_with("Assistant:")
            || line.starts_with("assistant:")
            || line.starts_with("AI:")
        {
            if !current_content.is_empty() {
                messages.push(crate::types::Message {
                    role: current_role,
                    content: current_content.trim().to_string(),
                });
            }
            current_role = "assistant".to_string();
            current_content = line
                .split_once(':')
                .map(|(_, c)| c.trim().to_string())
                .unwrap_or_default();
        } else if !line.trim().is_empty() {
            // Continue current message
            if !current_content.is_empty() {
                current_content.push('\n');
            }
            current_content.push_str(line);
        }
    }

    // Don't forget the last message
    if !current_content.is_empty() {
        messages.push(crate::types::Message {
            role: current_role,
            content: current_content.trim().to_string(),
        });
    }

    if messages.is_empty() {
        Err(anyhow::anyhow!("No valid messages found in text"))
    } else {
        Ok(crate::types::ConversationData {
            messages,
            metadata: std::collections::HashMap::new(),
        })
    }
}

async fn health_endpoint() -> impl IntoResponse {
    "OK"
}

async fn example_data_endpoint() -> AxumJson<ApiResponse<Vec<crate::types::ConversationData>>> {
    let examples = generate_example_conversations(10);
    AxumJson(ApiResponse {
        success: true,
        data: Some(examples),
        error: None,
    })
}

async fn progress_endpoint(
    AxumPath(session_id): AxumPath<String>,
) -> AxumJson<ApiResponse<serde_json::Value>> {
    let status = GLOBAL_PROGRESS
        .sessions
        .read()
        .await
        .get(&session_id)
        .cloned();

    match status {
        Some(status) => {
            // Create a progress response based on status
            let (overall_progress, current_stage) = match status.as_str() {
                "started" => (0.0, "Initializing"),
                "validation" => (10.0, "Validating data"),
                "dedup" => (15.0, "Deduplicating"),
                "facets" => (30.0, "Extracting facets"),
                "embeddings" => (50.0, "Generating embeddings"),
                "clustering" => (70.0, "Clustering"),
                "hierarchy" => (85.0, "Building hierarchy"),
                "umap" => (95.0, "Creating visualization"),
                "completed" => (100.0, "Complete"),
                _ => (0.0, status.as_str()),
            };

            AxumJson(ApiResponse {
                success: true,
                data: Some(serde_json::json!({
                    "status": status,
                    "overall_progress": overall_progress,
                    "current_stage": current_stage,
                    "message": format!("Processing: {}", current_stage),
                })),
                error: None,
            })
        }
        None => AxumJson(ApiResponse {
            success: false,
            data: None,
            error: Some("Session not found".to_string()),
        }),
    }
}

async fn example_endpoint() -> impl IntoResponse {
    let example_data = generate_example_conversations(100);
    AxumJson(ApiResponse {
        success: true,
        data: Some(example_data),
        error: None,
    })
}

async fn status_endpoint() -> impl IntoResponse {
    AxumJson(ApiResponse {
        success: true,
        data: Some("BriefXAI server is running"),
        error: None,
    })
}

fn generate_example_conversations(count: usize) -> Vec<crate::types::ConversationData> {
    crate::examples::generate_realistic_conversations(count, Some(42))
}

// WebSocket handler for real-time progress updates
async fn websocket_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_websocket)
}

async fn handle_websocket(socket: WebSocket) {
    let mut socket = socket;
    let mut progress_rx = GLOBAL_PROGRESS.subscribe();

    info!("WebSocket connection established for progress updates");

    // Send initial connection confirmation
    let welcome_msg = ProgressMessage::Update(ProgressUpdate {
        step: "connected".to_string(),
        progress: 0.0,
        message: "Connected to progress stream".to_string(),
        details: None,
        timestamp: chrono::Utc::now(),
    });

    if let Ok(json) = serde_json::to_string(&welcome_msg) {
        if socket.send(Message::Text(json)).await.is_err() {
            debug!("Failed to send welcome message");
            return;
        }
    }

    // Handle incoming messages and outgoing progress updates
    loop {
        tokio::select! {
            // Handle incoming WebSocket messages
            msg = socket.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        debug!("Received WebSocket message: {}", text);
                        // Handle client messages if needed (ping, session subscription, etc.)
                        if text == "ping" {
                            if socket.send(Message::Text("pong".to_string())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        debug!("WebSocket closed by client");
                        break;
                    }
                    Some(Err(e)) => {
                        warn!("WebSocket error: {}", e);
                        break;
                    }
                    None => break,
                    _ => {} // Ignore other message types
                }
            }

            // Handle progress updates
            progress_msg = progress_rx.recv() => {
                match progress_msg {
                    Ok(msg) => {
                        if let Ok(json) = serde_json::to_string(&msg) {
                            if socket.send(Message::Text(json)).await.is_err() {
                                debug!("Failed to send progress update, client likely disconnected");
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!("Progress receiver lagged, skipped {} messages", skipped);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("Progress broadcast channel closed");
                        break;
                    }
                }
            }
        }
    }

    info!("WebSocket connection closed");
}

// Session status endpoint
async fn session_status_handler(AxumPath(session_id): AxumPath<String>) -> impl IntoResponse {
    match GLOBAL_PROGRESS.get_session_status(&session_id).await {
        Some(status) => AxumJson(ApiResponse {
            success: true,
            data: Some(serde_json::json!({
                "session_id": session_id,
                "status": status
            })),
            error: None,
        }),
        None => AxumJson(ApiResponse {
            success: false,
            data: None,
            error: Some("Session not found".to_string()),
        }),
    }
}
