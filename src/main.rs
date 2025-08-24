use anyhow::Result;
use briefx::{types::ConversationData, BriefXAIConfig};
use clap::{Parser, Subcommand};
use serde_json;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{info, Level};
use tracing_subscriber;

#[derive(Parser)]
#[clap(name = "briefxai")]
#[clap(about = "BriefXAI - Advanced AI conversation analysis platform")]
#[clap(version)]
struct Cli {
    /// Start in UI mode (default if no command provided)
    #[clap(subcommand)]
    command: Option<Commands>,

    /// Port for web interface
    #[clap(short, long, global = true, default_value = "8080")]
    port: u16,

    /// Enable verbose output
    #[clap(short, long, global = true)]
    verbose: bool,

    /// Configuration file path
    #[clap(short, long, global = true)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the interactive web UI (default)
    UI {
        /// Port to serve on
        #[clap(short, long, default_value = "8080")]
        port: u16,

        /// Automatically open browser
        #[clap(long)]
        open: bool,
    },

    /// Analyze from command line (advanced)
    Analyze {
        /// Input data file
        #[clap(short, long)]
        input: PathBuf,

        /// Output directory
        #[clap(short, long, default_value = "output")]
        output: PathBuf,
    },

    /// Serve existing results
    Serve {
        /// Directory containing results
        #[clap(short, long, default_value = "output")]
        directory: PathBuf,
    },

    /// Generate realistic example data
    Example {
        /// Output file path
        #[clap(short, long, default_value = "realistic_conversations.json")]
        output: PathBuf,

        /// Number of conversations to generate
        #[clap(short = 'n', long, default_value = "50")]
        count: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if it exists
    dotenv::dotenv().ok();

    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    // Load configuration
    let config = if let Some(config_path) = cli.config {
        load_config(&config_path).await?
    } else {
        BriefXAIConfig::default()
    };

    // If no command specified, default to UI mode
    match cli.command {
        None | Some(Commands::UI { .. }) => {
            let port = if let Some(Commands::UI { port, .. }) = cli.command {
                port
            } else {
                cli.port
            };

            let open_browser = if let Some(Commands::UI { open, .. }) = cli.command {
                open
            } else {
                true // Default to opening browser
            };

            start_ui_mode(config, port, open_browser).await?;
        }

        Some(Commands::Analyze { input, output }) => {
            analyze_from_cli(config, input, output).await?;
        }

        Some(Commands::Serve { directory }) => {
            serve_existing(config, directory, cli.port).await?;
        }

        Some(Commands::Example { output, count }) => {
            example_command(output, count).await?;
        }
    }

    Ok(())
}

async fn start_ui_mode(mut config: BriefXAIConfig, port: u16, open_browser: bool) -> Result<()> {
    info!("Starting BriefXAI Web Interface");

    config.website_port = port;

    // Create a temporary directory for the UI
    let ui_dir = PathBuf::from("briefxai_ui_data");
    fs::create_dir_all(&ui_dir).await?;

    // Generate the interactive UI
    generate_interactive_ui(&ui_dir).await?;

    let url = format!("http://localhost:{}", port);

    info!("BriefXAI is ready!");
    info!("Open your browser to: {}", url);
    info!("   - Upload conversations or use example data");
    info!("   - Configure analysis settings");
    info!("   - View real-time results");
    info!("");
    info!("Press Ctrl+C to stop the server");

    // Open browser if requested
    if open_browser {
        if let Err(e) = webbrowser::open(&url) {
            info!("Could not open browser automatically: {}", e);
        }
    }

    // Start the web server
    briefx::web::serve_interactive_ui(config, ui_dir).await?;

    Ok(())
}

async fn generate_interactive_ui(output_dir: &PathBuf) -> Result<()> {
    let static_dir = output_dir.join("static");
    fs::create_dir_all(&static_dir).await?;

    // Use the clean embedded UI
    let index_html = include_str!("../assets/clean_ui.html");
    fs::write(output_dir.join("index.html"), index_html).await?;

    Ok(())
}

async fn analyze_from_cli(config: BriefXAIConfig, input: PathBuf, output: PathBuf) -> Result<()> {
    info!("Running analysis from CLI");

    let data = load_input_data(&input).await?;
    info!("Loaded {} conversations", data.len());

    // Run the analysis using the session manager
    use briefx::analysis::session_manager::AnalysisSessionManager;
    use briefx::persistence_v2::EnhancedPersistenceLayer;
    use std::sync::Arc;

    // Create persistence layer
    let persistence = Arc::new(EnhancedPersistenceLayer::new(output.clone()).await?);

    // Create session manager
    let session_manager = AnalysisSessionManager::new(persistence);

    // Start the analysis
    let (session_id, mut event_receiver) = session_manager.start_analysis(config, data).await?;

    info!("Started analysis with session ID: {}", session_id);

    // Wait for completion by listening to events
    while let Ok(event) = event_receiver.recv().await {
        use briefx::analysis::session_manager::SessionEvent;
        match event {
            SessionEvent::Completed { session_id: _ } => {
                info!("Analysis completed successfully");
                break;
            }
            SessionEvent::Failed {
                session_id: _,
                error,
            } => {
                return Err(anyhow::anyhow!("Analysis failed: {}", error));
            }
            SessionEvent::ProgressUpdate {
                session_id: _,
                stage,
                progress,
                message,
            } => {
                info!("Progress [{}]: {}% - {}", stage, progress, message);
            }
            SessionEvent::BatchStarted {
                session_id: _,
                batch_number,
                total_batches,
            } => {
                info!("Processing batch {} of {}", batch_number, total_batches);
            }
            SessionEvent::BatchCompleted {
                session_id: _,
                batch_number,
            } => {
                info!("Completed batch {}", batch_number);
            }
            _ => {}
        }
    }

    // Save completion marker
    tokio::fs::create_dir_all(&output).await?;
    let marker_path = output.join("analysis_complete.json");
    let marker = serde_json::json!({
        "session_id": session_id,
        "status": "completed",
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    tokio::fs::write(&marker_path, serde_json::to_string_pretty(&marker)?).await?;

    info!("Analysis complete! Results saved to {:?}", output);
    info!("Run 'briefxai serve -d {:?}' to view results", output);

    Ok(())
}

async fn serve_existing(config: BriefXAIConfig, directory: PathBuf, port: u16) -> Result<()> {
    info!("Serving existing results from {:?}", directory);
    let mut config = config;
    config.website_port = port;

    // Use serve_interactive_ui for full API support
    briefx::web::serve_interactive_ui(config, directory).await?;
    Ok(())
}

async fn load_input_data(path: &PathBuf) -> Result<Vec<ConversationData>> {
    let content = fs::read_to_string(path).await?;

    if path.extension().and_then(|s| s.to_str()) == Some("json") {
        let data: Vec<ConversationData> = serde_json::from_str(&content)?;
        Ok(data)
    } else {
        anyhow::bail!("Unsupported input format")
    }
}

async fn load_config(path: &PathBuf) -> Result<BriefXAIConfig> {
    let content = fs::read_to_string(path).await?;
    let config: BriefXAIConfig = serde_json::from_str(&content)?;
    Ok(config)
}

async fn example_command(output: PathBuf, count: usize) -> Result<()> {
    info!("Generating {} realistic example conversations", count);

    let conversations = briefx::examples::generate_realistic_conversations(count, Some(42));

    let json = serde_json::to_string_pretty(&conversations)?;
    fs::write(&output, json).await?;

    info!("Generated {} diverse conversations covering:", count);
    info!("  • Technical support");
    info!("  • Creative writing");
    info!("  • Cooking advice");
    info!("  • Health & fitness");
    info!("  • Language learning");
    info!("  • Career advice");
    info!("  • Personal finance");
    info!("  • Relationships");
    info!("  • Academic help");
    info!("  • Travel planning");
    info!("");
    info!("Example data written to {:?}", output);
    Ok(())
}
