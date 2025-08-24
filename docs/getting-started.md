# Getting Started with BriefXAI

BriefXAI is an advanced conversation analysis platform that automatically extracts insights from your conversation data. This guide will help you get up and running quickly.

## What's New ✨

- **Smart Model Auto-Detection**: Automatically selects the most cost-effective model based on your content
- **Google Gemini Support**: Full integration with Google's Gemini models
- **Unified Dashboard**: Single-page interface with Overview, Clusters, Visualization, and Export tabs
- **OCR Support**: Upload PDFs, DOCX, and TXT files for automatic text extraction
- **Cost Optimization**: Built-in features to minimize cloud API costs

## Prerequisites

Before you begin, ensure you have the following installed:

- **Rust** 1.70 or higher ([Install Rust](https://rustup.rs/))
- **Git** for cloning the repository
- **A text editor** for editing configuration files
- **API Keys** (optional - for cloud models)

## Installation

### Option 1: Build from Source (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/briefcasebrain/briefxai.git
   cd briefxai
   ```

2. **Build the project:**
   ```bash
   cargo build --release
   ```

3. **Run the tests to verify installation:**
   ```bash
   cargo test
   ```

4. **Install globally (optional):**
   ```bash
   cargo install --path .
   ```

### Option 2: Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t briefxai:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 \
     -v $(pwd)/data:/data \
     -v $(pwd)/config.toml:/config.toml \
     briefxai:latest
   ```

### Option 3: Pre-built Binaries

Download the latest release from the [releases page](https://github.com/briefcasebrain/briefxai/releases).

## Configuration

### Quick Setup with .env File

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your API keys:
   ```bash
   # API Keys (Optional - for premium features)
   OPENAI_API_KEY=sk-your-key-here
   GOOGLE_API_KEY=your-gemini-key-here
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   
   # Server Configuration
   PORT=8080
   
   # Feature Flags
   ENABLE_DEMO_MODE=true
   MAX_CONVERSATIONS=100
   MAX_MESSAGES_PER_CONVERSATION=50
   
   # Privacy Settings
   MIN_PRIVACY_THRESHOLD=3
   ```

3. **No API Keys? No Problem!**
   - Leave API keys blank to use demo mode
   - Use local models like Ollama (see below)
   - Try the built-in example data

### Advanced Configuration (Optional)

For advanced users, create a `config.toml` file:

```toml
# LLM Settings
llm_provider = "gemini"  # openai, gemini, ollama, huggingface, vllm
llm_model = "gemini-1.5-flash"
embedding_provider = "google"
embedding_model = "text-embedding-004"

# Cost Control
max_cost_per_run = 5.0
rate_limit_requests_per_minute = 10
batch_size = 3

# Server Settings  
website_port = 8080
verbose = true
```

### Provider Setup

#### Google Gemini (Recommended - Most Cost-Effective)

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a free API key
3. Add to your `.env` file:
   ```bash
   GOOGLE_API_KEY=your-gemini-key-here
   ```

#### OpenAI

1. Sign up for an [OpenAI account](https://platform.openai.com/)
2. Generate an API key
3. Add to your `.env` file:
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

#### Ollama (Local - No API Keys Required)

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Pull a model:
   ```bash
   ollama pull llama3.2:3b  # or llama3.1:8b for better quality
   ```

3. Ollama will be auto-detected when you run BriefXAI

## Running BriefXAI

### Web Interface (Recommended)

1. **Start the server:**
   ```bash
   ./target/release/briefxai ui --port 8080
   # Or if globally installed:
   briefxai ui
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8080`

3. **Choose your data source:**
   - **Upload Files**: JSON, CSV, PDF, TXT, DOCX (with OCR)
   - **Use Example Data**: Try with sample conversations
   - **Paste Data**: Directly paste conversation data

4. **Smart Model Selection:**
   - Click "🎯 Auto-Detect" buttons for automatic model selection
   - Or manually choose your preferred models
   - Cost estimates are shown automatically

5. **Unified Dashboard:**
   - **Overview**: Key metrics and insights
   - **Clusters**: Hierarchical conversation groupings
   - **Visualization**: Interactive UMAP plots
   - **Browse**: Search and filter conversations
   - **Export**: Download results in various formats

### Command Line Interface

#### Analyze a file:
```bash
briefxai analyze conversations.json \
  --template customer_support \
  --output results.json
```

#### Resume a paused session:
```bash
briefxai resume session_abc123
```

#### Export results:
```bash
briefxai export session_abc123 \
  --format csv \
  --output analysis_results.csv
```

#### List sessions:
```bash
briefxai list-sessions
```

## Your First Analysis

### Step 1: Prepare Your Data

Create a file named `sample_conversations.json`:

```json
[
  {
    "conversation_id": "conv_001",
    "messages": [
      {
        "role": "user",
        "content": "I'm having trouble with my subscription"
      },
      {
        "role": "assistant",
        "content": "I'd be happy to help with your subscription issue. Can you tell me more about what's happening?"
      },
      {
        "role": "user",
        "content": "I was charged twice this month"
      },
      {
        "role": "assistant",
        "content": "I apologize for the double charge. Let me look into that for you right away."
      }
    ],
    "metadata": {
      "timestamp": "2024-01-15T10:30:00Z",
      "customer_id": "cust_123"
    }
  }
]
```

### Step 2: Run the Analysis

Using the CLI:
```bash
briefxai analyze sample_conversations.json \
  --template customer_support \
  --output analysis_results.json
```

Using the Web UI:
1. Navigate to `http://localhost:8080`
2. Click "New Analysis"
3. Upload `sample_conversations.json`
4. Select "Customer Support" template
5. Click "Start Analysis"

### Step 3: View Results

The analysis will extract:
- **Sentiments**: Customer satisfaction levels
- **Topics**: Main discussion themes
- **Issues**: Problems identified
- **Resolutions**: How issues were resolved
- **Patterns**: Common trends across conversations

Results are available in:
- JSON format for programmatic access
- CSV format for spreadsheet analysis
- Web dashboard for interactive exploration

## Common Use Cases

### Customer Support Analysis

```bash
briefxai analyze support_tickets.json \
  --template customer_support \
  --facets "sentiment,issues,resolution_status" \
  --output support_insights.json
```

### Sales Conversation Analysis

```bash
briefxai analyze sales_calls.json \
  --template sales \
  --facets "opportunities,objections,next_steps" \
  --output sales_analysis.json
```

### Medical Consultation Review

```bash
briefxai analyze consultations.json \
  --template medical \
  --facets "symptoms,diagnoses,treatments" \
  --pii-detection strict \
  --output medical_insights.json
```

## Troubleshooting

### Common Issues

#### Server won't start
- Check if port 8080 is already in use
- Verify configuration file syntax
- Ensure database directory exists and is writable

#### Analysis fails immediately
- Verify API keys are correct
- Check network connectivity
- Ensure input file format is valid JSON

#### Out of memory errors
- Reduce batch_size in configuration
- Process conversations in smaller chunks
- Increase system memory allocation

### Getting Help

1. Check the [FAQ](faq.md)
2. Search [existing issues](https://github.com/briefcasebrain/briefxai/issues)
3. Ask in [discussions](https://github.com/briefcasebrain/briefxai/discussions)
4. Contact support

## Next Steps

- Read the [Architecture Overview](architecture.md) to understand the system design
- Check the [API Reference](api.md) for programmatic access
- See [Configuration Guide](configuration.md) for advanced settings
- Review [Development Guide](development.md) to contribute

## Quick Reference

### Essential Commands

```bash
# Start server
briefxai serve

# Analyze file
briefxai analyze <file> --template <template>

# Resume session
briefxai resume <session_id>

# Export results
briefxai export <session_id> --format <format>

# List sessions
briefxai list-sessions

# Show help
briefxai --help
```

### Configuration Locations

1. `./config.toml` (current directory)
2. `~/.config/briefxai/config.toml` (user config)
3. `/etc/briefxai/config.toml` (system-wide)
4. Environment variables (highest priority)

### File Formats

- **Input**: JSON array of conversations
- **Output**: JSON, CSV, or HTML reports
- **Config**: TOML configuration files
- **Database**: SQLite database files