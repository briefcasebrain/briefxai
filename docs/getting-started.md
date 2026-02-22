# Getting Started with BriefX

This guide will help you get BriefX up and running on your system.

## Prerequisites

Before you begin, ensure you have:

- **Python** 3.9 or higher
- **pip** for installing Python packages
- **Git** for cloning the repository
- An API key from at least one supported LLM provider (OpenAI, Anthropic, Google Gemini), or a local [Ollama](https://ollama.com) installation

## Installation

### Option 1: Standard Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/briefcasebrain/briefxai.git
   cd briefxai/python
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation:**
   ```bash
   python cli_simple.py --help
   ```

### Option 2: Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t briefxai:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 \
     -e OPENAI_API_KEY="your-api-key" \
     briefxai:latest
   ```

## Configuration

### Setting API Keys

BriefX supports multiple LLM providers. Set keys for whichever you plan to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="AIza..."
```

### Server Configuration

```bash
# Server host and port
export BRIEFX_HOST="127.0.0.1"
export BRIEFX_PORT="8080"

# Log level (DEBUG, INFO, WARNING, ERROR)
export BRIEFX_LOG_LEVEL="INFO"
```

### Database Configuration

BriefX uses SQLite by default for local development. For production, PostgreSQL is recommended:

```bash
# SQLite (default - no configuration needed)
export DATABASE_URL="sqlite:///briefx.db"

# PostgreSQL
export DATABASE_URL="postgresql://user:password@localhost/briefx"
```

### Local Models with Ollama

To run BriefX without any cloud API keys:

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama2

# 3. Start BriefX - it will detect Ollama automatically
python app.py
```

## Running BriefX

### Web Interface

```bash
cd python/
python app.py
```

Open http://localhost:8080 in your browser.

### Command Line Interface

```bash
cd python/

# Show available commands
python cli_simple.py --help

# Generate sample conversations for testing
python cli_simple.py generate --count 50

# Run a test analysis
python cli_simple.py test
```

## Your First Analysis

### Step 1: Prepare Your Data

Create a JSON file with your conversations:

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
        "content": "I'd be happy to help. Can you tell me more about the issue?"
      },
      {
        "role": "user",
        "content": "I was charged twice this month"
      },
      {
        "role": "assistant",
        "content": "I apologize for the double charge. Let me look into that for you."
      }
    ],
    "metadata": {
      "timestamp": "2024-01-15T10:30:00Z",
      "customer_id": "cust_123"
    }
  }
]
```

### Step 2: Upload and Analyze

**Via the web interface:**

1. Navigate to http://localhost:8080
2. Click "New Analysis"
3. Upload your JSON file
4. Select "Customer Support" template (or configure manually)
5. Click "Start Analysis"

**Via the REST API:**

```bash
# Upload conversations
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d @conversations.json

# Start analysis
curl -X POST http://localhost:8080/api/analysis/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_001", "options": {"extract_facets": true, "perform_clustering": true}}'

# Poll for results
curl http://localhost:8080/api/analysis/results
```

### Step 3: View Results

The analysis extracts:
- **Sentiments** - Customer satisfaction levels per conversation
- **Topics** - Main discussion themes
- **Issues** - Problems identified
- **Resolutions** - How issues were resolved
- **Patterns** - Common trends across all conversations

Results are available in:
- Web dashboard for interactive exploration
- JSON format for programmatic access
- CSV format for spreadsheet analysis
- HTML reports for sharing

## Common Use Cases

### Customer Support Analysis

```bash
# Upload support tickets and analyze sentiment and issue types
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d @support_tickets.json
```

### Privacy-Sensitive Data

BriefX includes automatic PII detection. For medical or financial conversations, enable strict privacy mode in the analysis settings to mask or remove sensitive identifiers before clustering.

### Large Datasets

For datasets with thousands of conversations, use the batch processing API and monitor progress via the WebSocket endpoint or the real-time dashboard.

## Troubleshooting

### Server won't start

- Check if port 8080 is already in use: `lsof -i :8080`
- Use a different port: `BRIEFX_PORT=3000 python app.py`

### Analysis fails immediately

- Verify at least one API key is set: `echo $OPENAI_API_KEY`
- Check network connectivity to your LLM provider
- Ensure input file is valid JSON

### High memory usage

- Reduce batch size by setting smaller analysis chunks in the UI
- Use a lighter embedding model
- Process conversations in smaller files

### Getting Help

1. Check the [FAQ](https://github.com/briefcasebrain/briefxai/discussions)
2. Search [existing issues](https://github.com/briefcasebrain/briefxai/issues)
3. Open a new issue on GitHub

## Next Steps

- Read the [Architecture Overview](architecture.md) to understand the system design
- See the [API Reference](api.md) for programmatic access
- Review [Configuration Guide](configuration.md) for advanced settings
- Read the [Development Guide](development.md) to contribute
