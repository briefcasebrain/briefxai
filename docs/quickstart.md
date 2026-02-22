# Quick Start Guide

Get BriefX running in 5 minutes!

## Prerequisites

- Python 3.9+
- An API key from OpenAI, Anthropic, or Google Gemini (or a local Ollama installation)
- 2GB of available RAM

## Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/briefcasebrain/briefxai.git
cd briefxai/python

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Keys

Choose your preferred AI provider:

### Option A: OpenAI (Recommended)
```bash
export OPENAI_API_KEY="sk-..."
```

### Option B: Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option C: Google Gemini
```bash
export GOOGLE_API_KEY="AIza..."
```

### Option D: Local Ollama (No API key needed)
```bash
# Install Ollama first: https://ollama.com
ollama pull llama2
export OLLAMA_HOST="http://localhost:11434"
```

## Step 3: Start the Server

```bash
python app.py
```

You should see:
```
Starting BriefX Web Interface
Server running on http://localhost:8080
```

## Step 4: Analyze Your First Conversation

### Using the Web UI

1. Open your browser to <http://localhost:8080>
2. Click "Upload Conversations" or "Use Sample Data"
3. Configure analysis settings:
   - Select facets to extract (sentiment, topics, etc.)
   - Choose clustering parameters
   - Set privacy thresholds
4. Click "Start Analysis"
5. Watch real-time results appear!

### Using the CLI

```bash
# Generate sample data for testing
python cli_simple.py generate --count 20

# Run a test analysis
python cli_simple.py test

# See all CLI options
python cli_simple.py --help
```

### Using the API

```bash
# Create a sample conversation file
cat > conversations.json << 'EOF'
[
  {
    "messages": [
      {"role": "user", "content": "I can't log into my account"},
      {"role": "assistant", "content": "I'll help you reset your password"}
    ]
  }
]
EOF

# Upload and analyze via REST API
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d @conversations.json
```

## Step 5: View Results

Results include:
- **Clusters** - Grouped similar conversations
- **Facets** - Extracted dimensions (sentiment, topics, entities, etc.)
- **Visualizations** - UMAP projections and cluster maps
- **Insights** - Key patterns and trends

## Common Issues

### "API key not found"
Ensure your API key is exported in the current shell:
```bash
echo $OPENAI_API_KEY  # Should display your key
```

### "Port already in use"
Set a different port:
```bash
BRIEFX_PORT=3000 python app.py
```

### "Module not found" errors
Make sure you installed dependencies in the active virtual environment:
```bash
pip install -r requirements.txt
```

## Next Steps

- Read the [Getting Started guide](getting-started.md) for a deeper walkthrough
- Explore [API endpoints](api.md)
- Configure [advanced settings](configuration.md)
- Read the [Architecture overview](architecture.md)
- Join our [community](https://github.com/briefcasebrain/briefxai/discussions)
