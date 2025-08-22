# Quick Start Guide

Get BriefXAI running in 5 minutes!

## Prerequisites

Before starting, ensure you have:
- Rust 1.70+ installed ([Install Rust](https://rustup.rs/))
- An API key from OpenAI or Google Gemini
- 4GB of available RAM
- 1GB of free disk space

## Step 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/briefcasebrain/briefxai.git
cd briefxai

# Build the project (takes 2-3 minutes)
cargo build --release
```

## Step 2: Configure API Keys

Choose your preferred AI provider:

### Option A: OpenAI (Recommended)
```bash
export OPENAI_API_KEY="sk-..."
```

### Option B: Google Gemini
```bash
export GEMINI_API_KEY="AIza..."
```

### Option C: Local Ollama (No API key needed)
```bash
# Install Ollama first
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama2

# Set the host
export OLLAMA_HOST="http://localhost:11434"
```

## Step 3: Start the Server

```bash
./target/release/briefxai serve
```

You should see:
```
Starting BriefXAI Web Interface
BriefXAI is ready!
Open your browser to: http://localhost:8080
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

# Analyze via CLI
./target/release/briefxai analyze \
  --input conversations.json \
  --output results/
```

## Step 5: View Results

Results include:
- **Clusters** - Grouped similar conversations
- **Facets** - Extracted dimensions (sentiment, topics, etc.)
- **Visualizations** - UMAP projections and cluster maps
- **Insights** - Key patterns and trends

## Common Commands

```bash
# Generate sample data for testing
./target/release/briefxai example --count 100

# Analyze with custom config
./target/release/briefxai analyze \
  --input data.json \
  --config config.toml \
  --output results/

# Start server on custom port
./target/release/briefxai serve --port 3000

# Enable debug logging
RUST_LOG=debug ./target/release/briefxai serve
```

## Next Steps

- Read the [full documentation](../README.md)
- Explore [API endpoints](api.md)
- Configure [advanced settings](configuration.md)
- Join our [community](https://github.com/briefcasebrain/briefxai/discussions)

## Troubleshooting

### "API key not found"
Ensure your API key is exported:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

### "Port already in use"
Change the port:
```bash
./target/release/briefxai serve --port 3000
```

### "Out of memory"
Reduce batch size in config:
```toml
[performance]
embedding_batch_size = 50
llm_batch_size = 10
```

### "Rate limit exceeded"
Add delays between requests:
```toml
[rate_limiting]
requests_per_minute = 20
```

## Getting Help

- 📖 [Documentation](../README.md)
- 💬 [GitHub Discussions](https://github.com/briefcasebrain/briefxai/discussions)
- 🐛 [Report Issues](https://github.com/briefcasebrain/briefxai/issues)
- 📧 Email: <support@briefxai.com>

---

**Ready to analyze conversations at scale? Let's go!**