# BriefXAI Quick Start Guide

Get BriefXAI running in under 5 minutes with smart auto-detection and cost optimization! ⚡

## What You'll Get

✨ **Smart Model Selection** - Automatically picks the best model for your content  
💰 **Cost Optimization** - Built-in features to minimize cloud API costs  
🎯 **Unified Dashboard** - Single interface with Overview, Clusters, and Visualization  
📄 **OCR Support** - Upload PDFs, DOCX, TXT files with automatic text extraction  

## Prerequisites

Before starting, ensure you have:
- Rust 1.70+ installed ([Install Rust](https://rustup.rs/))
- **Optional**: API key from OpenAI, Google Gemini, or Anthropic
- 2GB of available RAM (reduced requirement)
- 500MB of free disk space

## Step 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/briefcasebrain/briefxai.git
cd briefxai

# Build the project (takes 2-3 minutes)
cargo build --release
```

## Step 2: Setup Environment (Optional)

Create a `.env` file for easy configuration:

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor
nano .env
```

**Choose your AI provider** (all are optional):

### Option A: Google Gemini (Most Cost-Effective) 🏆
```bash
# Get free API key: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your-gemini-key-here
```
**Cost**: ~$0.10 per 1000 conversations

### Option B: OpenAI (Premium Quality)
```bash
# Get API key: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-key-here
```
**Cost**: ~$0.50-2.00 per 1000 conversations

### Option C: Local Ollama (Completely Free) 🆓
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (3GB download)
ollama pull llama3.2:3b

# That's it! BriefXAI will auto-detect Ollama
```

### Option D: No Setup (Demo Mode)
```bash
# Just leave .env empty - use built-in example data!
```

## Step 3: Start the Server

```bash
./target/release/briefxai ui --port 8080
```

You should see:
```
🚀 BriefXAI Web Interface Starting...
📊 Loading conversation analysis models...
✅ Server ready at: http://localhost:8080
🎯 Auto-detection enabled for cost optimization
```

**Pro Tip**: The server will automatically open your browser! 🌐

## Step 4: Analyze Your First Conversation

### Using the Web UI (Recommended)

1. **Open your browser** to <http://localhost:8080>

2. **Choose your data source:**
   - 📁 **Upload Files**: JSON, CSV, PDF, TXT, DOCX (with OCR)
   - ⭐ **Use Example Data**: Try with 100 sample conversations
   - 📋 **Paste Data**: Copy/paste conversation text directly

3. **Smart Configuration:**
   - Click **"🎯 Auto-Detect"** buttons for automatic model selection
   - Or manually choose LLM and embedding models
   - **Cost estimate** appears automatically!

4. **Start Analysis** and watch real-time progress:
   ```
   ✅ Data validation complete
   🧠 Extracting conversation facets...
   🔄 Generating embeddings (batch 1/3)...
   🎯 Clustering conversations...
   📊 Creating visualizations...
   🎉 Analysis complete!
   ```

5. **Explore Results** in the unified dashboard:
   - **Overview**: Key metrics and insights
   - **Clusters**: Hierarchical conversation groups  
   - **Visualization**: Interactive UMAP plots
   - **Browse**: Search and filter conversations
   - **Export**: Download CSV, JSON, or HTML reports

### Using the API

```bash
# Create a sample conversation file
cat > conversations.json << 'EOF'
[
  {
    "messages": [
      {"role": "user", "content": "I can't log into my account"},
      {"role": "assistant", "content": "I'll help you reset your password"},
      {"role": "user", "content": "Thanks! That worked perfectly."}
    ],
    "metadata": {
      "timestamp": "2024-01-15T10:30:00Z",
      "category": "support"
    }
  }
]
EOF

# Analyze via CLI with smart defaults
./target/release/briefxai analyze \
  conversations.json \
  --output results/

# Or analyze via API
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": [...], "config": {"llm_provider": "gemini"}}'
```

## Step 5: View Results

Results include:
- **Clusters** - Grouped similar conversations
- **Facets** - Extracted dimensions (sentiment, topics, etc.)
- **Visualizations** - UMAP projections and cluster maps
- **Insights** - Key patterns and trends

## Common Commands

```bash
# Start web interface (recommended)
./target/release/briefxai ui --port 8080

# Generate sample data for testing  
./target/release/briefxai example --output sample.json --count 100

# CLI analysis with auto-detection
./target/release/briefxai analyze sample.json --output results/

# Start server on custom port
./target/release/briefxai ui --port 3000

# Enable debug logging
RUST_LOG=debug ./target/release/briefxai ui

# Check what models are available
curl http://localhost:8080/api/status
```

## Pro Tips for Cost Optimization 💰

```bash
# Set cost-effective defaults in .env
echo "DEFAULT_LLM_MODEL=gemini-1.5-flash-8b" >> .env
echo "DEFAULT_EMBEDDING_MODEL=text-embedding-004" >> .env
echo "BATCH_SIZE=3" >> .env

# Monitor costs in real-time
tail -f briefxai.log | grep "Cost estimate"

# Use demo mode for testing (no API calls)
echo "ENABLE_DEMO_MODE=true" >> .env
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