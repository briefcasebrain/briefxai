# BriefX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/briefcasebrain/briefxai)

A high-performance conversation analysis platform for extracting insights from conversational data at scale.

Try the free, hosted version out here: [https://xai.briefcasebrain.io/](https://xai.briefcasebrain.io/)

> **Note**: This repository contains both Python and Rust implementations. **The Python implementation is the active, maintained version**. The Rust implementation is deprecated and no longer maintained.

## Overview

BriefX provides enterprise-grade conversation analysis capabilities with advanced clustering algorithms and pattern discovery methodologies. Built for performance, privacy, and scalability.

## Features

### Core Capabilities
- **Advanced Clustering** - Multi-level conversation grouping with hierarchical analysis
- **Facet Extraction** - Automatic identification of topics, sentiments, intents, and entities  
- **Privacy Protection** - PII detection and threshold-based anonymization
- **Real-time Processing** - Stream processing with WebSocket support
- **Multi-provider Support** - Compatible with various LLM providers (OpenAI, Anthropic, local models)

### Technical Features  
- Concurrent processing for high throughput
- Session persistence with pause/resume capability
- Efficient caching and batch processing
- REST and WebSocket APIs
- UMAP dimensionality reduction for visualization
- Smart preprocessing with language detection

## Quick Start

### Python Implementation (Active)

1. **Navigate to Python Directory**
   ```bash
   cd python/
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # or
   export ANTHROPIC_API_KEY="your-api-key-here" 
   ```

4. **Run Web Interface**
   ```bash
   python app.py
   ```

5. **Access Interface**
   Open http://localhost:8080 in your browser

### Rust Implementation (Deprecated)

> ⚠️ **Deprecated**: The Rust implementation is no longer maintained. Use the Python implementation above.

## Usage Examples

### Python API

```python
from briefx.data.models import ConversationData, Message
from briefx.examples import generate_example_conversations

# Create conversation data
conversations = [
    ConversationData(messages=[
        Message(role="user", content="I need help with my order"),
        Message(role="assistant", content="I'd be happy to help with your order")
    ])
]

# Or generate examples for testing
test_conversations = generate_example_conversations(count=5, seed=42)
```

### REST API

```bash
# Upload conversations
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"conversations": [{"messages": [{"role": "user", "content": "Hello"}]}]}'

# Get analysis results
curl http://localhost:8080/api/analysis/results
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `BRIEFX_PORT` - Server port (default: 8080)
- `BRIEFX_HOST` - Server host (default: 127.0.0.1)
- `BRIEFX_LOG_LEVEL` - Log level (default: INFO)

### Configuration File
Create `config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8080

[analysis]
batch_size = 100
max_concurrent_requests = 10

[providers.openai]
enabled = true
model = "gpt-4"
api_key = "${OPENAI_API_KEY}"
```

See `docs/configuration.md` for complete configuration options.

## Architecture

- **Data Models** - Structured conversation and analysis data types
- **Preprocessing** - Text normalization, validation, and language detection
- **Analysis Pipeline** - Clustering, facet extraction, and pattern discovery
- **Provider System** - Pluggable LLM and embedding provider architecture
- **Web Interface** - Interactive visualization and analysis tools
- **Persistence** - Session management and result caching

## Development

### Python Development (Active)

```bash
# Navigate to Python directory
cd python/

# Install development dependencies
pip install -r requirements.txt

# Run tests
python tests/test_updated.py

# Use CLI tools
python cli_simple.py --help
python cli_simple.py generate --count 5
python cli_simple.py test

# Start development server
python app.py
```

### Rust Development (Deprecated)

> ⚠️ **Deprecated**: The Rust implementation is no longer maintained.

## API Documentation

### Core Endpoints

- `GET /` - Web interface
- `POST /api/conversations` - Upload conversation data
- `GET /api/analysis/results` - Get analysis results
- `POST /api/analysis/start` - Start analysis session
- `GET /api/monitoring/health` - Health check

### WebSocket Endpoints

- `/ws/analysis` - Real-time analysis updates
- `/ws/progress` - Analysis progress updates

See `docs/api.md` for complete API documentation.

## Performance

### Benchmarks (Python Implementation)
- **Throughput**: 1000+ conversations/minute
- **Memory Usage**: ~512MB for 10k conversations  
- **Clustering**: 100 conversations clustered in <5 seconds
- **Facet Extraction**: 50ms average per conversation
- **API Response Time**: <100ms for most endpoints

## Privacy & Security

- **PII Detection** - Automatic detection of emails, phone numbers, addresses
- **Data Anonymization** - Configurable masking and removal policies
- **Local Processing** - Option to run entirely offline with local models
- **Secure Storage** - Encrypted data persistence options
- **Access Control** - API key authentication and rate limiting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

For questions and support:
- Check the [documentation](docs/)
- Review [examples](python/briefx/examples.py)
- Use the CLI: `python python/cli_simple.py --help`
- Open an issue on GitHub

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
