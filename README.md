# BriefX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance conversation analysis platform for extracting insights from conversational data at scale.

Try the free, hosted version here: [https://xai.briefcasebrain.io/](https://xai.briefcasebrain.io/)

## Overview

BriefX provides enterprise-grade conversation analysis with advanced clustering algorithms and pattern discovery based on the [Clio methodology](https://arxiv.org/html/2412.13678v1). Built for performance, privacy, and scalability.

## Features

### Core Capabilities
- **Advanced Clustering** - Multi-level conversation grouping with hierarchical analysis
- **Facet Extraction** - Automatic identification of topics, sentiments, intents, and entities
- **Privacy Protection** - PII detection and threshold-based anonymization
- **Real-time Processing** - Stream processing with WebSocket support
- **Multi-provider Support** - OpenAI, Anthropic, Google Gemini, Ollama, and HuggingFace

### Technical Features
- Concurrent processing for high throughput
- Session persistence with pause/resume capability
- Efficient caching and batch processing
- REST and WebSocket APIs
- UMAP dimensionality reduction for visualization
- Smart preprocessing with language detection

## Quick Start

1. **Navigate to Python directory**
   ```bash
   cd python/
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # or
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

4. **Run the web interface**
   ```bash
   python app.py
   ```

5. **Open in your browser**
   Navigate to http://localhost:8080

See [docs/quickstart.md](docs/quickstart.md) for a more detailed walkthrough.

## Usage Examples

### Python API

```python
from briefx.data.models import ConversationData, Message
from briefx.analysis.pipeline import AnalysisPipeline

# Create pipeline
pipeline = AnalysisPipeline()

# Analyze conversations
results = pipeline.analyze(conversations)
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

### CLI

```bash
cd python/
python cli_simple.py --help
python cli_simple.py generate --count 5
python cli_simple.py test
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GOOGLE_API_KEY` | Google Gemini API key | — |
| `DATABASE_URL` | Database connection string | SQLite (local) |
| `BRIEFX_PORT` | Server port | `8080` |
| `BRIEFX_HOST` | Server host | `127.0.0.1` |
| `BRIEFX_LOG_LEVEL` | Log level | `INFO` |

See [docs/configuration.md](docs/configuration.md) for all configuration options.

## Architecture

- **Flask Web Server** - REST API and WebSocket endpoints
- **Analysis Pipeline** - Clustering, facet extraction, and pattern discovery
- **Provider System** - Pluggable LLM and embedding provider architecture
- **Clio Engine** - Privacy-preserving hierarchical conversation analysis
- **Web Interface** - Interactive visualization and analysis dashboard
- **Persistence** - SQLite (default) or PostgreSQL session management

See [docs/architecture.md](docs/architecture.md) for a full system overview.

## Development

```bash
cd python/

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python app.py
```

See [docs/development.md](docs/development.md) for the full development guide.

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

See [docs/api.md](docs/api.md) for complete API documentation.

## Deployment

BriefX is a Flask + Gunicorn application deployable to any container platform. See [docs/deployment.md](docs/deployment.md) for cost-optimized deployment guides for Google Cloud, AWS, and Vercel.

## Privacy & Security

- **PII Detection** - Automatic detection of emails, phone numbers, addresses
- **Data Anonymization** - Configurable masking and removal policies
- **Local Processing** - Option to run entirely offline with Ollama
- **Secure Storage** - Encrypted data persistence options
- **Access Control** - API key authentication and rate limiting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

- Check the [documentation](docs/)
- Review [examples](python/briefx/examples.py)
- Use the CLI: `python python/cli_simple.py --help`
- Open an issue on [GitHub](https://github.com/briefcasebrain/briefxai/issues)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
