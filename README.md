# BriefX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/briefcasebrain/briefxaiai)

A high-performance conversation analysis platform built in Rust, designed for extracting insights from conversations at scale.

## Overview

BriefX provides enterprise-grade conversation analysis capabilities with a focus on performance, privacy, and usability. It implements advanced clustering algorithms and the Clio methodology for pattern discovery in conversational data.

## Features

### Core Capabilities
- **Hierarchical Clustering** - Multi-level conversation grouping with pattern discovery
- **Facet Extraction** - Automatic identification of topics, sentiments, and intents
- **Privacy Protection** - PII detection and threshold-based anonymization
- **Real-time Analysis** - Stream processing with WebSocket support
- **Multi-provider Support** - Compatible with various processing engines

### Technical Features
- Concurrent processing for high throughput
- Session persistence with pause/resume capability
- Efficient caching and batch processing
- REST and WebSocket APIs
- UMAP dimensionality reduction for visualization

## Installation

### Prerequisites

- Rust 1.70 or higher
- SQLite 3.35 or higher
- Git 2.0 or higher

### Building from Source

```bash
git clone https://github.com/briefcasebrain/briefxai.git
cd briefx
cargo build --release
```

## Usage

### Starting the Web Interface

```bash
# Set your API key (required for cloud providers)
export OPENAI_API_KEY="your-key-here"

# Start the server
./target/release/briefx ui --port 8080
```

### Command Line Interface

```bash
# Analyze conversations from file
./target/release/briefx analyze -i data.json -o results/

# Generate example data
./target/release/briefx example -n 100 -o sample_data.json

# Serve existing results
./target/release/briefx serve -d results/
```

## Configuration

Create a `config.json` file to customize the analysis:

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "batch_size": 100,
  "enable_clio_features": true,
  "clio_privacy_min_cluster_size": 10
}
```

## API Documentation

### REST Endpoints

- `POST /api/analyze` - Start analysis session
- `POST /api/upload` - Upload conversation data
- `GET /api/status/:session_id` - Check analysis status
- `GET /api/example` - Generate example data

### WebSocket

Connect to `/ws/progress` for real-time analysis updates.

## Architecture

```
src/
├── analysis/           # Core analysis engine
├── clustering/         # Clustering algorithms
├── embeddings/         # Embedding generation
├── llm/               # LLM provider integrations
├── preprocessing/     # Data validation and cleaning
├── privacy/           # Privacy protection features
├── web/              # Web server and API
└── visualization/    # Data visualization components
```

## Development

### Running Tests

```bash
cargo test
```

### Code Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy -- -D warnings
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Clio methodology from recent research
- Built with the Rust ecosystem
- Powered by modern LLM APIs

## Deployment

See the [deployment documentation](docs/deployment/) for instructions on deploying to cloud platforms.

## Support

For issues and questions, please use the [GitHub issue tracker](https://github.com/briefcasebrain/briefxai/issues).