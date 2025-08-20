# BriefXAI

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.1.0-green.svg)](https://github.com/briefcasebrain/briefxai)
[![Documentation](https://docs.rs/briefxai/badge.svg)](https://docs.rs/briefxai)

> Advanced AI conversation analysis platform built with Rust for performance and reliability

BriefXAI is a production-ready platform for analyzing AI conversations at scale, extracting insights, identifying patterns, and visualizing conversation clusters.

## Features

- **Smart Analysis** - Extract facets, sentiments, and patterns from conversations automatically
- **High Performance** - Process thousands of conversations concurrently with Rust's efficiency
- **Pause/Resume** - Stop and resume long-running analyses without losing progress
- **Multi-Provider** - Support for OpenAI, Ollama, and other LLM providers with automatic fallback
- **Real-time Results** - Stream insights as they're discovered via WebSocket
- **Data Privacy** - PII detection, local processing options, and data validation
- **Templates** - Pre-built templates for customer support, sales, medical, and custom use cases
- **Modern UI** - Professional web interface with guided setup wizard

## Quick Start

```bash
# Install from crates.io (when published)
cargo install briefxai

# Or build from source
git clone https://github.com/briefcasebrain/briefxai.git
cd briefxai
cargo build --release

# Run the server
briefxai serve

# Access the UI
open http://localhost:8080
```

## Installation

### Requirements

- Rust 1.70 or higher
- SQLite 3.35 or higher

### From Source

```bash
git clone https://github.com/briefcasebrain/briefxai.git
cd briefxai
cargo build --release
cargo test
```

### Docker

```bash
docker build -t briefxai:latest .
docker run -p 8080:8080 briefxai:latest
```

## Usage

### Basic Example

```rust
use briefxai::{Config, AnalysisEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::from_file("config.toml")?;
    
    // Initialize engine
    let engine = AnalysisEngine::new(config).await?;
    
    // Analyze conversations
    let results = engine.analyze_file("conversations.json").await?;
    
    // Export results
    results.export_json("results.json")?;
    
    Ok(())
}
```

### CLI Usage

```bash
# Start web server
briefxai serve --port 8080

# Analyze conversations from file
briefxai analyze conversations.json --output results.json

# Resume a paused session
briefxai resume session_123

# Export results
briefxai export session_123 --format csv
```

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Development Guide](docs/development.md)

## Configuration

Create a `config.toml` file:

```toml
[server]
host = "127.0.0.1"
port = 8080

[database]
path = "briefxai.db"

[providers.openai]
api_key = "your-api-key"
model = "gpt-4"

[analysis]
batch_size = 100
max_concurrent = 10
```

See [Configuration Guide](docs/configuration.md) for all options.

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

### Testing

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test '*'

# With coverage
./scripts/coverage.sh

# Linting
cargo clippy -- -D warnings

# Format code
cargo fmt
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Rust for performance and reliability
- Uses SQLite for persistent storage
- Integrates with OpenAI and local LLM providers
- WebSocket support for real-time updates

## Support

- [Issue Tracker](https://github.com/briefcasebrain/briefxai/issues)
- [Discussions](https://github.com/briefcasebrain/briefxai/discussions)