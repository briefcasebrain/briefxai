# Development Guide

This guide covers the development workflow, testing strategies, and best practices for contributing to BriefXAI.

## Development Environment Setup

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- SQLite 3.35+
- Git
- Optional: Docker for containerized development

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/briefcasebrain/briefxai.git
   cd briefxai
   ```

2. **Install Rust toolchain:**
   ```bash
   rustup update stable
   rustup component add rustfmt clippy
   ```

3. **Install development tools:**
   ```bash
   cargo install cargo-watch cargo-edit cargo-outdated cargo-audit
   cargo install cargo-tarpaulin  # For code coverage
   ```

4. **Set up pre-commit hooks:**
   ```bash
   cp scripts/pre-commit .git/hooks/
   chmod +x .git/hooks/pre-commit
   ```

## Project Structure

```
briefxai/
├── src/                    # Source code
│   ├── lib.rs             # Library entry point
│   ├── main.rs            # Binary entry point
│   ├── config.rs          # Configuration handling
│   ├── types.rs           # Core data types
│   ├── analysis/          # Analysis modules
│   ├── preprocessing/     # Data preprocessing
│   ├── llm/              # LLM provider integrations
│   └── web/              # Web server and API
├── tests/                 # Integration tests
├── benches/              # Benchmarks
├── migrations/           # Database migrations
├── assets/               # Static assets
├── docs/                 # Documentation
└── scripts/              # Development scripts
```

## Development Workflow

### Running the Application

```bash
# Development mode with hot reload
cargo watch -x run

# Run with debug logging
RUST_LOG=debug cargo run

# Run with specific config
cargo run -- serve --config dev.toml

# Release mode
cargo run --release
```

### Code Formatting

```bash
# Format all code
cargo fmt

# Check formatting without changes
cargo fmt -- --check
```

### Linting

```bash
# Run clippy with all targets
cargo clippy --all-targets --all-features

# Strict mode (treat warnings as errors)
cargo clippy -- -D warnings

# With pedantic lints
cargo clippy -- -W clippy::pedantic
```

### Building

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Build for specific target
cargo build --target x86_64-unknown-linux-musl
```

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run with output displayed
cargo test -- --nocapture

# Run specific test
cargo test test_session_manager

# Run tests in single thread (for debugging)
cargo test -- --test-threads=1

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test '*'

# Run with coverage
cargo tarpaulin --out Html
```

### Writing Tests

#### Unit Tests

Place unit tests in the same file as the code:

```rust
// src/analysis/session_manager.rs

impl SessionManager {
    pub fn create_session(&self, config: Config) -> Result<Session> {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_session() {
        let manager = SessionManager::new();
        let config = Config::default();
        let session = manager.create_session(config).unwrap();
        assert_eq!(session.state, SessionState::Created);
    }

    #[tokio::test]
    async fn test_async_operation() {
        // Async test implementation
    }
}
```

#### Integration Tests

Place integration tests in the `tests/` directory:

```rust
// tests/integration_test.rs

use briefxai::{AnalysisEngine, Config};

#[tokio::test]
async fn test_full_analysis_pipeline() {
    let config = Config::test_config();
    let engine = AnalysisEngine::new(config).await.unwrap();
    
    let result = engine.analyze_file("test_data.json").await.unwrap();
    assert!(!result.is_empty());
}
```

#### Property-Based Tests

Use proptest for property-based testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_never_panics(s: String) {
        let _ = parse_conversation(&s);  // Should never panic
    }
}
```

### Test Organization

```
tests/
├── common/              # Shared test utilities
│   └── mod.rs
├── integration_test.rs  # Integration tests
├── api_test.rs         # API endpoint tests
└── e2e_test.rs         # End-to-end tests
```

## Debugging

### Using the Debugger

#### VS Code

1. Install the Rust extension
2. Add launch configuration:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug executable",
            "cargo": {
                "args": ["build", "--bin=briefxai"],
                "filter": {
                    "name": "briefxai",
                    "kind": "bin"
                }
            },
            "args": ["serve"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

#### Command Line

```bash
# Using rust-gdb
rust-gdb target/debug/briefxai

# Using lldb
rust-lldb target/debug/briefxai
```

### Logging

```rust
use tracing::{debug, error, info, trace, warn};

// Add logging to your code
info!("Starting analysis for session {}", session_id);
debug!("Configuration: {:?}", config);
trace!("Detailed trace information");
warn!("Deprecated feature used");
error!("Failed to connect: {}", err);

// Structured logging
info!(
    session_id = %session_id,
    conversation_count = conversations.len(),
    "Starting batch processing"
);
```

Run with different log levels:

```bash
RUST_LOG=trace cargo run
RUST_LOG=briefxai=debug cargo run
RUST_LOG=briefxai::analysis=trace cargo run
```

## Performance Optimization

### Profiling

```bash
# CPU profiling with flamegraph
cargo install flamegraph
cargo flamegraph

# Memory profiling with valgrind
valgrind --tool=massif target/release/briefxai
ms_print massif.out.<pid>

# Using cargo-profiling
cargo install cargo-profiling
cargo profiling callgrind
```

### Benchmarking

```rust
// benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_clustering(c: &mut Criterion) {
    c.bench_function("kmeans_1000", |b| {
        b.iter(|| {
            cluster_conversations(black_box(&conversations))
        });
    });
}

criterion_group!(benches, benchmark_clustering);
criterion_main!(benches);
```

Run benchmarks:

```bash
cargo bench
cargo bench -- --save-baseline before
cargo bench -- --baseline before
```

## Database Management

### Migrations

```bash
# Create new migration
./scripts/create_migration.sh "add_indexes"

# Run migrations
cargo run -- migrate

# Rollback migration
cargo run -- migrate rollback
```

### Database Debugging

```bash
# Open SQLite shell
sqlite3 data/briefxai.db

# Common queries
.tables
.schema sessions
SELECT * FROM sessions WHERE state = 'running';
```

## API Development

### Testing API Endpoints

```bash
# Test with curl
curl -X POST http://localhost:8080/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Session"}'

# Using httpie
http POST localhost:8080/api/sessions name="Test Session"

# Load testing with vegeta
echo "POST http://localhost:8080/api/sessions" | \
  vegeta attack -duration=30s -rate=10 | \
  vegeta report
```

### WebSocket Testing

```javascript
// WebSocket test client
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'subscribe',
        session_id: 'test_session'
    }));
};

ws.onmessage = (event) => {
    console.log('Received:', event.data);
};
```

## Documentation

### Generating Documentation

```bash
# Generate and open documentation
cargo doc --open

# Include private items
cargo doc --document-private-items

# With dependencies
cargo doc --no-deps
```

### Writing Documentation

```rust
/// Analyzes a batch of conversations and extracts insights.
///
/// This function processes conversations in parallel, extracting facets,
/// generating embeddings, and performing clustering analysis.
///
/// # Arguments
///
/// * `conversations` - Vector of conversations to analyze
/// * `config` - Analysis configuration
///
/// # Returns
///
/// Returns `Ok(AnalysisResult)` on success, or an error if analysis fails.
///
/// # Examples
///
/// ```
/// use briefxai::{analyze_batch, Config};
///
/// let config = Config::default();
/// let result = analyze_batch(conversations, &config)?;
/// println!("Found {} clusters", result.clusters.len());
/// ```
///
/// # Errors
///
/// This function will return an error if:
/// - The provider is unavailable
/// - Input validation fails
/// - Database operations fail
pub async fn analyze_batch(
    conversations: Vec<Conversation>,
    config: &Config,
) -> Result<AnalysisResult, AnalysisError> {
    // Implementation
}
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - uses: actions-rs/cargo@v1
        with:
          command: test
      
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: rustfmt, clippy
      - run: cargo fmt -- --check
      - run: cargo clippy -- -D warnings
```

## Release Process

### Version Management

```bash
# Update version in Cargo.toml
cargo set-version 2.2.0

# Create git tag
git tag -a v2.2.0 -m "Release version 2.2.0"
git push origin v2.2.0
```

### Building Releases

```bash
# Build for multiple targets
./scripts/build_releases.sh

# Create release artifacts
tar -czf briefxai-linux-x64.tar.gz target/release/briefxai
zip briefxai-windows-x64.zip target/release/briefxai.exe
```

## Troubleshooting

### Common Issues

#### Compilation Errors

```bash
# Clean build
cargo clean
cargo build

# Update dependencies
cargo update

# Check for breaking changes
cargo outdated
```

#### Test Failures

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run specific test with output
cargo test test_name -- --nocapture
```

#### Performance Issues

```bash
# Profile with release build
cargo build --release
perf record target/release/briefxai
perf report
```

## Best Practices

### Code Quality

1. **Error Handling**: Use `Result` types and avoid `unwrap()` in production code
2. **Documentation**: Document all public APIs
3. **Testing**: Maintain >80% code coverage
4. **Dependencies**: Keep dependencies minimal and up-to-date
5. **Security**: Run `cargo audit` regularly

### Performance

1. **Async/Await**: Use async for I/O operations
2. **Parallelism**: Use rayon for CPU-bound tasks
3. **Memory**: Avoid unnecessary allocations
4. **Caching**: Implement caching for expensive operations

### Git Workflow

1. **Branches**: Use feature branches
2. **Commits**: Write clear, atomic commits
3. **Pull Requests**: Include tests and documentation
4. **Reviews**: Request reviews before merging

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Async Book](https://rust-lang.github.io/async-book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Clippy Lints](https://rust-lang.github.io/rust-clippy/)
- [Cargo Documentation](https://doc.rust-lang.org/cargo/)