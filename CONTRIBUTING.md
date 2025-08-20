# Contributing to BriefXAI

Thank you for your interest in contributing to BriefXAI! We welcome contributions from the community and are grateful for any help you can provide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the project's best interests

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/briefcasebrain/briefxai.git
   cd briefxai
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/briefcasebrain/briefxai.git
   ```
4. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before reporting a bug, please:
- Check existing issues to avoid duplicates
- Gather relevant information (error messages, logs, environment details)
- Try to reproduce the issue with minimal steps

When reporting, include:
- Clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Rust version, etc.)
- Relevant logs or error messages

### Suggesting Features

We welcome feature suggestions! Please:
- Check if the feature has already been suggested
- Provide a clear use case
- Explain how it benefits users
- Consider implementation complexity

### Contributing Code

1. **Find an issue to work on** or create a new one
2. **Comment on the issue** to let others know you're working on it
3. **Write your code** following our coding standards
4. **Add tests** for your changes
5. **Update documentation** if needed
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Rust 1.70 or higher
- SQLite 3.35 or higher
- Git

### Building the Project

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build

# Run tests
cargo test

# Run with verbose output
RUST_LOG=debug cargo run
```

### Development Tools

```bash
# Install development tools
cargo install cargo-watch cargo-edit cargo-outdated

# Run tests on file change
cargo watch -x test

# Check for outdated dependencies
cargo outdated

# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

## Coding Standards

### Rust Style Guide

We follow the official [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/README.html) with these additions:

- Use descriptive variable names
- Keep functions small and focused
- Document public APIs with rustdoc comments
- Use `Result` and `Option` instead of panicking
- Prefer iterators over loops when appropriate

### Code Organization

```rust
// Good: Organized imports
use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::config::Config;
use crate::types::Analysis;

// Good: Clear function documentation
/// Analyzes a batch of conversations and returns insights.
///
/// # Arguments
///
/// * `conversations` - Vector of conversation data
/// * `config` - Analysis configuration
///
/// # Returns
///
/// Result containing analysis insights or error
///
/// # Example
///
/// ```
/// let insights = analyze_batch(conversations, config)?;
/// ```
pub async fn analyze_batch(
    conversations: Vec<Conversation>,
    config: &Config,
) -> Result<Insights, Error> {
    // Implementation
}
```

### Error Handling

```rust
// Good: Use custom error types
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Provider error: {0}")]
    ProviderError(#[from] ProviderError),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
}

// Good: Propagate errors with ?
fn process_data(input: &str) -> Result<Data, AnalysisError> {
    let parsed = parse_input(input)?;
    let validated = validate_data(parsed)?;
    Ok(validated)
}
```

## Testing Guidelines

### Test Organization

```rust
// Unit tests in the same file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test implementation
    }

    #[test]
    fn test_edge_cases() {
        // Test implementation
    }
}
```

### Integration Tests

Place integration tests in the `tests/` directory:

```rust
// tests/integration_test.rs
use briefxai::AnalysisEngine;

#[tokio::test]
async fn test_full_pipeline() {
    // Test implementation
}
```

### Test Coverage

- Aim for at least 80% code coverage
- Test both success and failure paths
- Include edge cases and boundary conditions
- Use property-based testing for complex logic

```bash
# Generate coverage report
./scripts/coverage.sh

# View coverage in browser
open target/coverage/index.html
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   cargo fmt
   
   # Run linter
   cargo clippy -- -D warnings
   
   # Run tests
   cargo test
   
   # Build documentation
   cargo doc --no-deps
   ```

3. **Update documentation** if you've made API changes

4. **Write a clear commit message**:
   ```
   feat: Add support for custom templates
   
   - Implement template parser
   - Add validation for template syntax
   - Include example templates
   
   Closes #123
   ```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No compiler warnings
```

### Review Process

1. All PRs require at least one review
2. Address reviewer feedback promptly
3. Keep PRs focused and reasonably sized
4. Be patient - reviews may take time

## Issue Guidelines

### Issue Templates

#### Bug Report
```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: 
- Rust version:
- BriefXAI version:
```

#### Feature Request
```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## Getting Help

If you need help:

1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/briefcasebrain/briefxai/issues)
3. Ask in [discussions](https://github.com/briefcasebrain/briefxai/discussions)
4. Reach out to maintainers

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- The contributors page

Thank you for contributing to BriefXAI!