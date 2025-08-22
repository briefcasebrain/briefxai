# Contributing to BriefXAI

Thank you for your interest in contributing to BriefXAI! This document provides comprehensive guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Rust 1.70+ (`rustc --version`)
- SQLite 3.35+ (`sqlite3 --version`)
- Git 2.0+ (`git --version`)
- Node.js 16+ (optional, for UI development)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/briefxai.git
   cd briefxai
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenAI, Gemini, etc.)
   ```

3. **Build the project:**
   ```bash
   cargo build
   cargo test
   ```

4. **Run in development mode:**
   ```bash
   RUST_LOG=debug cargo run -- ui
   ```

## Code Style Guidelines

### Rust Code

- Follow standard Rust formatting with `rustfmt`
- Use `clippy` for linting
- Write descriptive variable and function names
- Add documentation comments for public APIs
- Keep functions focused and small

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy -- -D warnings
```

### JavaScript/TypeScript

- Use ES6+ features
- Follow consistent indentation (2 spaces)
- Use meaningful component names
- Add JSDoc comments for complex functions

### Documentation

- Update README.md for new features
- Add inline documentation for complex logic
- Include examples in documentation
- Keep documentation up-to-date with code changes

## 🧪 Testing Requirements

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture

# Run integration tests
cargo test --test '*'
```

### Writing Tests

- Write unit tests for new functions
- Add integration tests for new features
- Test edge cases and error conditions
- Mock external API calls in tests

## 🔄 Pull Request Process

### Before Submitting

1. **Update your fork:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Verify your changes:**
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   cargo build --release
   ```

### Submitting a PR

1. **Commit with descriptive messages:**
   ```bash
   git commit -m "feat: add new analysis feature
   
   - Implement feature X
   - Add tests for feature X
   - Update documentation"
   ```

2. **Create Pull Request:**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include screenshots for UI changes

## Project Structure

```
briefxai/
├── src/
│   ├── lib.rs           # Main library entry
│   ├── main.rs          # CLI entry point
│   ├── web.rs           # Web server
│   ├── analysis/        # Analysis modules
│   ├── llm/             # LLM providers
│   └── clio_core.rs     # Clio methodology
├── tests/               # Integration tests
├── openclio_ui_data/    # Clio UI files
├── briefxai_ui_data/    # Main UI files
└── docs/                # Documentation
```

## 🐛 Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Rust version)
- Error messages and logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions considered

## Development Tips

### Performance

- Use async/await for I/O operations
- Batch API calls when possible
- Implement proper caching strategies
- Profile before optimizing

### Security

- Never commit API keys or secrets
- Validate all user input
- Follow OWASP guidelines
- Implement rate limiting

### Privacy

- Follow Clio methodology principles
- Implement PII detection/masking
- Use local processing when possible
- Document data handling

## 📚 Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Tokio Documentation](https://tokio.rs/tokio/tutorial)
- [Axum Framework](https://github.com/tokio-rs/axum)
- [Clio Paper](https://arxiv.org/html/2412.13678v1)

## License

By contributing to BriefXAI, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make BriefXAI better for everyone. We appreciate your time and effort!

---

Questions? Open an issue or contact the maintainers.