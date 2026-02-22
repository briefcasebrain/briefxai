# Contributing to BriefX

Thank you for your interest in contributing to BriefX! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9+
- Git 2.0+
- An API key from OpenAI, Anthropic, or Google Gemini (optional for local Ollama)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/briefxai.git
   cd briefxai/python
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # or
   export ANTHROPIC_API_KEY="your-api-key"
   ```

5. **Run the development server:**
   ```bash
   python app.py
   ```

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions
- Use `black` for code formatting
- Use `flake8` for linting
- Write descriptive variable and function names
- Add docstrings for public functions and classes

```bash
# Format code
black briefx/

# Run linter
flake8 briefx/
```

### Documentation

- Update the relevant docs in `docs/` for new features
- Include examples in documentation
- Keep documentation in sync with code changes

## Testing Requirements

### Running Tests

```bash
# From the python/ directory

# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_complete.py

# Run with coverage
python -m pytest tests/ --cov=briefx
```

### Writing Tests

- Write unit tests for new functions and classes
- Add integration tests for new API endpoints or analysis features
- Test edge cases and error conditions
- Use pytest fixtures for shared setup

```python
# tests/test_example.py
import pytest
from briefx.data.models import ConversationData, Message

def test_conversation_creation():
    conv = ConversationData(messages=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
    ])
    assert len(conv.messages) == 2

@pytest.fixture
def sample_conversations():
    return [
        ConversationData(messages=[
            Message(role="user", content="I need help"),
            Message(role="assistant", content="I can help with that"),
        ])
    ]
```

## Pull Request Process

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
   black briefx/
   flake8 briefx/
   python -m pytest tests/
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
├── python/                  # Active Python implementation
│   ├── app.py               # Flask web application
│   ├── cli_simple.py        # Command line interface
│   ├── requirements.txt     # Python dependencies
│   ├── tests/               # Test suite
│   └── briefx/              # Python package
│       ├── analysis/        # Analysis pipeline (Clio, clustering, etc.)
│       ├── data/            # Data models and parsers
│       ├── preprocessing/   # Data preprocessing
│       ├── providers/       # LLM provider integrations
│       ├── persistence/     # Database and session storage
│       └── prompts/         # LLM prompt templates
├── briefxai_ui_data/        # Frontend static assets
└── docs/                    # Documentation
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Error messages and logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions considered

## Development Tips

### Adding a New LLM Provider

1. Create a new file in `python/briefx/providers/` (e.g., `myprovider.py`)
2. Extend the `BaseProvider` class from `providers/base.py`
3. Implement the required methods (`complete`, `embed`)
4. Register the provider in `providers/factory.py`

### Adding a New Analysis Feature

1. Add feature logic to the appropriate module under `python/briefx/analysis/`
2. Wire it into the pipeline in `pipeline.py`
3. Add a corresponding API endpoint in `app.py`
4. Write tests in `python/tests/`

### Security

- Never commit API keys or secrets
- Validate all user input at API boundaries
- Follow OWASP guidelines for web application security
- Implement rate limiting for public endpoints

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [Clio Paper](https://arxiv.org/html/2412.13678v1)
- [scikit-learn Documentation](https://scikit-learn.org/)

## License

By contributing to BriefX, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open an issue or contact the maintainers.
