# Development Guide

This guide covers the development workflow, testing strategies, and best practices for contributing to BriefX.

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Git
- Optional: Docker for containerized development

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/briefcasebrain/briefxai.git
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

4. **Set up development tools:**
   ```bash
   pip install black flake8 pytest pytest-cov
   ```

5. **Set API keys:**
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

## Project Structure

```
briefxai/
├── python/                  # Active Python implementation
│   ├── app.py               # Flask web application entry point
│   ├── cli_simple.py        # Command line interface
│   ├── cli.py               # Extended CLI
│   ├── requirements.txt     # Python dependencies
│   ├── setup.py             # Package setup
│   ├── tests/               # Test suite
│   └── briefx/              # Python package
│       ├── analysis/        # Analysis modules
│       │   ├── clio.py      # Clio methodology implementation
│       │   ├── clustering.py # Clustering algorithms
│       │   ├── dimensionality.py # UMAP/dimensionality reduction
│       │   ├── pipeline.py  # Analysis pipeline orchestration
│       │   └── session_manager.py # Session lifecycle management
│       ├── config.py        # Configuration handling
│       ├── data/            # Data models and parsers
│       │   ├── models.py    # ConversationData, Message, etc.
│       │   └── parsers.py   # JSON, CSV, text parsers
│       ├── error_recovery.py # Error handling and recovery
│       ├── examples.py      # Example data generation
│       ├── monitoring.py    # System monitoring
│       ├── persistence/     # Database and session storage
│       ├── preprocessing/   # Data preprocessing pipeline
│       ├── providers/       # LLM provider integrations
│       │   ├── base.py      # BaseProvider interface
│       │   ├── factory.py   # Provider factory and registry
│       │   ├── openai.py    # OpenAI provider
│       │   ├── anthropic.py # Anthropic provider
│       │   ├── gemini.py    # Google Gemini provider
│       │   ├── ollama.py    # Ollama (local) provider
│       │   └── huggingface.py # HuggingFace provider
│       ├── prompts/         # LLM prompt templates
│       └── utils.py         # Shared utilities
├── briefxai_ui_data/        # Frontend static assets (HTML/CSS/JS)
├── docs/                    # Documentation
├── migrations/              # Database migration scripts
└── Dockerfile               # Container build configuration
```

## Development Workflow

### Running the Application

```bash
# Development server (auto-reloads on file changes via Flask debug mode)
FLASK_DEBUG=1 python app.py

# Standard server
python app.py

# Custom port
BRIEFX_PORT=3000 python app.py

# With debug logging
BRIEFX_LOG_LEVEL=DEBUG python app.py
```

### Code Formatting

```bash
# Format all Python code
black briefx/

# Check formatting without changing files
black briefx/ --check
```

### Linting

```bash
# Run flake8
flake8 briefx/

# With specific rules
flake8 briefx/ --max-line-length=100
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_complete.py

# Run tests matching a name pattern
python -m pytest tests/ -k "test_clustering"

# Run with code coverage
python -m pytest tests/ --cov=briefx --cov-report=html

# Stop on first failure
python -m pytest tests/ -x
```

### Writing Tests

Place tests in `python/tests/`. Use pytest fixtures for shared setup:

```python
# tests/test_analysis.py
import pytest
from briefx.data.models import ConversationData, Message
from briefx.examples import generate_example_conversations

@pytest.fixture
def sample_conversations():
    return generate_example_conversations(count=5, seed=42)

def test_conversation_has_messages(sample_conversations):
    for conv in sample_conversations:
        assert len(conv.messages) > 0

def test_message_roles(sample_conversations):
    valid_roles = {"user", "assistant", "system"}
    for conv in sample_conversations:
        for msg in conv.messages:
            assert msg.role in valid_roles
```

### Test Organization

```
tests/
├── test_complete.py      # End-to-end integration tests
├── test_models.py        # Data model tests
├── test_pipeline.py      # Analysis pipeline tests
├── test_providers.py     # LLM provider tests
└── test_preprocessing.py # Preprocessing pipeline tests
```

## Adding Features

### Adding a New LLM Provider

1. Create `python/briefx/providers/myprovider.py`:

```python
from .base import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self, api_key: str, model: str = "my-model"):
        self.api_key = api_key
        self.model = model

    async def complete(self, prompt: str, **kwargs) -> str:
        # Call your provider's API
        ...

    async def embed(self, text: str) -> list[float]:
        # Generate embeddings
        ...
```

2. Register in `python/briefx/providers/factory.py`:

```python
from .myprovider import MyProvider

# Add to the provider registry
```

### Adding a New Analysis Feature

1. Add the feature logic to `python/briefx/analysis/`
2. Wire it into the pipeline in `pipeline.py`
3. Add a corresponding API endpoint in `app.py`
4. Write tests in `python/tests/`

### Modifying the Frontend

The frontend assets live in `briefxai_ui_data/`. This is a static SPA served by Flask. Edit the HTML, CSS, and JS files there and refresh the browser — no build step required.

## API Development

### Testing Endpoints

```bash
# Health check
curl http://localhost:8080/api/monitoring/health

# Upload conversations
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"conversations": [{"messages": [{"role": "user", "content": "Hello"}]}]}'

# Get results
curl http://localhost:8080/api/analysis/results
```

### WebSocket Testing

```javascript
// Browser console or Node.js
const ws = new WebSocket('ws://localhost:8080/ws/analysis');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'subscribe',
        session_id: 'test_session'
    }));
};

ws.onmessage = (event) => {
    console.log('Received:', JSON.parse(event.data));
};
```

## Database Management

BriefX uses SQLite by default. The database file is created automatically at `python/briefx.db`.

### Inspecting the Database

```bash
# Open SQLite shell
sqlite3 python/briefx.db

# List tables
.tables

# Check sessions
SELECT id, state, created_at FROM sessions;

# Exit
.quit
```

### Switching to PostgreSQL

```bash
export DATABASE_URL="postgresql://user:password@localhost/briefx"
python app.py
```

Migration scripts are in `migrations/` and run automatically on startup.

## Debugging

### Enable Debug Logging

```bash
BRIEFX_LOG_LEVEL=DEBUG python app.py
```

### Using Python Debugger

```python
# Add a breakpoint in your code
import pdb; pdb.set_trace()
# or in Python 3.7+
breakpoint()
```

### Checking Provider Connectivity

```bash
# Test via API
curl -X POST http://localhost:8080/api/providers/openai/test
```

## Continuous Integration

Tests are expected to pass before merging PRs. Run the full test suite locally before submitting:

```bash
black briefx/ --check && flake8 briefx/ && python -m pytest tests/ -v
```

## Release Process

1. Update `CHANGELOG.md` with the new version's changes
2. Bump the version in `setup.py`
3. Create a git tag: `git tag -a v2.x.x -m "Release v2.x.x"`
4. Push the tag: `git push origin v2.x.x`

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [black Documentation](https://black.readthedocs.io/)
- [Clio Paper](https://arxiv.org/html/2412.13678v1)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
