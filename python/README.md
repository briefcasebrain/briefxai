# BriefX Python Implementation

The Python implementation of BriefX provides the same powerful conversation analysis capabilities in a pure Python package.

## Quick Start

1. **Install**
   ```bash
   pip install -e .
   ```

2. **Set API Key**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Run Server**
   ```bash
   python app.py
   ```

## Usage

### Command Line Interface
```bash
briefx analyze conversations.json
briefx serve --port 8080
briefx --help
```

### Python API
```python
from briefx.data.models import ConversationData, Message
from briefx.analysis.pipeline import AnalysisPipeline

# Create pipeline
pipeline = AnalysisPipeline()

# Analyze conversations
results = pipeline.analyze(conversations)
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_complete.py
```

## Development

```bash
# Install in development mode
pip install -e .

# Run linting
flake8 briefx/

# Format code
black briefx/
```

See the main [README.md](../README.md) for complete documentation.