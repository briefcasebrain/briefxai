# BriefX Project Structure

BriefX is a Python web application for privacy-preserving conversation analysis using the [Clio methodology](https://arxiv.org/html/2412.13678v1).

## Directory Structure

```
briefxai/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT license
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── Dockerfile                   # Container build (Python app)
│
├── python/                      # Python implementation (active)
│   ├── app.py                   # Flask web application entry point
│   ├── cli_simple.py            # Command line interface
│   ├── cli.py                   # Extended CLI
│   ├── requirements.txt         # Python dependencies
│   ├── setup.py                 # Python package setup
│   ├── tests/                   # Test suite
│   │   └── test_complete.py     # Integration test suite
│   └── briefx/                  # Python package
│       ├── __init__.py
│       ├── config.py            # Configuration handling
│       ├── utils.py             # Shared utilities
│       ├── examples.py          # Example data generation
│       ├── monitoring.py        # System monitoring
│       ├── error_recovery.py    # Error handling and recovery
│       ├── analysis/            # Analysis modules
│       │   ├── clio.py          # Clio methodology implementation
│       │   ├── clustering.py    # Clustering algorithms (k-means, hierarchical)
│       │   ├── dimensionality.py # UMAP dimensionality reduction
│       │   ├── pipeline.py      # Analysis pipeline orchestration
│       │   └── session_manager.py # Session lifecycle management
│       ├── data/                # Data models and parsers
│       │   ├── models.py        # ConversationData, Message, etc.
│       │   └── parsers.py       # JSON, CSV, text parsers
│       ├── preprocessing/       # Data preprocessing pipeline
│       │   └── ...              # Validation, PII detection, language detection
│       ├── providers/           # LLM provider integrations
│       │   ├── base.py          # BaseProvider interface
│       │   ├── factory.py       # Provider factory and auto-detection
│       │   ├── openai.py        # OpenAI provider
│       │   ├── anthropic.py     # Anthropic provider
│       │   ├── gemini.py        # Google Gemini provider
│       │   ├── ollama.py        # Ollama (local) provider
│       │   └── huggingface.py   # HuggingFace provider
│       ├── persistence/         # Database and session storage
│       │   └── ...              # SQLite and PostgreSQL backends
│       └── prompts/             # LLM prompt templates
│
├── briefxai_ui_data/            # Frontend static assets
│   ├── index.html               # Main SPA entry point
│   ├── static/                  # CSS, JavaScript
│   └── ...                      # Other UI assets
│
├── docs/                        # Documentation
│   ├── quickstart.md            # 5-minute quick start guide
│   ├── getting-started.md       # Full getting started guide
│   ├── api.md                   # REST and WebSocket API reference
│   ├── architecture.md          # System architecture overview
│   ├── configuration.md         # Configuration guide
│   ├── development.md           # Development workflow guide
│   ├── deployment.md            # Cloud deployment guide (GCP, AWS, Vercel)
│   └── advanced-features.md     # Advanced usage and features
│
├── migrations/                  # Database migration scripts
├── scripts/                     # Build and utility scripts
│
└── src/                         # Rust implementation (deprecated, unmaintained)
```

## Key Files

| File | Purpose |
|------|---------|
| `python/app.py` | Flask application — add API endpoints here |
| `python/briefx/analysis/clio.py` | Core Clio analysis engine |
| `python/briefx/analysis/pipeline.py` | Pipeline orchestration |
| `python/briefx/providers/factory.py` | Add/register new LLM providers here |
| `python/briefx/data/models.py` | Core data models (ConversationData, Message) |
| `briefxai_ui_data/index.html` | Frontend entry point |

## Getting Started

```bash
cd python/
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python app.py
```

See [docs/quickstart.md](docs/quickstart.md) for a full walkthrough.
