# BriefX Project Structure

This repository contains both Rust and Python implementations of the BriefX conversation analysis platform.

> **Important**: The **Python implementation is the active, maintained version**. The Rust implementation is deprecated and no longer maintained.

## Directory Structure

```
briefxai/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT license
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
│
├── src/                         # Rust implementation (DEPRECATED)
│   ├── lib.rs                   # Main Rust library
│   ├── main.rs                  # Rust binary entry point
│   ├── analysis/                # Analysis modules
│   ├── clustering.rs            # Clustering algorithms
│   ├── embeddings.rs            # Embedding generation
│   ├── facets.rs                # Facet extraction
│   ├── preprocessing/           # Data preprocessing
│   ├── web.rs                   # Web server
│   └── ...                      # Other Rust modules
│
├── python/                      # Python implementation (ACTIVE)
│   ├── README.md                # Python-specific documentation
│   ├── setup.py                 # Python package setup
│   ├── requirements.txt         # Python dependencies
│   ├── app.py                   # Flask web application
│   ├── cli_simple.py            # Command line interface
│   ├── tests/                   # Python tests
│   │   └── test_updated.py      # Updated test suite
│   └── briefx/                  # Python package
│       ├── __init__.py          # Package initialization
│       ├── data/                # Data models and parsers
│       ├── analysis/            # Analysis pipeline
│       ├── preprocessing/       # Data preprocessing
│       ├── providers/           # LLM provider integrations
│       ├── examples.py          # Example data generation
│       └── ...                  # Other Python modules
│
├── docs/                        # Documentation
│   ├── api.md                   # API documentation
│   ├── configuration.md         # Configuration guide
│   ├── getting-started.md       # Getting started guide
│   └── ...                      # Other documentation
│
├── tests/                       # Rust tests
│   ├── common/                  # Common test utilities
│   └── ...                      # Test files
│
├── scripts/                     # Build and deployment scripts
├── deployment/                  # Deployment configurations
├── assets/                      # Web UI assets
├── templates/                   # HTML templates
└── target/                      # Rust build artifacts
```

## Implementation Status

| Feature | Rust (Deprecated) | Python (Active) |
|---------|-------------------|-----------------|
| Status | ⚠️ **Deprecated** | ✅ **Active** |
| Maintenance | None | Full support |
| Performance | High (5000+ conversations/min) | Good (1000+ conversations/min) |
| Memory Usage | Low (~128MB for 10k conversations) | Moderate (~512MB for 10k conversations) |
| Development Speed | Moderate | Fast |
| Deployment | Single binary | Python runtime |
| Dependencies | Minimal | Standard Python ecosystem |

## Getting Started

### Python Implementation (Recommended)
```bash
cd python/
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python app.py
```

### Command Line Usage
```bash
cd python/
python cli_simple.py --help
python cli_simple.py generate --count 5
python cli_simple.py test
```

### Testing
```bash
cd python/
python tests/test_updated.py
```

The Python implementation provides all core functionality with active maintenance and support.