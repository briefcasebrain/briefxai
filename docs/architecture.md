# Architecture Overview

BriefX is a Python web application built with Flask, implementing the [Clio methodology](https://arxiv.org/html/2412.13678v1) for privacy-preserving conversation analysis at scale.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Web UI (Static SPA)                       │
│         (briefxai_ui_data/ — HTML/CSS/JS + D3.js)           │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼────────────────────────────────────┐
│             Web Server (Flask + Gunicorn)                   │
│      (REST API, WebSocket handler, static file serving)     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Core Analysis Engine                       │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Session   │  │ Preprocessing │  │   Provider   │      │
│  │   Manager   │  │   Pipeline    │  │   Manager    │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Clio     │  │  Embeddings  │  │  Clustering  │      │
│  │   Engine    │  │  (sklearn +  │  │  (sklearn +  │      │
│  │             │  │   UMAP)      │  │   k-means)   │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Persistence Layer                        │
│         (SQLite via aiosqlite / PostgreSQL via asyncpg)     │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Web Server Layer (`python/app.py`)

Flask application providing:

- **REST API Endpoints** — CRUD operations for sessions, analyses, and configurations
- **WebSocket Support** — Real-time streaming of analysis results
- **Static File Serving** — SPA assets from `briefxai_ui_data/`
- **CORS Handling** — Cross-origin resource sharing for web clients
- **Request Validation** — Input sanitization and format validation

### 2. Session Manager (`briefx/analysis/session_manager.py`)

Manages the lifecycle of analysis sessions:

- **State Management** — Tracks session states (created, running, paused, completed, failed)
- **Pause/Resume** — Checkpoint-based suspension and resumption
- **Event Broadcasting** — Real-time status updates via WebSocket channels
- **Resource Management** — Concurrent session handling

### 3. Preprocessing Pipeline (`briefx/preprocessing/`)

Data validation and preparation before analysis:

- **Format Validation** — Ensures conversation structure compliance
- **Duplicate Detection** — Identifies and handles repeated conversations
- **PII Detection** — Scans for and masks sensitive information (emails, phone numbers, etc.)
- **Language Detection** — Identifies conversation languages using `langdetect`
- **Smart Preprocessing** — Automated cleaning and normalization

### 4. Provider Manager (`briefx/providers/`)

Multi-provider LLM integration:

- **Provider Abstraction** — Unified `BaseProvider` interface for different LLM providers
- **Supported Providers** — OpenAI, Anthropic, Google Gemini, Ollama, HuggingFace
- **Auto-detection** — Automatically detects available providers from environment variables
- **Fallback Chains** — Graceful fallback to alternative providers on failure

**Key files:**
- `providers/base.py` — BaseProvider interface
- `providers/factory.py` — Provider factory and auto-detection
- `providers/openai.py`, `anthropic.py`, `gemini.py`, `ollama.py`, `huggingface.py`

### 5. Clio Analysis Engine (`briefx/analysis/clio.py`)

Implements the Clio methodology for privacy-preserving conversation analysis:

- **Hierarchical Clustering** — Multi-level conversation grouping
- **Facet Extraction** — LLM-powered extraction of topics, sentiments, intents, entities
- **Privacy Thresholds** — Clusters below a minimum size are suppressed to protect individual privacy
- **Summarization** — Cluster-level summaries at configurable granularity levels
- **Pattern Discovery** — Cross-cluster trend identification

### 6. Embedding & Clustering (`briefx/analysis/`)

Vector-space analysis:

- **Embeddings** — Text embeddings via LLM provider APIs or local HuggingFace models
- **Dimensionality Reduction** — UMAP projections for visualization (`dimensionality.py`)
- **Clustering** — K-means and hierarchical clustering via scikit-learn (`clustering.py`)
- **Similarity Search** — Cosine similarity for nearest-neighbor lookup

### 7. Persistence Layer (`briefx/persistence/`)

Async database layer supporting both SQLite and PostgreSQL:

- **SQLite** — Default for local development via `aiosqlite`
- **PostgreSQL** — Recommended for production via `asyncpg`
- **Session Storage** — Analysis sessions, results, and checkpoints
- **Response Cache** — LLM response caching to reduce API costs
- **Migration System** — Version-controlled schema migrations in `migrations/`

## Data Flow

### Analysis Pipeline

```
Input Conversations (JSON/CSV/text)
        │
        ▼
[Preprocessing]
   - Format validation
   - PII detection & masking
   - Duplicate removal
   - Language detection
        │
        ▼
[Facet Extraction]
   - LLM processing (via provider)
   - Topic/sentiment/entity extraction
        │
        ▼
[Embedding Generation]
   - Text to vector conversion
   - Similarity computation
        │
        ▼
[Dimensionality Reduction]
   - UMAP projection to 2D/3D
        │
        ▼
[Clustering]
   - K-means / hierarchical grouping
   - Centroid calculation
   - Privacy threshold enforcement
        │
        ▼
[Summarization]
   - Cluster-level LLM summaries
   - Cross-cluster pattern analysis
        │
        ▼
Output (Web UI / JSON / CSV / HTML report)
```

### Request Lifecycle

1. **Client Request** — HTTP/WebSocket request received by Flask
2. **Validation** — Request format and size checks
3. **Session Creation** — New session initialized in database
4. **Async Processing** — Analysis tasks run concurrently via asyncio
5. **Real-time Streaming** — Progressive results delivered via WebSocket
6. **Persistence** — Results saved to SQLite/PostgreSQL
7. **Response** — Final response or download link sent to client

## Database Schema

### Core Tables

#### `sessions`
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    config TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

#### `conversations`
```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP
);
```

#### `analysis_results`
```sql
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    conversation_id TEXT,
    facets TEXT,
    embeddings BLOB,
    cluster_id INTEGER,
    created_at TIMESTAMP
);
```

## Frontend (SPA)

The frontend is a static single-page application in `briefxai_ui_data/`:

- **No build step** — Plain HTML, CSS, and JavaScript
- **D3.js** — Force-directed graph visualizations and cluster maps
- **WebSocket client** — Real-time progress and result streaming
- **Progressive disclosure** — Expertise-based UI adaptation for new vs. advanced users
- **Served by Flask** — Static file serving at the root URL

## Performance Characteristics

| Workload | Specification |
|----------|--------------|
| API throughput | 1000+ conversations/minute |
| Memory (10k conversations) | ~512 MB |
| Clustering (100 conversations) | < 5 seconds |
| Facet extraction | ~50ms average per conversation |
| API response time | < 100ms for most endpoints |

### Key Optimizations

- **Asyncio** — Non-blocking I/O for concurrent LLM requests
- **Batch processing** — Groups API calls to minimize round-trips
- **Response caching** — Avoids repeated LLM calls for identical conversations
- **Streaming results** — Progressive delivery rather than waiting for full completion

## Security

- **PII Detection** — Automatic sensitive data identification and masking
- **Input Validation** — All user input validated at API boundaries
- **CORS** — Configurable allowed origins
- **SQL injection prevention** — Parameterized queries throughout

## Extension Points

### Custom LLM Provider

Implement the `BaseProvider` interface:

```python
from briefx.providers.base import BaseProvider

class MyProvider(BaseProvider):
    async def complete(self, prompt: str, **kwargs) -> str:
        ...

    async def embed(self, text: str) -> list[float]:
        ...
```

Register in `providers/factory.py` to make it available via auto-detection.

### Custom Preprocessor

Add a preprocessing step to the pipeline in `briefx/preprocessing/`:

```python
class MyPreprocessor:
    def process(self, conversations: list) -> list:
        # Transform conversations
        return conversations
```

### Custom Analysis Facets

Add new facet extraction prompts to `briefx/prompts/` and reference them in the Clio configuration.
