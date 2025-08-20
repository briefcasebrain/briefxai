# Architecture Overview

BriefXAI is built with a modular, scalable architecture designed for high-performance conversation analysis. This document provides a comprehensive overview of the system design, components, and data flow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Web UI                              │
│  (React-based Dashboard, WebSocket Client, Visualizations)  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼────────────────────────────────────┐
│                     Web Server (Axum)                       │
│  (REST API, WebSocket Handler, Static File Serving)         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Core Analysis Engine                      │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Session   │  │ Preprocessing │  │   Provider   │      │
│  │   Manager   │  │   Pipeline    │  │   Manager    │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Facet    │  │  Embeddings  │  │  Clustering  │      │
│  │  Extractor  │  │  Generator   │  │   Engine     │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Persistence Layer                        │
│            (SQLite with Migration System)                   │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Web Server Layer

The web server is built on Axum, providing:

- **REST API Endpoints**: CRUD operations for sessions, analyses, and configurations
- **WebSocket Support**: Real-time streaming of analysis results
- **Static File Serving**: UI assets and documentation
- **CORS Handling**: Cross-origin resource sharing for web clients
- **Request Validation**: Input sanitization and validation

**Key Files:**
- `src/web.rs` - Main web server implementation
- `src/web_clio.rs` - Clio-specific endpoints

### 2. Session Manager

Manages the lifecycle of analysis sessions:

- **State Management**: Tracks session states (created, running, paused, completed, failed)
- **Pause/Resume**: Checkpoint-based suspension and resumption
- **Recovery**: Automatic recovery from failures
- **Event Broadcasting**: Real-time status updates via channels
- **Resource Management**: Memory and CPU allocation per session

**Key Files:**
- `src/analysis/session_manager.rs` - Core session management
- `src/analysis/streaming.rs` - Real-time result streaming

### 3. Preprocessing Pipeline

Data validation and preparation:

- **Format Validation**: Ensures conversation structure compliance
- **Duplicate Detection**: Identifies and handles duplicate conversations
- **PII Detection**: Scans for sensitive information
- **Language Detection**: Identifies conversation languages
- **Quality Assessment**: Scores data quality and completeness
- **Smart Preprocessing**: Automated cleaning and normalization

**Key Files:**
- `src/preprocessing/mod.rs` - Pipeline orchestration
- `src/preprocessing/validators.rs` - Validation components
- `src/preprocessing/language_detector.rs` - Language detection
- `src/preprocessing/smart_preprocessor.rs` - Data cleaning

### 4. Provider Manager

Multi-provider LLM integration:

- **Provider Abstraction**: Unified interface for different LLM providers
- **Load Balancing**: Multiple strategies (round-robin, least-latency, cost-optimized)
- **Circuit Breakers**: Fault tolerance with automatic recovery
- **Rate Limiting**: Request throttling and quota management
- **Fallback Chains**: Automatic failover to alternative providers
- **Cost Tracking**: Usage and cost monitoring per provider

**Key Files:**
- `src/llm/provider_manager.rs` - Provider management
- `src/llm.rs` - LLM interface definitions

### 5. Analysis Components

#### Facet Extraction
Extracts structured insights from conversations:
- Topic identification
- Sentiment analysis
- Entity extraction
- Issue detection
- Resolution tracking

**File:** `src/facets.rs`

#### Embedding Generation
Creates vector representations:
- Text embeddings for semantic search
- Conversation embeddings for clustering
- Multiple embedding model support
- Caching for performance

**File:** `src/embeddings.rs`

#### Clustering Engine
Groups similar conversations:
- K-means clustering
- Hierarchical clustering
- Dynamic cluster sizing
- Outlier detection

**File:** `src/clustering.rs`

### 6. Persistence Layer

SQLite-based storage with:

- **Migration System**: Version-controlled schema updates
- **Transaction Support**: ACID compliance
- **Indexing**: Optimized query performance
- **Caching**: Multi-level caching strategy
- **Backup/Restore**: Data protection and recovery

**Key Files:**
- `src/persistence_v2.rs` - Enhanced persistence implementation
- `migrations/` - SQL migration scripts

## Data Flow

### 1. Analysis Pipeline

```
Input Conversations
        │
        ▼
[Preprocessing]
   - Validation
   - PII Detection
   - Language Detection
        │
        ▼
[Facet Extraction]
   - LLM Processing
   - Pattern Detection
        │
        ▼
[Embedding Generation]
   - Vector Creation
   - Similarity Computation
        │
        ▼
[Clustering]
   - Group Formation
   - Centroid Calculation
        │
        ▼
[Result Aggregation]
   - Statistics
   - Visualizations
        │
        ▼
Output (JSON/CSV/UI)
```

### 2. Request Lifecycle

1. **Client Request**: HTTP/WebSocket request received
2. **Authentication**: Request validation and authorization
3. **Session Creation**: New session initialized in database
4. **Task Queuing**: Analysis tasks queued for processing
5. **Parallel Processing**: Concurrent task execution
6. **Result Streaming**: Progressive result delivery
7. **Persistence**: Results saved to database
8. **Response**: Final response sent to client

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

#### `providers`
```sql
CREATE TABLE providers (
    id TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    health_status TEXT,
    metrics TEXT,
    updated_at TIMESTAMP
);
```

## Performance Optimizations

### 1. Concurrency
- **Tokio Runtime**: Async/await for I/O operations
- **Thread Pools**: CPU-bound task parallelization
- **Channel-based Communication**: Lock-free message passing

### 2. Caching
- **Memory Cache**: Hot data in RAM
- **Database Cache**: Query result caching
- **Embedding Cache**: Reuse computed embeddings
- **Provider Response Cache**: Reduce API calls

### 3. Batching
- **Request Batching**: Group API calls
- **Database Batching**: Bulk inserts/updates
- **Processing Batching**: Optimize throughput

### 4. Resource Management
- **Connection Pooling**: Database connection reuse
- **Memory Limits**: Prevent OOM conditions
- **Rate Limiting**: Prevent resource exhaustion

## Security Considerations

### 1. Data Protection
- **PII Detection**: Automatic sensitive data identification
- **Data Encryption**: At-rest and in-transit encryption
- **Access Control**: Role-based permissions
- **Audit Logging**: Complete activity tracking

### 2. Input Validation
- **SQL Injection Prevention**: Parameterized queries
- **XSS Prevention**: Input sanitization
- **Path Traversal Prevention**: File access restrictions
- **Size Limits**: Prevent DoS attacks

### 3. Provider Security
- **API Key Management**: Secure credential storage
- **TLS/SSL**: Encrypted provider communication
- **Request Signing**: Authentication verification

## Monitoring & Observability

### 1. Metrics Collection
- **Prometheus Integration**: Time-series metrics
- **Custom Metrics**: Business-specific KPIs
- **Performance Metrics**: Latency, throughput, errors

### 2. Logging
- **Structured Logging**: JSON-formatted logs
- **Log Levels**: Configurable verbosity
- **Correlation IDs**: Request tracing
- **Error Tracking**: Detailed error context

### 3. Health Checks
- **Liveness Probe**: System availability
- **Readiness Probe**: Service readiness
- **Dependency Checks**: Provider/database status

## Scalability

### Horizontal Scaling
- **Stateless Design**: No server-side session state
- **Database Sharding**: Data partitioning support
- **Load Balancer Ready**: Multiple instance support

### Vertical Scaling
- **Resource Configuration**: Tunable limits
- **Memory Management**: Efficient allocation
- **Connection Pooling**: Scalable connections

## Extension Points

### 1. Custom Providers
Implement the `LLMProvider` trait:
```rust
#[async_trait]
pub trait LLMProvider {
    async fn complete(&self, prompt: &str) -> Result<String>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}
```

### 2. Custom Preprocessors
Implement the `Preprocessor` trait:
```rust
pub trait Preprocessor {
    fn process(&self, data: &mut ConversationData) -> Result<()>;
}
```

### 3. Custom Analyzers
Implement the `Analyzer` trait:
```rust
#[async_trait]
pub trait Analyzer {
    async fn analyze(&self, conversation: &Conversation) -> Result<Analysis>;
}
```

## Future Considerations

### Planned Enhancements
- **Distributed Processing**: Multi-node support
- **Stream Processing**: Real-time conversation analysis
- **Machine Learning Pipeline**: Custom model training
- **GraphQL API**: Flexible query interface
- **Plugin System**: Dynamic extension loading

### Research Areas
- **Advanced NLP**: Deeper semantic understanding
- **Anomaly Detection**: Unusual pattern identification
- **Predictive Analytics**: Trend forecasting
- **Multi-modal Analysis**: Voice/video support