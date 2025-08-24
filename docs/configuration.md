# Configuration Guide

BriefXAI offers extensive configuration options with smart defaults and automatic model detection. Configure via `.env` files, TOML files, environment variables, and command-line arguments.

## ✨ New Features

- **Smart Model Auto-Detection**: Automatically selects optimal models based on content
- **Cost Optimization**: Built-in features to minimize cloud API costs
- **Google Gemini Support**: Full integration with Google's embedding and LLM models
- **Unified Environment**: Simple `.env` file configuration

## Configuration Priority

Configuration sources are applied in this order (highest to lowest priority):

1. Command-line arguments
2. Environment variables (including `.env` file)
3. User configuration file (`~/.config/briefxai/config.toml`)
4. Project configuration file (`./config.toml`)  
5. System configuration file (`/etc/briefxai/config.toml`)
6. Auto-detected values (models, providers)
7. Default values

## Quick Setup: .env File (Recommended)

The easiest way to configure BriefXAI is with a `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferences
nano .env
```

### Complete .env Example

```bash
# =============================================================================
# BriefXAI Configuration
# =============================================================================

# API Keys (Optional - leave blank for demo mode)
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-gemini-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
HUGGINGFACE_API_KEY=hf_your-huggingface-key-here
COHERE_API_KEY=your-cohere-key-here

# Server Configuration
PORT=8080
HOST=0.0.0.0
FLASK_ENV=production

# Model Defaults (Auto-detection will override based on content)
DEFAULT_LLM_MODEL=gemini-1.5-flash
DEFAULT_EMBEDDING_MODEL=text-embedding-004
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_EMBEDDING_PROVIDER=google

# Cost Control Settings
MAX_CONVERSATIONS=1000
MAX_MESSAGES_PER_CONVERSATION=100
BATCH_SIZE=3
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=30

# Feature Flags
ENABLE_DEMO_MODE=true
ENABLE_AUTO_DETECTION=true
ENABLE_COST_ESTIMATION=true
ENABLE_OCR=true

# Privacy Settings
MIN_PRIVACY_THRESHOLD=3
ENABLE_PII_DETECTION=false
PII_ACTION=mask

# Performance Settings
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
DEDUP_DATA=true

# Optional: Custom API Base URLs
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_BASE_URL=http://localhost:11434
VLLM_BASE_URL=http://localhost:8000

# Logging
LOG_LEVEL=info
VERBOSE=false
```

## Auto-Detection Features 🎯

BriefXAI includes intelligent auto-detection to optimize performance and cost:

### Model Auto-Detection

The system analyzes your conversation content and automatically selects the best models:

```javascript
// Content Analysis Triggers:
- Multilingual content → Multilingual embedding models
- Technical/code content → High-quality models (GPT-4, Gemini Pro)  
- Large datasets (500+ conversations) → Efficient models (Flash, GPT-4o-mini)
- Simple content → Cost-effective models (Flash-8B, GPT-3.5)
- General content → Balanced models (Flash, GPT-4o-mini)
```

### Cost Optimization

Built-in features to minimize cloud costs:

```bash
# Automatic cost controls:
✅ Smart model selection based on complexity
✅ Batch size optimization (reduces failed requests)  
✅ Rate limiting to avoid hitting API limits
✅ Deduplication to reduce processing
✅ Cost estimation before analysis
```

### Provider Auto-Detection

The system automatically detects available providers:

```bash
# Detection order:
1. Ollama (if running locally) → Free
2. Google Gemini (if API key provided) → Most cost-effective  
3. OpenAI (if API key provided) → High quality
4. Other providers (HuggingFace, Anthropic, etc.)
5. Demo mode (if no providers available) → No cost
```

## Configuration File Format

BriefXAI uses TOML format for advanced configuration. Here's a complete example:

```toml
# config.toml - Complete configuration example

# Server Configuration
[server]
host = "127.0.0.1"              # Bind address
port = 8080                     # Port number
workers = 4                     # Number of worker threads
max_connections = 1000          # Maximum concurrent connections
request_timeout_ms = 30000      # Request timeout in milliseconds
cors_enabled = true             # Enable CORS
cors_origins = ["*"]            # Allowed CORS origins

# Database Configuration
[database]
path = "data/briefxai.db"       # SQLite database path
max_connections = 25            # Connection pool size
busy_timeout_ms = 5000          # Busy timeout for SQLite
journal_mode = "WAL"            # SQLite journal mode
synchronous = "NORMAL"          # SQLite synchronous mode
cache_size_kb = 2000           # SQLite cache size in KB

# Analysis Configuration
[analysis]
batch_size = 100                # Conversations per batch
max_concurrent_requests = 10    # Concurrent LLM requests
timeout_seconds = 300           # Analysis timeout
max_retries = 3                 # Retry attempts for failures
retry_delay_ms = 1000          # Delay between retries
cache_enabled = true            # Enable result caching
cache_ttl_seconds = 3600       # Cache time-to-live

# Preprocessing Configuration
[preprocessing]
detect_duplicates = true        # Enable duplicate detection
duplicate_threshold = 0.95      # Similarity threshold for duplicates
detect_pii = true              # Enable PII detection
pii_action = "mask"            # PII action: "mask", "remove", or "flag"
validate_format = true          # Validate conversation format
min_message_length = 1          # Minimum message length
max_message_length = 10000      # Maximum message length
detect_language = true          # Enable language detection
supported_languages = ["en", "es", "fr", "de", "zh", "ja"]

# Facet Extraction Configuration
[facets]
enabled = true                  # Enable facet extraction
default_facets = [              # Default facets to extract
    "sentiment",
    "topics",
    "entities",
    "intents",
    "issues",
    "resolutions"
]
custom_prompts_path = "prompts/" # Path to custom prompt templates
max_facets_per_conversation = 50 # Maximum facets per conversation

# Embedding Configuration
[embeddings]
model = "text-embedding-ada-002" # Embedding model
dimensions = 1536                # Embedding dimensions
batch_size = 100                 # Embedding batch size
cache_enabled = true             # Cache embeddings
normalize = true                 # Normalize embeddings

# Clustering Configuration
[clustering]
algorithm = "kmeans"             # Clustering algorithm: "kmeans" or "hierarchical"
min_clusters = 2                 # Minimum number of clusters
max_clusters = 20                # Maximum number of clusters
auto_detect = true               # Auto-detect optimal clusters
distance_metric = "cosine"       # Distance metric
max_iterations = 100             # Maximum iterations for k-means
convergence_threshold = 0.0001   # Convergence threshold

# Provider Configuration - OpenAI
[providers.openai]
enabled = true                   # Enable this provider
api_key = "${OPENAI_API_KEY}"   # API key (can use env vars)
base_url = "https://api.openai.com/v1"
model = "gpt-4"                 # Default model
temperature = 0.7                # Generation temperature
max_tokens = 2000               # Maximum tokens per request
timeout_ms = 30000              # Request timeout
max_retries = 3                 # Retry attempts
rate_limit_rpm = 60             # Requests per minute limit

# Provider Configuration - Ollama (Local)
[providers.ollama]
enabled = false                  # Enable this provider
base_url = "http://localhost:11434"
model = "llama2"                # Default model
temperature = 0.7
max_tokens = 2000
timeout_ms = 60000              # Longer timeout for local models
max_concurrent = 4              # Concurrent requests

# Provider Configuration - Custom
[providers.custom]
enabled = false
base_url = "https://your-api.com"
api_key = "${CUSTOM_API_KEY}"
model = "your-model"
headers = { "X-Custom-Header" = "value" }

# Load Balancing Configuration
[load_balancing]
strategy = "round_robin"         # Strategy: "round_robin", "least_latency", "weighted", "cost_optimized"
health_check_interval_seconds = 30
failover_enabled = true          # Enable automatic failover
failover_threshold = 3           # Failures before failover

# Circuit Breaker Configuration
[circuit_breaker]
enabled = true                   # Enable circuit breaker
failure_threshold = 5            # Failures to open circuit
success_threshold = 2            # Successes to close circuit
timeout_seconds = 60             # Time before half-open
half_open_requests = 3           # Requests in half-open state

# Monitoring Configuration
[monitoring]
enabled = true                   # Enable monitoring
metrics_port = 9090             # Prometheus metrics port
collect_interval_seconds = 10   # Metrics collection interval
export_format = "prometheus"    # Export format

# Logging Configuration
[logging]
level = "info"                   # Log level: "trace", "debug", "info", "warn", "error"
format = "json"                  # Log format: "json" or "text"
output = "stdout"                # Output: "stdout", "stderr", or file path
file_rotation = "daily"          # Rotation: "daily", "size", or "never"
max_file_size_mb = 100          # Maximum log file size
max_files = 7                    # Maximum number of log files
include_timestamp = true         # Include timestamps
include_location = false         # Include code location

# Security Configuration
[security]
api_key_required = false         # Require API key authentication
api_keys_file = "api_keys.json" # Path to API keys file
rate_limiting_enabled = true     # Enable rate limiting
rate_limit_rpm = 100            # Requests per minute per client
max_request_size_mb = 10        # Maximum request size
allowed_origins = ["*"]          # Allowed origins for CORS
enable_tls = false              # Enable TLS/SSL
tls_cert_path = ""              # TLS certificate path
tls_key_path = ""               # TLS key path

# Export Configuration
[export]
formats = ["json", "csv", "html"] # Supported export formats
max_export_size_mb = 100         # Maximum export size
compression_enabled = true        # Enable compression
include_metadata = true          # Include metadata in exports

# Template Configuration
[templates]
path = "templates/"              # Template directory
builtin = [                     # Built-in templates
    "customer_support",
    "sales",
    "medical",
    "technical_support"
]
custom_enabled = true           # Enable custom templates
validation_strict = true        # Strict template validation

# Cache Configuration
[cache]
type = "memory"                 # Cache type: "memory" or "disk"
max_size_mb = 500              # Maximum cache size
ttl_seconds = 3600             # Default TTL
eviction_policy = "lru"        # Eviction policy: "lru", "lfu", "fifo"
```

## Environment Variables

All configuration values can be overridden using environment variables. Use the following format:

```bash
BRIEFXAI_<SECTION>_<KEY>=value
```

Examples:

```bash
# Server configuration
export BRIEFXAI_SERVER_HOST=0.0.0.0
export BRIEFXAI_SERVER_PORT=8080

# Database configuration
export BRIEFXAI_DATABASE_PATH=/var/lib/briefxai/data.db

# Provider configuration
export BRIEFXAI_PROVIDERS_OPENAI_API_KEY=sk-your-api-key
export BRIEFXAI_PROVIDERS_OPENAI_MODEL=gpt-4

# Logging configuration
export BRIEFXAI_LOGGING_LEVEL=debug
```

## Command-Line Arguments

Override configuration at runtime:

```bash
briefxai serve \
  --host 0.0.0.0 \
  --port 8080 \
  --config custom-config.toml \
  --log-level debug
```

Available arguments:

```bash
briefxai serve --help

OPTIONS:
    --host <HOST>           Server host address
    --port <PORT>           Server port
    --config <PATH>         Configuration file path
    --database <PATH>       Database file path
    --log-level <LEVEL>     Log level (trace|debug|info|warn|error)
    --workers <NUM>         Number of worker threads
    --api-key <KEY>         API key for authentication
```

## Configuration Templates

### Minimal Configuration

```toml
# Minimal config for quick start
[server]
port = 8080

[providers.openai]
api_key = "your-api-key"
```

### Development Configuration

```toml
# Development configuration with debugging
[server]
host = "127.0.0.1"
port = 8080

[database]
path = "dev.db"

[logging]
level = "debug"
format = "text"

[providers.ollama]
enabled = true
base_url = "http://localhost:11434"
```

### Production Configuration

```toml
# Production configuration with security and monitoring
[server]
host = "0.0.0.0"
port = 443
workers = 16

[database]
path = "/var/lib/briefxai/prod.db"
max_connections = 50

[security]
api_key_required = true
enable_tls = true
tls_cert_path = "/etc/ssl/certs/briefxai.crt"
tls_key_path = "/etc/ssl/private/briefxai.key"

[monitoring]
enabled = true
metrics_port = 9090

[logging]
level = "warn"
format = "json"
output = "/var/log/briefxai/app.log"
```

## Provider-Specific Configuration

### OpenAI

```toml
[providers.openai]
api_key = "${OPENAI_API_KEY}"
organization_id = "org-xxxxx"  # Optional
model = "gpt-4"
temperature = 0.7
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0
```

### Anthropic Claude

```toml
[providers.anthropic]
api_key = "${ANTHROPIC_API_KEY}"
model = "claude-3-opus-20240229"
max_tokens = 4096
temperature = 0.7
```

### Local Models (Ollama)

```toml
[providers.ollama]
base_url = "http://localhost:11434"
model = "mixtral"
num_ctx = 4096          # Context window size
num_gpu = 1             # Number of GPUs to use
num_thread = 8          # Number of CPU threads
```

### Azure OpenAI

```toml
[providers.azure]
api_key = "${AZURE_API_KEY}"
endpoint = "https://your-resource.openai.azure.com"
deployment_name = "your-deployment"
api_version = "2023-12-01-preview"
```

## Advanced Configuration

### Multi-Provider Setup

```toml
# Primary provider
[providers.openai]
enabled = true
api_key = "${OPENAI_API_KEY}"
model = "gpt-4"
priority = 1

# Fallback provider
[providers.ollama]
enabled = true
base_url = "http://localhost:11434"
model = "llama2"
priority = 2

# Load balancing
[load_balancing]
strategy = "weighted"
weights = { openai = 70, ollama = 30 }
```

### Custom Prompt Templates

```toml
[facets]
custom_prompts_path = "prompts/"

[facets.custom_prompts.customer_satisfaction]
prompt = """
Analyze the customer satisfaction level:
- Very Satisfied
- Satisfied
- Neutral
- Dissatisfied
- Very Dissatisfied
"""
```

### Performance Tuning

```toml
[performance]
# Connection pooling
connection_pool_size = 50
connection_timeout_ms = 5000

# Request handling
max_concurrent_requests = 100
request_queue_size = 1000
request_timeout_ms = 30000

# Memory management
max_memory_mb = 4096
gc_interval_seconds = 60

# Threading
worker_threads = 16
blocking_threads = 512
```

## Validation

Validate your configuration:

```bash
briefxai config validate --file config.toml
```

Check effective configuration:

```bash
briefxai config show --format json
```

## Troubleshooting

### Common Issues

1. **Provider API Key Issues**
   ```toml
   # Use environment variable
   api_key = "${OPENAI_API_KEY}"
   # Or direct value (not recommended for production)
   api_key = "sk-xxxxx"
   ```

2. **Database Permission Issues**
   ```toml
   [database]
   path = "/tmp/briefxai.db"  # Use writable directory
   ```

3. **Port Already in Use**
   ```toml
   [server]
   port = 8081  # Use alternative port
   ```

### Debug Configuration

Enable debug logging to troubleshoot:

```toml
[logging]
level = "trace"
include_location = true

[debug]
print_config_on_start = true
validate_config = true
```

## Best Practices

1. **Use Environment Variables for Secrets**
   - Never commit API keys to version control
   - Use `${VAR_NAME}` syntax in config files

2. **Separate Development and Production Configs**
   - `config.dev.toml` for development
   - `config.prod.toml` for production

3. **Monitor Resource Usage**
   - Set appropriate limits for your hardware
   - Monitor metrics to optimize settings

4. **Regular Backups**
   - Configure database backups
   - Export configurations for disaster recovery

5. **Security Hardening**
   - Enable API key authentication in production
   - Use TLS for encrypted connections
   - Implement rate limiting

## Migration from Previous Versions

### From v1.x to v2.x

1. Configuration file format changed from JSON to TOML
2. New required fields:
   - `providers` section is now mandatory
   - `analysis.batch_size` replaces `batch_size`

3. Deprecated fields:
   - `llm_provider` → use `providers` section
   - `embedding_model` → use `embeddings.model`

Migration script:

```bash
briefxai config migrate --from v1 --input old-config.json --output config.toml
```