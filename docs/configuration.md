# Configuration Guide

BriefX is configured primarily through environment variables. This guide covers all available options.

## Configuration Priority

Configuration sources are applied in this order (highest to lowest priority):

1. Environment variables
2. `.env` file in the project root (loaded via `python-dotenv`)
3. Default values

## Environment Variables

### API Keys

Set at least one LLM provider API key to enable analysis features:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="AIza..."

# Ollama (local — no key needed, just set the host)
export OLLAMA_HOST="http://localhost:11434"

# HuggingFace (for local embedding models)
export HUGGINGFACE_API_KEY="hf_..."

# Briefcase AI platform (optional)
export BRIEFCASE_API_KEY="sk-..."
```

### Server Configuration

```bash
# Bind address (default: 127.0.0.1)
export BRIEFX_HOST="0.0.0.0"

# Port (default: 8080)
export BRIEFX_PORT="8080"

# Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
export BRIEFX_LOG_LEVEL="INFO"
```

### Database Configuration

```bash
# SQLite (default — no configuration needed)
export DATABASE_URL="sqlite:///briefx.db"

# PostgreSQL (recommended for production)
export DATABASE_URL="postgresql://user:password@localhost/briefx"

# PostgreSQL with Cloud SQL (Google Cloud)
export DATABASE_URL="postgresql://user:pass@/briefx?host=/cloudsql/PROJECT:REGION:INSTANCE"
```

BriefX auto-detects whether to use SQLite or PostgreSQL based on the `DATABASE_URL` prefix. The database schema is created automatically on first run.

## Using a .env File

For local development, create a `.env` file in the `python/` directory:

```bash
# python/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
BRIEFX_PORT=8080
BRIEFX_LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///briefx.db
```

> **Never commit `.env` files to version control.** Add `.env` to your `.gitignore`.

## Provider Configuration

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Supported models: `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Supported models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`

### Google Gemini

```bash
export GOOGLE_API_KEY="AIza..."
```

Supported models: `gemini-1.5-pro`, `gemini-1.5-flash`

### Ollama (Local)

No API key required. Install Ollama and pull a model:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama2
# or
ollama pull mistral

# BriefX will auto-detect Ollama at http://localhost:11434
```

To use a non-default Ollama host:
```bash
export OLLAMA_HOST="http://my-ollama-server:11434"
```

## Analysis Defaults

These settings control analysis behavior and can be adjusted in the web UI or passed via the API:

| Setting | Default | Description |
|---------|---------|-------------|
| Batch size | 100 | Conversations processed per API batch |
| Min cluster size | 5 | Privacy threshold (clusters smaller than this are hidden) |
| Clustering algorithm | k-means | Algorithm for grouping conversations |
| Max clusters | 20 | Maximum number of conversation clusters |
| Facets | sentiment, topics, entities | Default facets to extract |

## Docker Configuration

When running via Docker, pass environment variables with `-e`:

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY="sk-..." \
  -e DATABASE_URL="sqlite:///app/briefx.db" \
  -e BRIEFX_PORT="8080" \
  briefxai:latest
```

Or use an env file:

```bash
docker run -p 8080:8080 --env-file .env briefxai:latest
```

## Production Configuration Example

For a production deployment, set strict logging and use PostgreSQL:

```bash
# API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Server
BRIEFX_HOST=0.0.0.0
BRIEFX_PORT=8080
BRIEFX_LOG_LEVEL=WARNING

# Database
DATABASE_URL=postgresql://briefx:password@db-host/briefx
```

See [docs/deployment.md](deployment.md) for cloud-specific deployment guides.

## Security Best Practices

1. **Never hardcode API keys** — always use environment variables or a secrets manager
2. **Use `.env` for local development** only — never commit it
3. **In production**, use your cloud provider's secret management (AWS Secrets Manager, GCP Secret Manager, etc.)
4. **Restrict `BRIEFX_HOST`** to `127.0.0.1` unless you need external access
5. **Use PostgreSQL** for multi-instance or persistent production deployments

## Troubleshooting

### "No provider available"

At least one of the following must be set:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- A running Ollama instance at `OLLAMA_HOST`

### "Database connection failed"

For SQLite: ensure the directory is writable. The default path is `python/briefx.db`.

For PostgreSQL: verify the `DATABASE_URL` format and that the database exists:
```bash
createdb briefx
```

### "Port already in use"

Change the port:
```bash
BRIEFX_PORT=3000 python app.py
```
