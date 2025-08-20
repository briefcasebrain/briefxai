# API Reference

BriefXAI provides both REST and WebSocket APIs for programmatic access to analysis capabilities.

## Base URL

```
http://localhost:8080/api
```

## Authentication

Currently, BriefXAI uses API key authentication. Include your API key in the request header:

```http
Authorization: Bearer YOUR_API_KEY
```

## REST API Endpoints

### Sessions

#### Create Session

Create a new analysis session.

```http
POST /api/sessions
```

**Request Body:**
```json
{
  "name": "Customer Support Analysis",
  "description": "Q1 2024 support ticket analysis",
  "config": {
    "template": "customer_support",
    "batch_size": 100,
    "provider": "openai"
  }
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "state": "created",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Session

Retrieve session details.

```http
GET /api/sessions/{session_id}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "name": "Customer Support Analysis",
  "state": "running",
  "progress": {
    "total": 1000,
    "completed": 450,
    "percentage": 45
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:15:00Z"
}
```

#### List Sessions

Get all sessions with optional filtering.

```http
GET /api/sessions?state=running&limit=10&offset=0
```

**Query Parameters:**
- `state` (optional): Filter by state (created, running, paused, completed, failed)
- `limit` (optional): Number of results (default: 20)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "sess_abc123",
      "name": "Customer Support Analysis",
      "state": "running",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

#### Update Session State

Pause, resume, or cancel a session.

```http
PUT /api/sessions/{session_id}/state
```

**Request Body:**
```json
{
  "action": "pause"  // pause, resume, or cancel
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "state": "paused",
  "message": "Session paused successfully"
}
```

### Analysis

#### Start Analysis

Begin analyzing conversations for a session.

```http
POST /api/sessions/{session_id}/analyze
```

**Request Body:**
```json
{
  "conversations": [
    {
      "id": "conv_001",
      "messages": [
        {
          "role": "user",
          "content": "I need help with my order"
        },
        {
          "role": "assistant",
          "content": "I'd be happy to help you with your order."
        }
      ],
      "metadata": {
        "timestamp": "2024-01-15T10:00:00Z",
        "customer_id": "cust_123"
      }
    }
  ],
  "options": {
    "extract_facets": true,
    "generate_embeddings": true,
    "perform_clustering": true
  }
}
```

**Response:**
```json
{
  "message": "Analysis started",
  "batch_id": "batch_xyz789",
  "estimated_time_seconds": 300
}
```

#### Get Analysis Results

Retrieve analysis results for a session.

```http
GET /api/sessions/{session_id}/results
```

**Query Parameters:**
- `format` (optional): Response format (json, csv)
- `include_embeddings` (optional): Include embedding vectors (default: false)

**Response:**
```json
{
  "session_id": "sess_abc123",
  "results": {
    "conversations_analyzed": 1000,
    "facets": {
      "sentiments": {
        "positive": 450,
        "neutral": 300,
        "negative": 250
      },
      "topics": [
        {
          "name": "billing",
          "count": 320
        },
        {
          "name": "shipping",
          "count": 280
        }
      ]
    },
    "clusters": [
      {
        "id": 0,
        "size": 150,
        "centroid": [0.1, 0.2, ...],
        "description": "Order status inquiries"
      }
    ]
  }
}
```

### Preprocessing

#### Validate Data

Validate conversation data before analysis.

```http
POST /api/preprocessing/validate
```

**Request Body:**
```json
{
  "conversations": [...],
  "options": {
    "check_duplicates": true,
    "detect_pii": true,
    "validate_format": true
  }
}
```

**Response:**
```json
{
  "valid": true,
  "issues": [],
  "statistics": {
    "total_conversations": 100,
    "duplicates_found": 2,
    "pii_detected": 5
  }
}
```

### Providers

#### List Providers

Get available LLM providers and their status.

```http
GET /api/providers
```

**Response:**
```json
{
  "providers": [
    {
      "id": "openai",
      "name": "OpenAI",
      "status": "healthy",
      "models": ["gpt-4", "gpt-3.5-turbo"],
      "metrics": {
        "requests_today": 1500,
        "average_latency_ms": 250
      }
    },
    {
      "id": "ollama",
      "name": "Ollama",
      "status": "healthy",
      "models": ["llama2", "mistral"]
    }
  ]
}
```

#### Test Provider

Test connectivity to a specific provider.

```http
POST /api/providers/{provider_id}/test
```

**Response:**
```json
{
  "provider_id": "openai",
  "status": "success",
  "latency_ms": 150,
  "message": "Provider is responsive"
}
```

### Export

#### Export Results

Export analysis results in various formats.

```http
POST /api/sessions/{session_id}/export
```

**Request Body:**
```json
{
  "format": "csv",  // json, csv, html
  "include": {
    "facets": true,
    "clusters": true,
    "raw_conversations": false
  }
}
```

**Response:**
```json
{
  "export_id": "exp_123",
  "download_url": "/api/exports/exp_123/download",
  "expires_at": "2024-01-16T10:30:00Z"
}
```

## WebSocket API

Connect to receive real-time updates during analysis.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### Subscribe to Session

After connection, subscribe to a session:

```json
{
  "type": "subscribe",
  "session_id": "sess_abc123"
}
```

### Message Types

#### Progress Update
```json
{
  "type": "progress",
  "session_id": "sess_abc123",
  "progress": {
    "current": 500,
    "total": 1000,
    "percentage": 50
  }
}
```

#### Partial Results
```json
{
  "type": "partial_results",
  "session_id": "sess_abc123",
  "results": {
    "conversations_processed": 100,
    "insights": [
      {
        "type": "pattern",
        "description": "High volume of billing inquiries detected"
      }
    ]
  }
}
```

#### Error
```json
{
  "type": "error",
  "session_id": "sess_abc123",
  "error": {
    "code": "PROVIDER_ERROR",
    "message": "OpenAI API rate limit exceeded"
  }
}
```

#### Completion
```json
{
  "type": "complete",
  "session_id": "sess_abc123",
  "summary": {
    "total_processed": 1000,
    "duration_seconds": 300,
    "status": "success"
  }
}
```

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid conversation format",
    "details": {
      "field": "messages",
      "reason": "Messages array cannot be empty"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `NOT_FOUND` | 404 | Resource not found |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `RATE_LIMITED` | 429 | Too many requests |
| `PROVIDER_ERROR` | 502 | LLM provider error |
| `INTERNAL_ERROR` | 500 | Server error |

## Rate Limiting

Default rate limits:
- 100 requests per minute per API key
- 10 concurrent sessions per account
- 10,000 conversations per analysis batch

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705318800
```

## SDK Examples

### Python

```python
import briefxai

client = briefxai.Client(api_key="YOUR_API_KEY")

# Create session
session = client.create_session(
    name="Q1 Analysis",
    template="customer_support"
)

# Start analysis
result = client.analyze(
    session_id=session.id,
    conversations=conversations,
    wait=True  # Wait for completion
)

# Export results
export = client.export(
    session_id=session.id,
    format="csv"
)
```

### JavaScript/TypeScript

```typescript
import { BriefXAI } from 'briefxai-sdk';

const client = new BriefXAI({ apiKey: 'YOUR_API_KEY' });

// Create and run analysis
const session = await client.sessions.create({
  name: 'Q1 Analysis',
  template: 'customer_support'
});

const result = await client.analyze({
  sessionId: session.id,
  conversations: conversations
});

// Subscribe to real-time updates
client.subscribe(session.id, (update) => {
  console.log(`Progress: ${update.progress.percentage}%`);
});
```

### Rust

```rust
use briefxai_client::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(Config {
        api_key: "YOUR_API_KEY".to_string(),
        base_url: "http://localhost:8080".to_string(),
    });

    // Create session
    let session = client.create_session(
        "Q1 Analysis",
        "customer_support"
    ).await?;

    // Analyze conversations
    let result = client.analyze(
        &session.id,
        conversations
    ).await?;

    Ok(())
}
```

## Pagination

For endpoints returning lists, use pagination parameters:

```http
GET /api/sessions?limit=20&offset=40
```

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "total": 100,
    "limit": 20,
    "offset": 40,
    "has_more": true
  }
}
```

## Versioning

The API version is included in the URL path:

```
http://localhost:8080/api/v1/sessions
```

Version compatibility:
- Breaking changes increment major version
- New features increment minor version
- Bug fixes increment patch version

## Health Check

Monitor service health:

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": "ok",
    "providers": "ok",
    "memory": "ok"
  }
}
```