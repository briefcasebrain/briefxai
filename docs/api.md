# BriefXAI API Reference

BriefXAI provides REST APIs and WebSocket connections for programmatic access to conversation analysis capabilities.

## Base URL

```
http://localhost:8080/api
```

## New Features ✨

- **Smart Auto-Detection APIs**: Automatically recommend optimal models
- **Cost Estimation**: Get cost estimates before running analysis
- **File Upload Support**: Upload JSON, CSV, PDF, TXT, DOCX files with OCR
- **Real-time Progress**: WebSocket updates for live analysis progress
- **Google Gemini Integration**: Full support for Gemini models

## Authentication

BriefXAI currently supports multiple authentication methods:

### Option 1: API Key (via Environment)
Configure in your `.env` file - no headers needed:
```bash
OPENAI_API_KEY=sk-your-key
GOOGLE_API_KEY=your-gemini-key
```

### Option 2: Request Headers (Optional)
```http
Authorization: Bearer YOUR_API_KEY
# Or provider-specific:
X-OpenAI-Key: sk-your-openai-key
X-Google-Key: your-gemini-key  
```

## REST API Endpoints

### Core Analysis Endpoints

#### Direct Analysis (Recommended)

Analyze conversations directly without session management.

```http
POST /api/analyze
```

**Request Body:**
```json
{
  "data": [
    {
      "messages": [
        {
          "role": "user", 
          "content": "I can't log into my account"
        },
        {
          "role": "assistant",
          "content": "I'll help you reset your password"
        }
      ],
      "metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "customer_id": "cust_123"
      }
    }
  ],
  "config": {
    "llm_provider": "gemini",
    "llm_model": "gemini-1.5-flash",
    "embedding_provider": "google", 
    "embedding_model": "text-embedding-004",
    "dedup": true,
    "batch_size": 10,
    "api_key": "optional-override-key"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "analysis_abc123", 
    "results": {
      "conversations": [...],
      "facets": [...],
      "hierarchy": [...],
      "umap": [...],
      "stats": {
        "total_conversations": 1,
        "total_clusters": 3,
        "embedding_dimension": 768
      }
    }
  }
}
```

#### File Upload

Upload files for analysis (supports JSON, CSV, PDF, TXT, DOCX).

```http  
POST /api/upload
Content-Type: multipart/form-data
```

**Form Data:**
```
file0: [binary file data]
file1: [binary file data]  
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "upload_xyz789",
    "files": [
      {
        "name": "conversations.json",
        "size": 15420,
        "mime_type": "application/json",
        "conversations": 25
      }
    ],
    "conversations": [...],
    "total_conversations": 25,
    "warnings": []
  }
}
```

### Auto-Detection & Cost APIs

#### Model Recommendation

Get automatic model recommendations based on content.

```http
POST /api/recommend-models
```

**Request Body:**
```json
{
  "sample_data": [
    {
      "messages": [
        {"role": "user", "content": "Sample conversation..."}
      ]
    }
  ],
  "provider_preference": "gemini"  // optional
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "llm_recommendation": {
      "provider": "gemini",
      "model": "gemini-1.5-flash",
      "reason": "General content - optimal speed and cost"
    },
    "embedding_recommendation": {
      "provider": "google", 
      "model": "text-embedding-004",
      "reason": "Latest general-purpose model"
    },
    "cost_estimate": {
      "estimated_cost": 0.12,
      "currency": "USD",
      "breakdown": {
        "llm_cost": 0.08,
        "embedding_cost": 0.04
      }
    }
  }
}
```

#### Cost Estimation

Get cost estimates without running analysis.

```http
POST /api/estimate-cost
```

**Request Body:**  
```json
{
  "conversation_count": 100,
  "config": {
    "llm_model": "gemini-1.5-flash",
    "embedding_model": "text-embedding-004"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "estimated_cost": 0.50,
    "currency": "USD", 
    "confidence": "medium",
    "breakdown": {
      "llm_tokens": 50000,
      "llm_cost": 0.38,
      "embedding_tokens": 25000, 
      "embedding_cost": 0.12
    },
    "recommendations": [
      "Consider using gemini-1.5-flash-8b for 60% cost reduction",
      "Enable deduplication to reduce processing"
    ]
  }
}
```

### Status & Health Endpoints

#### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "features": {
    "auto_detection": true,
    "ocr_support": true,
    "cost_estimation": true
  }
}
```

#### Provider Status

Check which AI providers are available and configured.

```http
GET /api/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "providers": {
      "openai": {
        "available": true,
        "configured": true,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
      },
      "gemini": {
        "available": true,
        "configured": true, 
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
      },
      "ollama": {
        "available": false,
        "configured": false,
        "error": "Ollama not running"
      }
    },
    "recommended_provider": "gemini"
  }
}
```

#### Ollama Status

Check Ollama installation and available models.

```http
GET /api/ollama-status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "running": true,
    "installed": true,
    "models": ["llama3.2:3b", "llama3.1:8b", "codellama:13b"]
  }
}
```

### Server Management APIs

#### Start Local Servers

Start Ollama or vLLM servers automatically.

```http
POST /api/start-server
```

**Request Body:**
```json
{
  "serverType": "ollama"  // "ollama" or "vllm"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Ollama server started successfully"
  }
}
```

#### Pull Models

Download models for Ollama.

```http
POST /api/pull-model
```

**Request Body:**
```json
{
  "model": "llama3.2:3b"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Model llama3.2:3b pulled successfully"
  }
}
```

### Progress & Session Management

#### Session Status

Get analysis session status and progress.

```http
GET /api/session/{session_id}/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "analysis_abc123",
    "status": "running",
    "progress": {
      "step": "embeddings",
      "progress": 65.0,
      "message": "Generating embeddings (batch 3/5)",
      "estimated_completion": "2024-01-15T11:45:00Z"
    }
  }
}
```

#### Progress Updates (Deprecated - Use WebSocket)

```http
GET /api/progress/{session_id}
```

### Example Data

#### Generate Example Conversations

Get sample data for testing.

```http
GET /api/example?count=50
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "messages": [
        {
          "role": "user",
          "content": "I need help with shipping"
        },
        {
          "role": "assistant", 
          "content": "I'll help you track your order"
        }
      ],
      "metadata": {
        "generated": true,
        "category": "customer_support"
      }
    }
  ]
}
```

## WebSocket API

### Real-time Progress Updates

Connect to receive live progress updates during analysis.

**WebSocket URL:**
```
ws://localhost:8080/ws/progress
```

#### Connection Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/progress');

ws.onopen = function() {
    console.log('Connected to progress stream');
    // Send ping to keep connection alive
    ws.send('ping');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type || message.step) {
        case 'Started':
            console.log(`Analysis ${message.session_id} started`);
            break;
            
        case 'Update':
        default:
            // Progress update
            console.log(`${message.step}: ${message.progress}% - ${message.message}`);
            if (message.details) {
                console.log(`Details: ${message.details}`);
            }
            break;
            
        case 'Completed':
            console.log(`Analysis ${message.session_id} completed: ${message.result}`);
            break;
            
        case 'Error':
            console.error(`Analysis ${message.session_id} failed: ${message.error}`);
            break;
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function() {
    console.log('WebSocket connection closed');
};
```

#### Message Types

**Connected Message:**
```json
{
  "step": "connected",
  "progress": 0.0,
  "message": "Connected to progress stream",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Progress Update:**
```json
{
  "step": "embeddings",
  "progress": 65.5,
  "message": "Generating embeddings",
  "details": "Processing batch 3/5 (10 conversations)",
  "timestamp": "2024-01-15T10:32:15Z"
}
```

**Analysis Steps:**
- `validation` (0-15%): Data validation and preprocessing
- `dedup` (10-15%): Deduplication if enabled  
- `facets` (15-35%): LLM facet extraction
- `embeddings` (35-55%): Embedding generation
- `clustering` (55-70%): K-means clustering
- `hierarchy` (70-80%): Building cluster hierarchy
- `umap` (80-95%): UMAP visualization
- `complete` (100%): Analysis finished

**Completion Message:**
```json
{
  "type": "Completed",
  "session_id": "analysis_abc123",
  "result": "Analysis completed successfully"
}
```

**Error Message:**
```json
{
  "type": "Error", 
  "session_id": "analysis_abc123",
  "error": "API key invalid for selected provider"
}
```

## Code Examples

### Python Client Example

```python
import requests
import json
import websocket
import threading

class BriefXAIClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.ws_url = f"ws://localhost:8080/ws/progress"
    
    def analyze_conversations(self, conversations, config=None):
        """Analyze conversations with optional progress tracking."""
        
        # Default config
        if config is None:
            config = {
                "llm_provider": "gemini",
                "llm_model": "gemini-1.5-flash",
                "embedding_provider": "google", 
                "embedding_model": "text-embedding-004",
                "dedup": True,
                "batch_size": 10
            }
        
        # Start WebSocket for progress  
        def on_message(ws, message):
            data = json.loads(message)
            if data.get('step') in ['validation', 'facets', 'embeddings', 'clustering', 'complete']:
                print(f"[{data['progress']:.1f}%] {data['step']}: {data['message']}")
                if data.get('details'):
                    print(f"  └─ {data['details']}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            print("Progress stream closed")
        
        # Connect to progress stream
        ws = websocket.WebSocketApp(self.ws_url,
                                  on_message=on_message,
                                  on_error=on_error, 
                                  on_close=on_close)
        
        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Start analysis
        response = requests.post(
            f"{self.api_url}/analyze",
            json={
                "data": conversations,
                "config": config
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            result = response.json()
            if result['success']:
                print(f"\\n✅ Analysis complete! Session: {result['data']['session_id']}")
                return result['data']['results']
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return None
        else:
            print(f"❌ Request failed: {response.status_code}")
            return None
    
    def get_cost_estimate(self, conversation_count, config):
        """Get cost estimate before running analysis."""
        response = requests.post(
            f"{self.api_url}/estimate-cost",
            json={
                "conversation_count": conversation_count,
                "config": config
            }
        )
        
        if response.ok:
            return response.json()['data']
        return None
    
    def get_model_recommendations(self, sample_conversations):
        """Get auto-detected model recommendations."""
        response = requests.post(
            f"{self.api_url}/recommend-models", 
            json={"sample_data": sample_conversations}
        )
        
        if response.ok:
            return response.json()['data']
        return None

# Usage example
client = BriefXAIClient()

# Sample conversations
conversations = [
    {
        "messages": [
            {"role": "user", "content": "I can't access my account"},
            {"role": "assistant", "content": "I'll help you reset your password"}
        ]
    }
]

# Get recommendations first
recommendations = client.get_model_recommendations(conversations[:1])
if recommendations:
    print(f"Recommended LLM: {recommendations['llm_recommendation']['model']}")
    print(f"Estimated cost: ${recommendations['cost_estimate']['estimated_cost']}")

# Run analysis  
results = client.analyze_conversations(conversations)
if results:
    print(f"Found {len(results['hierarchy'])} clusters")
    print(f"Generated {len(results['umap'])} visualization points")
```

### JavaScript/Node.js Client Example

```javascript
const fetch = require('node-fetch');
const WebSocket = require('ws');

class BriefXAIClient {
    constructor(baseUrl = 'http://localhost:8080') {
        this.baseUrl = baseUrl;
        this.apiUrl = `${baseUrl}/api`;
        this.wsUrl = `ws://localhost:8080/ws/progress`;
    }
    
    async analyzeConversations(conversations, config = null) {
        // Default config
        config = config || {
            llm_provider: 'gemini',
            llm_model: 'gemini-1.5-flash', 
            embedding_provider: 'google',
            embedding_model: 'text-embedding-004',
            dedup: true,
            batch_size: 10
        };
        
        // Setup progress tracking
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(this.wsUrl);
            
            ws.on('message', (data) => {
                const message = JSON.parse(data);
                if (message.step && ['validation', 'facets', 'embeddings', 'clustering', 'complete'].includes(message.step)) {
                    console.log(`[${message.progress.toFixed(1)}%] ${message.step}: ${message.message}`);
                    if (message.details) {
                        console.log(`  └─ ${message.details}`);
                    }
                }
            });
            
            // Start analysis
            fetch(`${this.apiUrl}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: conversations,
                    config: config
                })
            })
            .then(response => response.json())
            .then(result => {
                ws.close();
                if (result.success) {
                    console.log(`\\n✅ Analysis complete! Session: ${result.data.session_id}`);
                    resolve(result.data.results);
                } else {
                    reject(new Error(result.error || 'Analysis failed'));
                }
            })
            .catch(error => {
                ws.close();
                reject(error);
            });
        });
    }
    
    async getCostEstimate(conversationCount, config) {
        const response = await fetch(`${this.apiUrl}/estimate-cost`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_count: conversationCount,
                config: config  
            })
        });
        
        const result = await response.json();
        return result.success ? result.data : null;
    }
}

// Usage
const client = new BriefXAIClient();

const conversations = [
    {
        messages: [
            { role: "user", content: "I need help with my order" },
            { role: "assistant", content: "I'll help you track your shipment" }
        ]
    }
];

// Run analysis with progress tracking
client.analyzeConversations(conversations)
    .then(results => {
        console.log(`Analysis complete: ${results.stats.total_conversations} conversations processed`);
    })
    .catch(error => {
        console.error('Analysis failed:', error.message);
    });
```

## Error Handling

### Common Error Responses

**Authentication Error:**
```json
{
  "success": false,
  "error": "API key required for selected provider",
  "code": "AUTH_REQUIRED"
}
```

**Rate Limit Error:**  
```json
{
  "success": false,
  "error": "Rate limit exceeded. Try again in 60 seconds.",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60
}
```

**Validation Error:**
```json
{
  "success": false,
  "error": "Invalid conversation format", 
  "code": "VALIDATION_ERROR",
  "details": {
    "field": "messages",
    "message": "Messages array cannot be empty"
  }
}
```

**Cost Limit Error:**
```json
{
  "success": false,
  "error": "Estimated cost ($5.00) exceeds limit ($2.00)",
  "code": "COST_LIMIT_EXCEEDED", 
  "estimated_cost": 5.00,
  "cost_limit": 2.00
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request format
- `401 Unauthorized` - Missing or invalid API key  
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Provider unavailable

### Sessions (Legacy)

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