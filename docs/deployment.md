# BriefX Deployment Architecture

Cost-optimized deployment strategies for Google Cloud, AWS, and Vercel.

## Architecture Overview

BriefX is a Flask application with three workload types that scale independently:

| Layer | What it does | CPU/Memory profile |
|-------|-------------|-------------------|
| **Static UI** | SPA served from `briefxai_ui_data/` | Zero compute (CDN) |
| **API server** | Flask + Gunicorn serving `/api/*` | Low idle, bursty on requests |
| **Analysis workers** | Embedding generation, clustering, LLM calls | CPU+RAM heavy, intermittent |

The cheapest architecture in every cloud separates these three layers so you never pay for idle GPU/CPU while serving static files.

---

## Google Cloud (Recommended for existing GCP users)

**Estimated monthly cost: $5 -- $25 for light usage**

### Tier 1: Cheapest possible

```
Static UI  -->  Cloud Storage bucket (free tier: 5 GB)
API + Workers  -->  Cloud Run (free tier: 2M requests/mo, 360k vCPU-seconds)
Database  -->  SQLite on Cloud Run ephemeral disk (zero cost)
```

#### Deploy

```bash
# 1. Build and push container
gcloud builds submit --tag gcr.io/$PROJECT_ID/briefx:latest .

# 2. Deploy to Cloud Run
gcloud run deploy briefx \
  --image gcr.io/$PROJECT_ID/briefx:latest \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "DATABASE_URL=sqlite:///app/briefx.db" \
  --allow-unauthenticated
```

Key cost levers:
- `--min-instances 0` -- scale to zero when idle (no charge)
- `--cpu 1` and `--memory 1Gi` -- sufficient for API; clustering happens in bursts
- `--max-instances 3` -- cap spend during traffic spikes
- `--timeout 300` -- long enough for Clio analysis pipelines

#### When to add Cloud SQL

Only add a managed database when you need data to survive across deploys or share state between instances. Until then, SQLite on the container's ephemeral disk is free and sufficient for single-user or demo deployments.

```bash
# Cloud SQL Postgres (smallest instance): ~$7/mo
gcloud sql instances create briefx-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Then set DATABASE_URL on Cloud Run
gcloud run services update briefx \
  --set-env-vars "DATABASE_URL=postgresql://user:pass@/briefx?host=/cloudsql/PROJECT:REGION:briefx-db"
```

### Tier 2: Production-ready

```
Cloud CDN + Load Balancer  -->  Static UI from Cloud Storage
Cloud Run (API)            -->  min=1 for low latency
Cloud SQL (Postgres)       -->  db-f1-micro
Secret Manager             -->  API keys for LLM providers
```

Adds ~$15/mo: $7 for Cloud SQL, $7 for a warm Cloud Run instance, $1 for load balancer.

---

## AWS

**Estimated monthly cost: $0 -- $20 for light usage**

### Tier 1: Cheapest possible

```
Static UI  -->  S3 + CloudFront (free tier: 1TB/mo transfer for 12 months)
API        -->  Lambda + API Gateway (free tier: 1M requests/mo)
Database   -->  DynamoDB on-demand or SQLite in /tmp
```

#### Using Lambda with the Flask app

```bash
# Install adapter
pip install mangum

# lambda_handler.py
from mangum import Mangum
from app import app
handler = Mangum(app)
```

Package with a container image to stay under Lambda's 10 GB limit (torch + sklearn are large):

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY python/ ${LAMBDA_TASK_ROOT}/
COPY briefxai_ui_data/ ${LAMBDA_TASK_ROOT}/briefxai_ui_data/

CMD ["lambda_handler.handler"]
```

```bash
# Build and push to ECR
aws ecr create-repository --repository-name briefx
docker build -t briefx-lambda -f Dockerfile.lambda .
docker tag briefx-lambda:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/briefx:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/briefx:latest

# Create function
aws lambda create-function \
  --function-name briefx \
  --package-type Image \
  --code ImageUri=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/briefx:latest \
  --role arn:aws:iam::$ACCOUNT:role/briefx-lambda-role \
  --memory-size 1024 \
  --timeout 300 \
  --ephemeral-storage '{"Size": 1024}'
```

Key cost levers:
- Lambda scales to zero automatically
- 1M free requests/month covers most development and light production
- CloudFront free tier covers 1 TB transfer/month for 12 months

#### Limitations

- Lambda has a 15-minute max timeout -- Clio analysis on very large datasets may need to be split or moved to ECS
- Cold starts add 3-8 seconds on first request (mitigate with provisioned concurrency at ~$5/mo)

### Tier 2: Production-ready

```
CloudFront          -->  S3 static assets
ALB + ECS Fargate   -->  API container (spot pricing)
RDS Postgres        -->  db.t4g.micro (free tier eligible)
Secrets Manager     -->  LLM API keys
```

```bash
# ECS Fargate spot pricing: ~60-70% cheaper than on-demand
aws ecs create-service \
  --service-name briefx-api \
  --task-definition briefx:1 \
  --capacity-provider-strategy \
    capacityProvider=FARGATE_SPOT,weight=3 \
    capacityProvider=FARGATE,weight=1 \
  --desired-count 1
```

Estimated cost: $5-15/mo (Fargate Spot) + $0 (RDS free tier first year) + $0.50 (Secrets Manager).

---

## Vercel

**Estimated monthly cost: $0 (free tier) -- $20 (Pro)**

Best for: getting a public demo running in under 5 minutes.

### Architecture

```
Static UI        -->  Vercel Edge Network (automatic CDN)
API              -->  Vercel Serverless Functions (Python runtime)
Database         -->  Vercel Postgres (free tier: 256 MB) or external
```

Vercel runs Python serverless functions natively but with a 50 MB bundle limit on the free tier. BriefX's dependencies (torch, sklearn, numpy) exceed this, so the approach differs from GCP/AWS:

### Option A: Vercel for static UI + external API

This is the cheapest hybrid. Vercel serves the SPA for free; the API runs on Cloud Run or Lambda.

```
vercel.json
```

```json
{
  "buildCommand": "",
  "outputDirectory": "briefxai_ui_data",
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://briefx-api-xxxxx.a.run.app/api/:path*" }
  ]
}
```

```bash
# Deploy static UI to Vercel
cd briefxai_ui_data
vercel --prod
```

Cost: $0 for Vercel (static hosting is free), API cost from GCP/AWS as above.

### Option B: Vercel Functions (lightweight API only)

If you strip torch and use only the core analysis (sklearn, numpy), the bundle fits:

```
api/health.py       -->  /api/health
api/analyze.py      -->  /api/analyze
api/upload.py       -->  /api/upload
```

```python
# api/health.py
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy"}).encode())
```

Limitations:
- 10-second execution limit on free tier (60s on Pro)
- 50 MB bundle limit (250 MB on Pro)
- No persistent filesystem -- must use external DB

---

## Cost Comparison Summary

| Scenario | Google Cloud | AWS | Vercel |
|----------|-------------|-----|--------|
| **Demo / side project** (< 1K req/day) | $0 | $0 | $0 |
| **Light production** (< 10K req/day) | $5-10/mo | $5-10/mo | $0-20/mo |
| **Production with DB** (< 100K req/day) | $15-25/mo | $10-20/mo | $20/mo + external API |
| **Heavy analysis workloads** | $25-50/mo | $20-40/mo | Not recommended |

### When to use which

| Choose | When |
|--------|------|
| **Google Cloud (Cloud Run)** | You already have GCP, want simplest Docker deployment, or need long-running analysis (>15 min) |
| **AWS (Lambda)** | You want true zero-cost at idle, have AWS credits, or need the broadest service ecosystem |
| **Vercel** | You want the fastest path to a public demo, or only need the static UI with an external API backend |

---

## Shared Cost Optimizations

These apply regardless of cloud provider:

### 1. Separate static assets from compute

The SPA in `briefxai_ui_data/` is ~500 KB. Serve it from a CDN (Cloud Storage, S3+CloudFront, Vercel Edge) instead of through Gunicorn. This eliminates compute cost for the majority of HTTP requests.

### 2. Scale to zero

All three platforms support scaling to zero. Configure `min-instances=0` (Cloud Run), use Lambda's default behavior, or rely on Vercel's serverless model. You pay nothing when nobody is using the app.

### 3. Use SQLite until you can't

BriefX supports both SQLite (`aiosqlite`) and PostgreSQL (`asyncpg`). SQLite has zero operational cost and is sufficient for single-instance deployments. Switch to managed Postgres only when you need:
- Multiple API instances sharing state
- Data durability beyond container lifecycle
- Concurrent write-heavy workloads

### 4. Cache LLM responses

The `response_cache` table in the schema already supports this. Enabling caching for embeddings and LLM facet extraction avoids repeat API calls, which are the dominant cost driver. A single Clio analysis can make dozens of LLM calls -- caching identical conversations saves 80%+ on provider costs.

### 5. Use briefcase_ai.CostCalculator to choose models

```python
import briefcase_ai
cc = briefcase_ai.CostCalculator()

# Find cheapest model that fits your context window
cheapest = cc.get_cheapest_model(min_context_window=4096)

# Project monthly spend before committing
projection = cc.project_monthly_cost(
    model_name="gpt-4o-mini",
    daily_input_tokens=100_000,
    daily_output_tokens=20_000,
    days_per_month=30
)
```

### 6. Right-size container resources

BriefX's analysis is CPU-bound (sklearn clustering, numpy, UMAP). Memory matters more than CPU count:

| Workload | Recommended | Why |
|----------|-------------|-----|
| API only (no analysis) | 0.5 vCPU / 512 MB | Flask + Gunicorn idle footprint |
| Light analysis (< 100 conversations) | 1 vCPU / 1 GB | sklearn fits in memory |
| Heavy analysis (1K+ conversations) | 2 vCPU / 2 GB | UMAP + large embedding matrices |

---

## Environment Variables

All deployments need these configured (via Secret Manager, Secrets Manager, or Vercel env vars):

```bash
# Required for LLM-powered features (pick one or more)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Optional
DATABASE_URL=sqlite:///app/briefx.db   # or postgresql://...
BRIEFCASE_API_KEY=sk-...               # for briefcase_ai platform features
```
