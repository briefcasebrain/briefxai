# Deploy BriefxAI to Google Cloud Run (Free Tier)

This guide helps you deploy BriefxAI to Google Cloud Run while staying within the **free tier limits**.

## Google Cloud Free Tier Limits (Monthly)

- **Cloud Run**: 2 million requests, 360,000 GB-seconds memory, 180,000 vCPU-seconds
- **Cloud Build**: 120 build-minutes per day
- **Container Registry**: 0.5 GB storage
- **Secret Manager**: 6 active secret versions, 10,000 access operations

## Prerequisites

1. **Create Google Cloud Account** (includes $300 free credits for 90 days)
   ```bash
   # Sign up at https://cloud.google.com/free
   ```

2. **Install Google Cloud CLI**
   ```bash
   # Download from https://cloud.google.com/sdk/docs/install
   # After installation, initialize:
   gcloud init
   ```

3. **Create a New Project**
   ```bash
   # Create project (choose a unique project ID)
   gcloud projects create briefxai-free-tier --name="BriefxAI Free"
   
   # Set as default project
   gcloud config set project briefxai-free-tier
   ```

## Step 1: Enable Required APIs (Free)

```bash
# Enable only essential APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com
```

## Step 2: Store API Keys (Free Tier Method)

Since Secret Manager has limits, we'll use environment variables directly:

```bash
# Create a .env.production file locally (DO NOT COMMIT)
cat > .env.production << EOF
OPENAI_API_KEY=sk-your-actual-key-here
GEMINI_API_KEY=AIza-your-actual-key-here
EOF
```

## Step 3: Build and Deploy (Manual - Saves Build Minutes)

### Option A: Build Locally and Push (Recommended for Free Tier)

```bash
# 1. Build the Docker image locally
docker build -t gcr.io/briefxai-free-tier/briefxai:latest .

# 2. Authenticate Docker with GCR
gcloud auth configure-docker

# 3. Push to Google Container Registry (uses free 0.5GB)
docker push gcr.io/briefxai-free-tier/briefxai:latest

# 4. Deploy to Cloud Run with free tier limits
gcloud run deploy briefxai \
  --image gcr.io/briefxai-free-tier/briefxai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 1 \
  --min-instances 0 \
  --concurrency 80 \
  --set-env-vars "RUST_LOG=info,OPENAI_API_KEY=$OPENAI_API_KEY"
```

### Option B: Use Cloud Build (Uses Daily Build Minutes)

```bash
# Only if you have build minutes remaining today
gcloud builds submit --config cloudbuild.yaml
```

## Step 4: Optimize for Free Tier

### Configure Auto-Scaling to Minimize Costs

```bash
# Ensure scale-to-zero is enabled
gcloud run services update briefxai \
  --min-instances=0 \
  --max-instances=1 \
  --cpu-throttling \
  --region us-central1
```

### Set Memory and CPU Limits

```bash
# Use minimum resources that still work
gcloud run services update briefxai \
  --memory=512Mi \
  --cpu=1 \
  --region us-central1
```

## Step 5: Monitor Free Tier Usage

### Check Current Usage

```bash
# View Cloud Run metrics
gcloud run services describe briefxai --region us-central1

# Check billing (should show $0.00)
gcloud billing accounts list
```

### Set Up Budget Alerts (Important!)

```bash
# Create a $1 budget alert to ensure you stay free
gcloud billing budgets create \
  --billing-account=$(gcloud billing accounts list --format="value(name)") \
  --display-name="Free Tier Alert" \
  --budget-amount=1 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

## Step 6: Access Your Application

```bash
# Get the URL
gcloud run services describe briefxai \
  --platform managed \
  --region us-central1 \
  --format "value(status.url)"
```

Your app will be available at: `https://briefxai-xxxxx-uc.a.run.app`

## Free Tier Best Practices

### 1. **Minimize Container Size**
```dockerfile
# Use alpine images when possible
# Remove development dependencies
# Clear package manager caches
```

### 2. **Scale to Zero**
- Always set `--min-instances=0`
- App sleeps when not in use (no charges)
- ~5-10 second cold start is acceptable

### 3. **Limit Features for Free Tier**
```bash
# Disable resource-intensive features
--set-env-vars "ENABLE_OCR=false,MAX_UPLOAD_SIZE=10485760"
```

### 4. **Use Client-Side Processing**
- Move computations to browser when possible
- Use WebAssembly for client-side ML
- Cache aggressively

## Free Tier Limitations & Workarounds

| Limitation | Impact | Workaround |
|------------|--------|------------|
| 512MB RAM | Limited file processing | Process smaller batches |
| 1 vCPU | Slower analysis | Add progress indicators |
| 1 instance max | No concurrent users | Add queue system |
| 300s timeout | Long tasks fail | Break into smaller tasks |
| Scale to zero | Cold starts | Add loading screen |

## Monitoring Costs

### Daily Check Commands

```bash
# Check if you're within free tier
gcloud run services list --region us-central1

# View request count (stay under 2M/month)
gcloud monitoring read \
  "run.googleapis.com/request_count" \
  --project briefxai-free-tier \
  --format "table(resource.service_name, value)"
```

## Emergency Cost Control

If you accidentally exceed free tier:

```bash
# IMMEDIATELY delete the service
gcloud run services delete briefxai --region us-central1

# Delete container images
gcloud container images delete gcr.io/briefxai-free-tier/briefxai

# Disable APIs
gcloud services disable run.googleapis.com
```

## Alternative Free Hosting Options

If Cloud Run free tier is insufficient:

1. **Railway.app**: $5 free credits/month
2. **Fly.io**: 3 shared-cpu-1x VMs free
3. **Render.com**: 750 hours free/month
4. **Heroku** (no longer free)
5. **Local deployment**: ngrok for temporary public access

## Typical Free Tier Usage

With optimization, you can handle:
- ~50-100 daily users
- ~1000 analysis requests/day
- Files up to 10MB
- 5-minute max processing time

## Next Steps

1. **Add Cloudflare** (free CDN) for static assets
2. **Use GitHub Actions** (free tier) for CI/CD
3. **Implement caching** to reduce API calls
4. **Add rate limiting** to prevent abuse

## Support & Cost Calculator

- Monitor usage: https://console.cloud.google.com/billing
- Pricing calculator: https://cloud.google.com/products/calculator
- Free tier details: https://cloud.google.com/free/docs/free-cloud-features

## Quick Reference

```bash
# Deploy command (all-in-one)
gcloud run deploy briefxai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 1 \
  --min-instances 0 \
  --set-env-vars "RUST_LOG=info"

# Check status
gcloud run services describe briefxai --region us-central1 --format "value(status.url)"

# View logs
gcloud run services logs read briefxai --region us-central1 --limit 50

# Delete everything (stop all charges)
gcloud run services delete briefxai --region us-central1 --quiet
```

**Remember**: The free tier is perfect for development, testing, and small-scale personal use. Monitor your usage regularly to avoid unexpected charges!