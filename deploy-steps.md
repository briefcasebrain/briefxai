# Deploy BriefX to Google Cloud Run - Step by Step

## Prerequisites Installation

### 1. Install Google Cloud CLI

```bash
# For macOS (using Homebrew)
brew install --cask google-cloud-sdk

# Or download directly from:
# https://cloud.google.com/sdk/docs/install-sdk#mac
```

### 2. Initialize gcloud and login

```bash
# Login to your Google account
gcloud auth login

# Create a new project for BriefX (free tier)
gcloud projects create briefx-free-tier --name="BriefX Free"

# Set as active project
gcloud config set project briefx-free-tier

# Set default region
gcloud config set run/region us-central1
```

## Deployment Steps

### Step 1: Enable Required APIs

```bash
# Enable the necessary APIs (free)
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com
```

### Step 2: Build Docker Image Locally

```bash
# Navigate to project directory
cd /Users/aansh/Documents/xai-tool/briefxai

# Build the Docker image
docker build -t gcr.io/briefx-free-tier/briefx:latest .
```

### Step 3: Configure Docker Authentication

```bash
# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker
```

### Step 4: Push Image to Google Container Registry

```bash
# Push the image (uses free 0.5GB storage)
docker push gcr.io/briefx-free-tier/briefx:latest
```

### Step 5: Deploy to Cloud Run (Free Tier)

```bash
# Deploy with free tier limits
gcloud run deploy briefx \
  --image gcr.io/briefx-free-tier/briefx:latest \
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
  --set-env-vars "RUST_LOG=warn,OPENAI_API_KEY=${OPENAI_API_KEY}"
```

### Step 6: Get Your Application URL

```bash
# Get the deployed URL
gcloud run services describe briefx \
  --platform managed \
  --region us-central1 \
  --format "value(status.url)"
```

## Alternative: One-Command Deploy (From Source)

If you want to deploy directly from source without building locally:

```bash
# Deploy directly from source code
gcloud run deploy briefx \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 1 \
  --min-instances 0 \
  --set-env-vars "RUST_LOG=warn"
```

## Set Up Budget Alert (Important!)

```bash
# Create a $1 budget alert to ensure free tier
gcloud billing budgets create \
  --billing-account=$(gcloud billing accounts list --format="value(name)" | head -1) \
  --display-name="Free Tier Alert" \
  --budget-amount=1USD \
  --threshold-rule=percent=50
```

## Monitor Your Deployment

```bash
# View logs
gcloud run services logs read briefx --limit 50

# Check metrics
gcloud run services describe briefx --region us-central1

# View current usage (ensure it's free)
gcloud run services list --region us-central1
```

## Quick Commands Reference

```bash
# Update environment variables
gcloud run services update briefx \
  --update-env-vars KEY=value \
  --region us-central1

# Redeploy after code changes
gcloud run deploy briefx \
  --source . \
  --region us-central1

# Delete service (stop all charges)
gcloud run services delete briefx --region us-central1
```

## Troubleshooting

If deployment fails:

1. **Check Docker is running**: `docker ps`
2. **Check project exists**: `gcloud projects list`
3. **Check APIs enabled**: `gcloud services list --enabled`
4. **View build logs**: `gcloud builds log --stream`

## Expected Result

Your app will be available at:
```
https://briefx-[random-string]-uc.a.run.app
```

With free tier, you get:
- 2 million requests/month
- 360,000 GB-seconds/month
- 180,000 vCPU-seconds/month
- Scales to zero when not in use (no charges)