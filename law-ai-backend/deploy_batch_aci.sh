#!/bin/bash
# Deploy build_embeddings_test.py to Azure Container Instances
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Azure Container Registry created
#   - Environment variables set (see below)
#
# Usage:
#   chmod +x deploy_batch_aci.sh
#   ./deploy_batch_aci.sh

set -e

# === CONFIGURATION - EDIT THESE ===
RESOURCE_GROUP="sophieai-rg"
ACR_NAME="sophieaiacr"               # Azure Container Registry name
CONTAINER_NAME="sophieai-batch"
IMAGE_NAME="sophieai-batch"
IMAGE_TAG="latest"
LOCATION="australiaeast"              # Choose region close to your data/APIs

# Storage account for persisting output (case summaries JSONL + chroma_db)
STORAGE_ACCOUNT="sophieaistorage"
SHARE_NAME="batchdata"

# === STEP 1: Build and push Docker image ===
echo "🔨 Building Docker image..."
az acr build \
  --registry $ACR_NAME \
  --image $IMAGE_NAME:$IMAGE_TAG \
  --file Dockerfile.batch \
  .

# === STEP 2: Create Azure File Share (for persistent output) ===
echo "📁 Creating file share for output..."
az storage share create \
  --name $SHARE_NAME \
  --account-name $STORAGE_ACCOUNT \
  2>/dev/null || echo "Share already exists"

STORAGE_KEY=$(az storage account keys list \
  --resource-group $RESOURCE_GROUP \
  --account-name $STORAGE_ACCOUNT \
  --query '[0].value' -o tsv)

# === STEP 3: Deploy container ===
echo "🚀 Deploying container..."
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)

az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" \
  --registry-login-server "$ACR_NAME.azurecr.io" \
  --registry-username $ACR_NAME \
  --registry-password "$ACR_PASSWORD" \
  --cpu 2 \
  --memory 4 \
  --restart-policy Never \
  --location $LOCATION \
  --environment-variables \
    ENV=prd \
    GOOGLE_API_KEY="$GOOGLE_API_KEY" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
  --azure-file-volume-account-name $STORAGE_ACCOUNT \
  --azure-file-volume-account-key "$STORAGE_KEY" \
  --azure-file-volume-share-name $SHARE_NAME \
  --azure-file-volume-mount-path /app/out_summaries

echo "✅ Container deployed. Check logs with:"
echo "   az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
echo ""
echo "📊 Check status with:"
echo "   az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query 'instanceView.state'"
