#!/bin/bash
# ============================================
# MinIO Initialization Script
# ============================================
# This script creates buckets and sets up MinIO for the Stripe architecture
# ============================================

echo "============================================"
echo "  Initializing MinIO"
echo "============================================"
echo ""

# Load credentials
source secret.sh

# MinIO endpoint (inside Docker network)
MINIO_ENDPOINT="http://minio:9000"
MINIO_ALIAS="stripe-minio"

echo "[1/4] Configuring MinIO client (mc)..."
docker exec stripe-minio mc alias set $MINIO_ALIAS $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY

if [ $? -eq 0 ]; then
    echo "     ✓ MinIO client configured"
else
    echo "     ✗ Failed to configure MinIO client"
    exit 1
fi

echo ""
echo "[2/4] Creating buckets..."

# Create bucket for ML models
docker exec stripe-minio mc mb --ignore-existing $MINIO_ALIAS/mlflow
echo "     ✓ Created bucket: mlflow (for ML models)"

# Create bucket for data lake
docker exec stripe-minio mc mb --ignore-existing $MINIO_ALIAS/datalake
echo "     ✓ Created bucket: datalake (for raw data)"

# Create bucket for backups
docker exec stripe-minio mc mb --ignore-existing $MINIO_ALIAS/backups
echo "     ✓ Created bucket: backups (for database backups)"

# Create bucket for logs
docker exec stripe-minio mc mb --ignore-existing $MINIO_ALIAS/logs
echo "     ✓ Created bucket: logs (for application logs)"

echo ""
echo "[3/4] Setting bucket policies (public read for mlflow)..."
docker exec stripe-minio mc anonymous set download $MINIO_ALIAS/mlflow
echo "     ✓ MLflow bucket is publicly readable"

echo ""
echo "[4/4] Verifying setup..."
docker exec stripe-minio mc ls $MINIO_ALIAS

echo ""
echo "============================================"
echo "  MinIO Initialization Complete!"
echo "============================================"
echo ""
echo "Access MinIO Console:"
echo "  URL: http://localhost:9001"
echo "  Username: $MINIO_ACCESS_KEY"
echo "  Password: $MINIO_SECRET_KEY"
echo ""
echo "Buckets created:"
echo "  - mlflow    (ML models and artifacts)"
echo "  - datalake  (raw data storage)"
echo "  - backups   (database backups)"
echo "  - logs      (application logs)"
echo ""
