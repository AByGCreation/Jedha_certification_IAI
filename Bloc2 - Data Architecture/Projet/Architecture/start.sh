#!/bin/bash
# =============================================================================
# QUICK START SCRIPT FOR STRIPE DATA ARCHITECTURE
# =============================================================================
# This script loads credentials and starts all Docker services
# Usage: ./start.sh
# =============================================================================

set -e  # Exit on error

echo "============================================"
echo "  Stripe Data Architecture - Quick Start"
echo "============================================"
echo ""

# Check if secret.sh exists
if [ ! -f "secret.sh" ]; then
    echo "❌ ERROR: secret.sh file not found!"
    echo "   Please create the secret.sh file with your credentials."
    exit 1
fi

# Load environment variables
echo "📋 Loading credentials from secret.sh..."
source secret.sh

# Verify critical variables are set
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
    echo "❌ ERROR: Required environment variables are not set!"
    echo "   Please check your secret.sh file."
    exit 1
fi

echo "✓ Credentials loaded successfully"
echo ""


# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Docker is not running. Attempting to start Docker daemon..."
    
    if command -v systemctl &> /dev/null; then
        echo "try with Systemctl"
        sudo systemctl daemon-reload
        sudo systemctl restart docker
    elif command -v service &> /dev/null; then
        echo "try with Service"
        sudo service docker start
    else
        echo "❌ ERROR: Could not start Docker daemon!"
        echo "   Please start Docker manually and try again."
        exit 1
    fi
    
    echo "✓ Docker daemon started"
else
    echo "✓ Docker is running"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running!"
    echo "   Please start Docker and try again."
    
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Start Docker services
echo "🚀 Starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for core services to be ready (30 seconds)..."
sleep 30

echo ""
echo "🔧 Initializing MinIO buckets..."
docker exec stripe-minio mc alias set stripe-minio http://minio:9000 stripe_user stripe_password 2>/dev/null
docker exec stripe-minio mc mb --ignore-existing stripe-minio/mlflow 2>/dev/null
docker exec stripe-minio mc mb --ignore-existing stripe-minio/datalake 2>/dev/null
docker exec stripe-minio mc mb --ignore-existing stripe-minio/backups 2>/dev/null
docker exec stripe-minio mc mb --ignore-existing stripe-minio/logs 2>/dev/null
docker exec stripe-minio mc anonymous set download stripe-minio/mlflow 2>/dev/null
echo "✓ MinIO buckets created"

echo ""
echo "📊 Checking service status..."
docker-compose ps





echo ""
echo "============================================"
echo "  ✅ All Services Started Successfully!"
echo "============================================"
echo ""
echo "📝 Note: Some services may take 2-5 minutes to become fully healthy"
echo "   (Elasticsearch, Airflow, MLflow, Neo4j)"
echo ""
echo "Access your services at:"
echo "  • FastAPI:         http://localhost:8000"
echo "  • Flask:           http://localhost:5050"
echo "  • Airflow:         http://localhost:8080  (user: admin, pass: stripe_password)"
echo "  • Grafana:         http://localhost:3000  (user: admin, pass: stripe_password)"
echo "  • MinIO Console:   http://localhost:9001  (user: stripe_user, pass: stripe_password)"
echo "  • Neo4j Browser:   http://localhost:7474 (user: neo4j, pass: stripe_password)"
echo "  • Kibana:          http://localhost:5601"
echo ""
echo "Database ports:"
echo "  • PostgreSQL:      localhost:5432"
echo "  • MongoDB:         localhost:27017"
echo "  • Redis:           localhost:6379"
echo "  • ClickHouse:      localhost:8123 (HTTP), localhost:9000 (Native)"
echo "  • Elasticsearch:   localhost:9200"
echo "  • Neo4j:           localhost:7687 (Bolt)"
echo "  • MinIO API:       localhost:9002"
echo ""
echo "Useful commands:"
echo "  • Check status:       docker-compose ps"
echo "  • View logs:          docker-compose logs -f [service-name]"
echo "  • Stop services:      docker-compose down"
echo "  • Restart service:    docker-compose restart [service-name]"
echo ""
echo "Documentation:"
echo "  • System status:      cat CURRENT_STATUS.md"
echo "  • Credentials:        cat CREDENTIALS_REFERENCE.md"
echo "  • Expose to internet: cat EXPOSE_TO_INTERNET.md"
echo ""
