#!/bin/bash
# =============================================================================
# Train and Deploy Fraud Detection Model
# =============================================================================
# This script trains the ML model and makes it available to FastAPI
# =============================================================================

echo "============================================"
echo "  Training Fraud Detection Model"
echo "============================================"
echo ""

# Step 1: Train the model using the training script
echo "[1/4] Training model in MLflow container..."
docker exec stripe-mlflow python -c "
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

np.random.seed(42)
n_samples = 1000

# Features: 10 engineered features
X = np.random.randn(n_samples, 10)
y = np.random.binomial(1, 0.15, n_samples)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

os.makedirs('/mlflow/model', exist_ok=True)
joblib.dump(model, '/mlflow/model/logistic_regression_fraud.pkl')

print(f'Model trained! Accuracy: {model.score(X_test, y_test):.2%}')
"

if [ $? -eq 0 ]; then
    echo "     ✓ Model trained successfully"
else
    echo "     ✗ Failed to train model"
    exit 1
fi

echo ""
echo "[2/4] Verifying model file..."
docker exec stripe-mlflow ls -lh /mlflow/model/logistic_regression_fraud.pkl

echo ""
echo "[3/4] Uploading model to MinIO for FastAPI access..."
docker exec stripe-minio mc cp /mlflow/model/logistic_regression_fraud.pkl stripe-minio/mlflow/logistic_regression_fraud.pkl 2>/dev/null || (
    echo "     Using docker cp method..."
    docker cp mlflow/model/logistic_regression_fraud.pkl stripe-minio:/tmp/model.pkl
    docker exec stripe-minio mc cp /tmp/model.pkl stripe-minio/mlflow/logistic_regression_fraud.pkl
)

echo "     ✓ Model uploaded to MinIO"

echo ""
echo "[4/4] Restarting FastAPI to load new model..."
docker-compose restart fastapi

echo ""
echo "============================================"
echo "  Model Training Complete!"
echo "============================================"
echo ""
echo "Model Location:"
echo "  • MLflow:  /mlflow/model/logistic_regression_fraud.pkl"
echo "  • MinIO:   s3://mlflow/logistic_regression_fraud.pkl"
echo "  • FastAPI: /mlflow/model/logistic_regression_fraud.pkl"
echo ""
echo "Model Details:"
echo "  • Algorithm: Logistic Regression"
echo "  • Features: 10 engineered features"
echo "  • Fraud Threshold: 70%"
echo ""
echo "Test the model:"
echo '  curl -X POST http://localhost:8000/api/transactions \'
echo '       -H "Content-Type: application/json" \'
echo '       -d "{\"amount\": 1500, \"user_id\": \"user_001\", \"merchant_id\": \"merchant_001\"}"'
echo ""
