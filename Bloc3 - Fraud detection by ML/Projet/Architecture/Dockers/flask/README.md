# Flask Transaction Simulator

Real-time transaction simulation with instant fraud detection using MLFlow model.

## Overview

This Flask application provides an interactive web interface to:
- Simulate realistic credit card transactions
- Get instant fraud predictions from the trained MLFlow model
- Save transactions with predictions to the database
- Batch simulate multiple transactions for testing

## Features

### Transaction Generation
- Realistic transaction data including:
  - Random amounts (fraud: $100-$1000, normal: $5-$200)
  - US cities and locations with coordinates
  - Transaction categories and merchant names
  - Customer demographics (age, gender, job, etc.)
  - Merchant locations (fraud: further from customer)

### Real-time Fraud Detection
- Loads trained model from MLFlow on startup
- Instant predictions on simulated transactions
- Confidence scoring for predictions
- Comparison with actual fraud status

### Database Integration
- Automatic saving of transactions with predictions
- Uses same PostgreSQL database as other services
- Tracks prediction accuracy over time

## Endpoints

### `GET /`
Main web interface with interactive buttons

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mlflow_uri": "http://mlflow:4000"
}
```

### `POST /simulate`
Simulate a single transaction
```json
{
  "force_fraud": false  // Optional: force fraudulent transaction
}
```

Response:
```json
{
  "success": true,
  "transaction": {
    "id": "TXN1234567",
    "amount": 45.23,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "grocery_pos",
    "date": "2024-01-15 14:30:00",
    "city": "New York",
    "state": "NY"
  },
  "prediction": {
    "is_fraud": false,
    "confidence": "low",
    "actual_fraud": 0
  },
  "saved_to_db": true
}
```

### `POST /simulate/batch`
Simulate multiple transactions
```json
{
  "count": 10,        // Number of transactions
  "fraud_rate": 0.2   // Percentage of fraudulent transactions
}
```

Response:
```json
{
  "success": true,
  "count": 10,
  "accuracy": 85.5,
  "transactions": [
    {
      "id": "TXN1234567",
      "amount": 45.23,
      "category": "grocery_pos",
      "actual_fraud": 0,
      "predicted_fraud": 0
    }
  ]
}
```

## Usage

### Running with Docker Compose

The simulator is integrated into the docker-compose.yml file:

```bash
# Start all services including simulator
docker-compose up -d

# View simulator logs
docker-compose logs -f flask_simulator

# Stop all services
docker-compose down
```

Access the web interface at: http://localhost:5000

### Running Standalone

```bash
cd flask_simulator

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MLFLOW_TRACKING_URI=http://localhost:4000
export MODEL_URI=runs:/bf781e6a105445afa07d064f8f1f30a3/fraud_detector
export BACKEND_STORE_URI="postgresql://..."

# Run the application
python app.py
```

## Web Interface

The web interface provides three main actions:

1. **Simulate Normal Transaction** (Green button)
   - Generates a realistic legitimate transaction
   - Shows instant fraud detection result
   - Displays full transaction details

2. **Simulate Fraudulent Transaction** (Red button)
   - Generates a transaction with fraud indicators
   - Higher amounts, distant merchant locations
   - Tests model's fraud detection capability

3. **Simulate Batch (10 transactions)** (Blue button)
   - Generates 10 transactions with 20% fraud rate
   - Shows aggregate statistics
   - Displays accuracy metrics
   - Lists all transactions with predictions

## Configuration

Environment variables (set in `.env` or docker-compose):

```env
MLFLOW_TRACKING_URI=http://mlflow:4000
MODEL_URI=runs:/bf781e6a105445afa07d064f8f1f30a3/fraud_detector
BACKEND_STORE_URI=postgresql://neondb_owner:...
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-north-1
```

## Architecture Integration

The Flask simulator integrates with:

- **MLFlow Server** (port 4000): Loads trained fraud detection model
- **PostgreSQL Database**: Saves transactions with predictions
- **FastAPI Service** (port 8000): Alternative prediction API
- **Streamlit Dashboard** (port 8501): Visualizes saved transactions

## Development

### Project Structure
```
flask_simulator/
├── app.py                  # Main Flask application
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── README.md            # This file
```

### Adding New Features

To add custom transaction scenarios:
1. Modify `generate_random_transaction()` in app.py
2. Add new parameters to control transaction characteristics
3. Update the web interface to include new simulation options

### Model Updates

When you retrain the model:
1. Update `MODEL_URI` in docker-compose.yml with new run ID
2. Restart the simulator: `docker-compose restart flask_simulator`
3. Model will reload automatically on startup

## Troubleshooting

### Model Not Loading
- Check MLFlow server is running: `curl http://localhost:4000/health`
- Verify MODEL_URI points to valid run ID
- Check MLFlow tracking URI is correct

### Database Connection Issues
- Verify BACKEND_STORE_URI is correct
- Ensure transactions table exists (run create_database.py)
- Check network connectivity to Neon PostgreSQL

### Port Already in Use
- Change port mapping in docker-compose.yml
- Default: `"5000:5000"` → `"5001:5000"` (use port 5001 on host)

## Testing

Test the simulator with curl:

```bash
# Health check
curl http://localhost:5000/health

# Simulate normal transaction
curl -X POST http://localhost:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"force_fraud": false}'

# Simulate fraudulent transaction
curl -X POST http://localhost:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"force_fraud": true}'

# Batch simulation
curl -X POST http://localhost:5000/simulate/batch \
  -H "Content-Type: application/json" \
  -d '{"count": 10, "fraud_rate": 0.2}'
```

## Performance

- Startup time: ~5-10 seconds (model loading)
- Prediction time: ~50-100ms per transaction
- Batch processing: ~500ms for 10 transactions
- Memory usage: ~200-300MB

## Security Notes

- This is a simulation tool for testing
- Do not expose to public internet without authentication
- Generated credit card numbers are random and fake
- Customer data is randomly generated (not real people)
