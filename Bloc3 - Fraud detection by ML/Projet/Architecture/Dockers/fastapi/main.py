from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Set tracking URI for MLFlow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Load model at startup
model = None


def getModelRunID():
    """
    Retrieve the run ID of the best registered model from MLflow.
    
    This function connects to the MLflow tracking server and retrieves the best model
    based on the experiment name and model prefix specified in environment variables.
    
    Environment Variables:
        EXPERIMENT_NAME (str, optional): The name of the MLflow experiment. 
            Defaults to "LBPFraudDetector".
        MODEL_PREFIX (str, optional): The prefix for the registered model name. 
            Defaults to "LBP_fraud_detector_".
    
    Returns:
        RegisteredModel: The MLflow registered model object containing metadata about 
            the best model.
    
    Raises:
        Exception: If the specified experiment is not found in the MLflow tracking server.
    
    Example:
        >>> model = getModelRunID()
        >>> print(model.name)
        LBP_fraud_detector_best_model
    """
    client = mlflow.tracking.MlflowClient()
    # Get the best model from the experiment
    experiment = client.get_experiment_by_name(os.getenv("EXPERIMENT_NAME", "LBPFraudDetector"))
    if experiment is None:
        raise Exception("Experiment not found")

    best_model_name = os.getenv("MODEL_PREFIX", "LBP_fraud_detector_") + "best_model"
    best_model = client.get_registered_model(best_model_name)

    print(f"[INFO] Registered Model: {best_model.name}")



@app.on_event("startup")
async def load_model():
    global model
    try:
        # Set model information - update with your model run ID
        test = getModelRunID()
        logged_model = os.getenv("MODEL_URI", "runs:/bf781e6a105445afa07d064f8f1f30a3/fraud_detector")
        model = mlflow.pyfunc.load_model(logged_model)
        print(f"Model loaded successfully from {logged_model}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

class Transaction(BaseModel):
    amt: float
    city_pop: int
    unix_time: float
    merch_lat: float
    merch_long: float
    age: int
    category_entertainment: int
    category_food_dining: int
    category_gas_transport: int
    category_grocery_net: int
    category_grocery_pos: int
    category_health_fitness: int
    category_home: int
    category_kids_pets: int
    category_misc_net: int
    category_misc_pos: int
    category_personal_care: int
    category_shopping_net: int
    category_shopping_pos: int
    category_travel: int

class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction.dict()])

        # Make prediction
        prediction = model.predict(df)

        return PredictionResponse(
            prediction=int(prediction[0]),
            is_fraud=bool(prediction[0] == 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(transactions: List[Transaction]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert transactions to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions])

        # Make predictions
        predictions = model.predict(df)

        return {
            "predictions": [int(p) for p in predictions],
            "fraud_count": int(sum(predictions)),
            "total_count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    getModelRunID()  