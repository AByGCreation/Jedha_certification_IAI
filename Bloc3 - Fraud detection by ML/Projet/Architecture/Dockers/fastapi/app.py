from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    try:
        # Configuration MLFlow avec authentification
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://davidrambeau-bloc3-mlflow.hf.space")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Authentification Hugging Face (si nécessaire)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = hf_token
            # Ou utiliser les headers HTTP
            os.environ["MLFLOW_TRACKING_USERNAME"] = "hf_token"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = hf_token
        
        # Charger le modèle
        logged_model = os.getenv("MODEL_URI", "runs:/55dd4bb4b7774f32984e374fba65f1dc/RandomForest")
        model = mlflow.pyfunc.load_model(logged_model)
        print(f"✅ Model loaded successfully from {logged_model}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        # L'API démarre quand même pour permettre le diagnostic
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="Fraud Detection API", 
    version="1.0.0",
    lifespan=lifespan
)

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

class ExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    total_runs: int
    last_update: Optional[str] = None

class MLflowTestResponse(BaseModel):
    status: str
    tracking_uri: str
    connection_success: bool
    total_experiments: int
    experiments: List[ExperimentInfo]
    error: Optional[str] = None



@app.get("/")
async def root():
    return {
        "message": "LBP Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "mlflow_test": "/mlflow-test",
            "mlflow_experiments": "/mlflow-experiments",
            "mlflow_experiment_detail": "/mlflow-experiment/{experiment_id}"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "https://davidrambeau-bloc3-mlflow.hf.space")
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([transaction.model_dump()])
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
        df = pd.DataFrame([t.model_dump() for t in transactions])
        predictions = model.predict(df)
        
        return {
            "predictions": [int(p) for p in predictions],
            "fraud_count": int(sum(predictions)),
            "total_count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/mlflow-test")
async def test_mlflow_connection():
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return {
            "status": "connected",
            "experiments_count": len(experiments),
            "tracking_uri": mlflow.get_tracking_uri()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "tracking_uri": mlflow.get_tracking_uri()
        }


@app.get("/mlflow-test", response_model=MLflowTestResponse)
async def test_mlflow_connection():
    """
    Teste la connexion à MLflow et liste toutes les expériences disponibles
    """
    tracking_uri = mlflow.get_tracking_uri()
    
    try:
        client = MlflowClient()
        
        # Récupérer toutes les expériences
        experiments = client.search_experiments()
        
        # Construire la liste des expériences avec détails
        experiment_list = []
        for exp in experiments:
            # Récupérer les runs de l'expérience
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1000
            )
            
            # Trouver la dernière mise à jour
            last_update = None
            if runs:
                last_update = max(run.info.start_time for run in runs)
                last_update = pd.to_datetime(last_update, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            
            experiment_list.append(
                ExperimentInfo(
                    experiment_id=exp.experiment_id,
                    name=exp.name,
                    artifact_location=exp.artifact_location,
                    lifecycle_stage=exp.lifecycle_stage,
                    total_runs=len(runs),
                    last_update=last_update
                )
            )
        
        return MLflowTestResponse(
            status="connected",
            tracking_uri=tracking_uri,
            connection_success=True,
            total_experiments=len(experiments),
            experiments=experiment_list
        )
        
    except Exception as e:
        return MLflowTestResponse(
            status="error",
            tracking_uri=tracking_uri,
            connection_success=False,
            total_experiments=0,
            experiments=[],
            error=str(e)
        )

@app.get("/mlflow-experiments")
async def list_experiments(
    active_only: bool = True,
    limit: Optional[int] = None
):
    """
    Liste toutes les expériences MLflow avec option de filtrage
    
    - **active_only**: Afficher uniquement les expériences actives (non supprimées)
    - **limit**: Nombre maximum d'expériences à retourner
    """
    try:
        client = MlflowClient()
        
        # Filtrer les expériences
        filter_string = "lifecycle_stage = 'active'" if active_only else None
        experiments = client.search_experiments(filter_string=filter_string)
        
        # Limiter le nombre de résultats
        if limit:
            experiments = experiments[:limit]
        
        # Formater les résultats
        results = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            results.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "total_runs": len(runs),
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location
            })
        
        return {
            "status": "success",
            "total_experiments": len(results),
            "experiments": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching experiments: {str(e)}")

@app.get("/mlflow-experiment/{experiment_id}")
async def get_experiment_detail(
    experiment_id: str,
    max_runs: int = 10
):
    """
    Récupère les détails d'une expérience spécifique avec ses runs
    
    - **experiment_id**: ID de l'expérience
    - **max_runs**: Nombre maximum de runs à retourner
    """
    try:
        client = MlflowClient()
        
        # Récupérer l'expérience
        experiment = client.get_experiment(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        # Récupérer les runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_runs,
            order_by=["start_time DESC"]
        )
        
        # Formater les runs
        runs_data = []
        for run in runs:
            runs_data.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "status": run.info.status,
                "start_time": pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
        
        return {
            "status": "success",
            "experiment": {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "total_runs": len(client.search_runs(experiment_ids=[experiment_id]))
            },
            "runs": runs_data,
            "returned_runs": len(runs_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching experiment details: {str(e)}")