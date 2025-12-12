import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from datetime import datetime
import warnings
import traceback


current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, "..", "..")) + "/"
sys.path.append(os.path.join(current_path, 'libraries'))

warnings.filterwarnings("ignore", category=DeprecationWarning)

#======================================
# Charger le .env
#======================================

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.append('./Bloc4 - Final project/Projet/Architecture')

env_path = find_dotenv()
load_dotenv(env_path, override=True)

dotenv_path = find_dotenv()

# Charger explicitement un .env
load_dotenv(".env")  # Depuis le répertoire courant

print(f"✅ .env chargé dans main.py")
print(f"   TesVAr : {os.getenv('MAIL_FROM_NAME')}")

# Global model variable
separator = ("="*80)
EXPERIMENT_NAME = "LBPFraudDetector"

#======================================
# Helper functions
#======================================

def getModelRunID():
    # Récupérer la dernière version du modèle
    client = mlflow.MlflowClient()

    models = client.search_registered_models()

    if not models:
        print("[ERROR] Aucun modèle enregistré dans MLflow Registry")
        return None

    best_model = sorted(models, key=lambda x: x.creation_timestamp, reverse=True)[0]
    best_model_name = best_model.name
    
    # Print all model information
    print(f"[INFO] Model Details:")
    print(f"  Name: {best_model.name}")
    print(f"  Creation Timestamp: {best_model.creation_timestamp}")
    print(f"  Last Updated: {best_model.last_updated_timestamp}")
    print(f"  Description: {best_model.description}")
    print(f"  Tags: {best_model.tags}")
    print(f"  Latest Versions: {best_model.latest_versions}")
    print(f"  RunID: {best_model.latest_versions[0].run_id}")

    if not best_model.latest_versions:
        print("[ERROR] Le modèle n'a aucune version enregistrée")
        return None
    
    print(f"  RunID: {best_model.latest_versions[0].run_id}")
    print(f"[INFO] Latest registered model: {best_model_name}")

    # latest = client.get_latest_versions(
    #     best_model_name, stages=["None"]
    # )
    
    latest = client.search_model_versions(f"name='{best_model_name}'")

    if latest:
        model_version = latest[-1].version
        print(f"[INFO] Model logged as version {model_version}")

        # Mettre à jour l’alias "candidate"
        client.set_registered_model_alias(
            name=best_model_name,
            alias="production",
            version=model_version,
        )
        print(f"[INFO] Alias '{client.get_registered_model(best_model_name).aliases}' now points to version {model_version}")
        return best_model#.latest_versions[0].run_id
    else:
        print("[WARN] Aucun modèle trouvé dans le registre.")

def Preprocessor(eda_input_dataframe : pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame for exploratory data analysis (EDA).
    
    This function performs feature engineering and data cleanup to prepare
    the dataset for visualization and analysis. It calculates derived features
    such as distance, age, and temporal features, then removes columns that
    are no longer needed.
    
    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.   
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for EDA.
        
    """
    if "Column1" in eda_input_dataframe.columns:
        eda_input_dataframe.drop(columns=["Column1"], inplace=True)

    eda_input_dataframe = eda_input_dataframe.astype({col: "float64" for col in eda_input_dataframe.select_dtypes(include=["int"]).columns})
    # STEP 1: Handle datetime columns FIRST
    print("Converting datetime columns...")

    # Define which columns are datetime columns


    eda_input_dataframe['ccn_len'] = eda_input_dataframe['cc_num'].astype(str).str.len()
    eda_input_dataframe['bin'] = pd.to_numeric(eda_input_dataframe['cc_num'].astype(str).str[:6], errors='coerce')
    eda_input_dataframe['distance_km'] = eda_input_dataframe.apply(
        lambda row: haversine(row['long'], row['lat'], row['merch_long'], row['merch_lat']), 
        axis=1
    )


    print(f"📍 Distance calculated. Min: {eda_input_dataframe['distance_km'].min():.2f} km, "
        f"Max: {eda_input_dataframe['distance_km'].max():.2f} km, "
        f"Mean: {eda_input_dataframe['distance_km'].mean():.2f} km")

    # Convert dob to datetime and calculate age
    if not pd.api.types.is_datetime64_any_dtype(eda_input_dataframe['dob']):
        eda_input_dataframe['dob'] = pd.to_datetime(eda_input_dataframe['dob'], errors='coerce')

    eda_input_dataframe['age'] = ((pd.Timestamp.now() - eda_input_dataframe['dob']).dt.days // 365).astype('float32')
    #df = df.sort_values(by='age', ascending=True)

    # Convert amt to numeric
    eda_input_dataframe['amt'] = pd.to_numeric(eda_input_dataframe['amt'], errors='coerce').astype('float32')
    
    # Extract hour from transaction datetime
    if 'trans_date_trans_time' in eda_input_dataframe.columns:
        eda_input_dataframe['trans_hour'] = pd.to_datetime(eda_input_dataframe['trans_date_trans_time']).dt.hour.astype('float32')
    else:
        eda_input_dataframe['trans_hour'] = eda_input_dataframe['current_time']/1000

    # Drop columns that are no longer needed (only if they exist)
   
    columns_to_drop = [
        'dob', 'trans_date_trans_time', 'unix_time', 'merchant', 'gender', 'state',
        'lat', 'long', 'merch_lat', 'merch_long', 'city', 'zip', 'city_pop', 'job', 'bin',
        'street', 'first', 'last', 'Column1', 'trans_num', "unamed: 0", 'current_time', 'trans_hour'
    ]
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in eda_input_dataframe.columns]
    
    numeric_cols = eda_input_dataframe.select_dtypes(include=['number']).columns
    corr = eda_input_dataframe[numeric_cols].corr()

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr, annot=True, fmt=".2f", cmap=jedhaCMInverted, square=True)
    # plt.title("Matrice de corrélation des variables numériques")
    # #plt.show()
    # plt.savefig(current_path + '/outputs/Analysis_correlationMatrix_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
    # plt.close()

    #drawCorrelationMatrix(df.drop(columns=["is_fraud"], inplace=False), title_suffix="_before_preprocessing", current_path=current_path)

    if existing_columns_to_drop:
        eda_input_dataframe.drop(columns=existing_columns_to_drop, inplace=True, axis=1)
        print(f"Dropped columns: {existing_columns_to_drop}")
    else:
        print("No columns to drop (already removed or not present).")

    #drawCorrelationMatrix(df.drop(columns=["is_fraud"], inplace=False), title_suffix="_after_preprocessing", current_path=current_path)
    
    print("✅ Preprocessing complete.")
    
    return eda_input_dataframe

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r
    
def datetimeConverter(df: pd.DataFrame, datetime_columns: list) -> None:
    """Convert specified columns in a DataFrame to datetime dtype.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        datetime_columns (list): List of column names to convert to datetime.
    """
    for col in datetime_columns:
        if col in df.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"✅  {col}: converted to datetime64")
                else:
                    print(f"⊘ {col}: already datetime64")
            except Exception as e:
                print(f"✗ {col}: Failed to convert ({e})")

def getMyModel():
    """Load the MLflow model using the best run ID."""
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://davidrambeau-bloc3-mlflow.hf.space")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
                
    modelRunID = getModelRunID()

    if modelRunID is None:
        print("⚠️ WARNING: Aucun modèle disponible")
        return None
    
    try:
        run_id = modelRunID.latest_versions[0].run_id
        model_name = modelRunID.latest_versions[0].name
        
        print(f"Model runid: {run_id}")
        print(f"Model name: {model_name}")

        logged_model = os.getenv("MODEL_URI", f"runs:/{run_id}/{EXPERIMENT_NAME}")
        print(f"Loading model from: {logged_model}")
        
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Model type: {type(loaded_model)}")
        print(f"✅ Model dir: {dir(loaded_model)}")
        print(f"✅ Has predict: {hasattr(loaded_model, 'predict')}")

        return loaded_model
        
    except Exception as e:
        print(f"❌ Error loading PyFunc model: {str(e)}")
        traceback.print_exc()
        return None

def load_reference_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load a reference dataset for model quality tests."""
    dataset_path =  os.path.join(os.path.dirname(project_path),"localModel", "datasSources", "inputDataset", "fraudTest.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Reference dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"✅ Reference dataset loaded from {dataset_path} with shape {df.shape}")
    
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    
    return X, y

#======================================
# Lifespan context manager
#======================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application FastAPI, notamment le chargement du modèle MLflow au démarrage."""
    try:
        app.state.loaded_model = getMyModel()
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        app.state.loaded_model = None
    yield
    
    print("Shutting down...")


#======================================
# FastAPI application setup
#======================================

app = FastAPI( title="LBFraud Detection API", version="1.0.0",lifespan=lifespan, debug=True )


#======================================
# Pydantic models for request and response validation
#======================================

class Transaction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    cc_num: int  # 
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int  # 
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    merch_lat: float
    merch_long: float
    is_fraud: int  # 
    current_time: int  #

class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    trans_num: Optional[str] = None
    amt: Optional[float] = None
    category: Optional[str] = None

#======================================
# API Endpoints
#======================================

@app.get("/")
async def root():
    return {
        "message": "LBP Fraud Detection API",
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
        "model_loaded": getattr(app.state, "loaded_model", None) is not None,
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "https://davidrambeau-bloc3-mlflow.hf.space")
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Make a fraud prediction for a single transaction.
    """
    # Convert Transaction object to DataFrame
    trans_dict = pd.DataFrame([transaction.model_dump()])
    trans_dict = Preprocessor(trans_dict)

    print(separator)
    print(f"📥 Received transaction for prediction at : {datetime.now().strftime('%H:%M:%S')} ")
    print(separator)

    # Safely get loaded model
    loaded_model = getattr(app.state, "loaded_model", None)
    if loaded_model is None:
        print("❌ Model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check model type
    if not hasattr(loaded_model, 'predict'):
        print(f"❌ Invalid model type: {type(loaded_model)}")
        raise HTTPException(status_code=500, detail="Model does not have predict method")
    
    try:
        print(f"📊 DataFrame shape: {trans_dict.shape}")
        print(f"📊 DataFrame columns: {trans_dict.columns.tolist()}")
        
        # Make prediction
        prediction = loaded_model.predict(trans_dict)
        
        # Get model metadata
        modelRunID = getModelRunID()
        if modelRunID is None:
            model_name = "Unknown"
            model_version = "Unknown"
        else:
            model_name = modelRunID.name
            model_version = modelRunID.latest_versions[0].version

        print(f"✅ Prediction: {prediction[0]} | Model: {model_name} (v{model_version})")
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            is_fraud=bool(prediction[0] == 1),
            trans_num=transaction.trans_num,
            amt=transaction.amt,
            category=transaction.category,
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")