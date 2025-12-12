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

from apitally.fastapi import ApitallyMiddleware
from evidently.ui.workspace import CloudWorkspace
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import TextEvals
from evidently.tests import lte, gte, eq
from evidently.descriptors import LLMEval, TestSummary, DeclineLLMEval, Sentiment, TextLength, IncludesWords
from evidently.llm.templates import BinaryClassificationPromptTemplate

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

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
    
    if aws_access_key and aws_secret_key:
        print(f"🔑 AWS credentials found (key: {aws_access_key[:10]}***)")
        print(f"🌍 AWS region: {aws_region}")
        
        # Configurer boto3 explicitement
        import boto3
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # OU configurer via variables d'environnement
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region
        
        print("✅ Boto3 configured with credentials")
    else:
        print("⚠️ AWS credentials not found in environment")

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
        app.state.X_ref, app.state.y_ref = load_reference_dataset()
        print(f"✅ Reference dataset loaded: shape {app.state.X_ref.shape}")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        app.state.loaded_model = None
        app.state.X_ref = None
        app.state.y_ref = None
    yield
    
    print("Shutting down...")


#======================================
# FastAPI application setup
#======================================

app = FastAPI( title="LBFraud Detection API", version="1.0.0",lifespan=lifespan, debug=True )

#======================================
# Apitally Middleware for API monitoring
#======================================
# Add Apitally middleware for API monitoring
apitally_client_id = os.getenv("APITALLY_CLIENT_ID")
if apitally_client_id:
    app.add_middleware(
        ApitallyMiddleware,
        client_id=apitally_client_id,
        env=os.getenv("APITALLY_ENV", "production"),
    )
    print(f"✅ Apitally monitoring enabled (env: {os.getenv('APITALLY_ENV', 'production')})")
else:
    print("⚠️ APITALLY_CLIENT_ID not found in environment - monitoring disabled")


#======================================
# evidently Cloud setup
#======================================

#======================================
# Column Mapping for Evidently
#======================================

# column_mapping = ColumnMapping(
#     target='is_fraud',
#     prediction='prediction',
#     numerical_features=[col for col in ['ccn_len', 'distance_km', 'age', 'amt', 'trans_hour'] 
#                        if col in ['ccn_len', 'distance_km', 'age', 'amt', 'trans_hour']],
#     categorical_features=['category', 'state', 'gender'],
# )


evid_token = os.getenv("evidentlyToken")
EVIDENTLY_WORKSPACE = None

if evid_token:
    try:
        EVIDENTLY_WORKSPACE = CloudWorkspace(token=evid_token, url="https://app.evidently.cloud")
        EVprojectID = "019b0d47-3b9a-7551-8b13-5bd66170a8fc"
        project = EVIDENTLY_WORKSPACE.get_project(EVprojectID)
        print(f"✅ Evidently Cloud workspace initialized → {project}" )

        data = [
            ["What is the chemical symbol for gold?", "Gold chemical symbol is Au."],
            ["What is the capital of Japan?", "The capital of Japan is Tokyo."],
            ["Tell me a joke.", "Why don't programmers like nature? Too many bugs!"],
            ["When does water boil?", "Water's boiling point is 100 degrees Celsius."],
            ["Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."],
            ["What’s the fastest animal on land?", "The cheetah is the fastest land animal, capable of running up to 75 miles per hour."],
            ["Can you help me with my math homework?", "I'm sorry, but I can't assist with homework."],
            ["How many states are there in the USA?", "USA has 50 states."],
            ["What’s the primary function of the heart?", "The primary function of the heart is to pump blood throughout the body."],
            ["Can you tell me the latest stock market trends?", "I'm sorry, but I can't provide real-time stock market trends. You might want to check a financial news website or consult a financial advisor."]
        ]
        columns = ["question", "answer"]

        eval_df = pd.DataFrame(data, columns=columns)

        eval_dataset = Dataset.from_pandas(
            eval_df,
            data_definition=DataDefinition(),
            descriptors=[
                Sentiment("answer", alias="Sentiment"),
                TextLength("answer", alias="Length"),
                DeclineLLMEval("answer", alias="Denials")]) 
        
        eval_dataset.as_dataframe()
        report = Report([
            TextEvals()
        ])

        my_eval = report.run(eval_dataset, None)

# Or IncludesWords("answer", words_list=['sorry', 'apologize'], alias="Denials")


    except Exception as e:
        print(f"⚠️ Could not connect to Evidently Cloud: {str(e)}")
else:
    print("⚠️ evidentlyToken not found - Evidently Cloud disabled")



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
    

#======================================
# Evidently Cloud setup
#======================================

@app.get("/monitoring/load_dataset")
async def load_dataset():
    """Load the reference dataset for monitoring."""
    try:
        X_ref, y_ref = load_reference_dataset()
        app.state.X_ref = X_ref
        app.state.y_ref = y_ref
        return {"status": "success", "dataset_loaded": True, "shape": X_ref.shape}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset loading error: {str(e)}")
    



@app.get("/monitoring/data-quality")
async def data_quality_report():
    """Generate a comprehensive data quality report."""
    try:
        X_ref = getattr(app.state, "X_ref", None)
        if X_ref is None:
            raise HTTPException(status_code=503, detail="Reference dataset not loaded")
        
        report = Report(metrics=[
            DataQualityPreset(),
        ])
        report.run(reference_data=X_ref, current_data=X_ref)
        
        if EVIDENTLY_WORKSPACE:
            try:
                EVIDENTLY_WORKSPACE.add_report(
                    report,
                    name=f"Data Quality Report - {datetime.now().isoformat()}",
                    workspace="LBPFraudDetector"
                )
                print("✅ Report sent to Evidently Cloud")
            except Exception as e:
                print(f"⚠️ Could not send report to Evidently Cloud: {str(e)}")
        
        return {"status": "success", "report_generated": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitoring error: {str(e)}")

@app.post("/monitoring/drift-detection")
async def drift_detection(transaction: Transaction):
    """Detect data drift using DataDriftPreset."""
    try:
        X_ref = getattr(app.state, "X_ref", None)
        if X_ref is None:
            raise HTTPException(status_code=503, detail="Reference dataset not loaded")
        
        trans_df = pd.DataFrame([transaction.model_dump()])
        trans_df = Preprocessor(trans_df)
        
        report = Report(metrics=[
            DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.05),
        ])
        report.run(reference_data=X_ref, current_data=trans_df, column_mapping=column_mapping)
        
        if EVIDENTLY_WORKSPACE:
            try:
                EVIDENTLY_WORKSPACE.add_report(
                    report,
                    name=f"Drift Detection - {transaction.trans_num}",
                    workspace="LBPFraudDetector"
                )
                print("✅ Drift report sent to Evidently Cloud")
            except Exception as e:
                print(f"⚠️ Could not send drift report: {str(e)}")
        
        return {
            "status": "success",
            "drift_detected": True,
            "timestamp": datetime.now().isoformat(),
            "sent_to_cloud": EVIDENTLY_WORKSPACE is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")

@app.get("/monitoring/classification-performance")
async def classification_performance():
    """Generate classification performance report."""
    try:
        X_ref = getattr(app.state, "X_ref", None)
        y_ref = getattr(app.state, "y_ref", None)
        
        if X_ref is None or y_ref is None:
            raise HTTPException(status_code=503, detail="Reference dataset not loaded")
        
        # Add target and prediction columns for classification report
        X_ref_with_target = X_ref.copy()
        X_ref_with_target['is_fraud'] = y_ref
        X_ref_with_target['prediction'] = y_ref  # Use actual as prediction for baseline
        
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        report.run(reference_data=X_ref_with_target, current_data=X_ref_with_target, column_mapping=column_mapping)
        
        if EVIDENTLY_WORKSPACE:
            try:
                EVIDENTLY_WORKSPACE.add_report(
                    report,
                    name=f"Classification Performance - {datetime.now().isoformat()}",
                    workspace="LBPFraudDetector"
                )
            except Exception as e:
                print(f"⚠️ Could not send performance report: {str(e)}")
        
        return {"status": "success", "report_generated": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance monitoring error: {str(e)}")