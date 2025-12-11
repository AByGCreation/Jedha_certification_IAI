import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel , EmailStr, validator
from typing import List, Union, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from dotenv import load_dotenv , find_dotenv
from math import radians, cos, sin, asin, sqrt
import pandas as pd

from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType, MultipartSubtypeEnum


from typing import List


#============================= Load environment variables ============================#
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Global model variable
model = None
separator = ("="*80)
EXPERIMENT_NAME = "LBPFraudDetector"
conf = ConnectionConfig(
    MAIL_USERNAME = os.getenv("SENDGRID_USER_KEY"),
    MAIL_PASSWORD = os.getenv("SENDGRID_API_KEY"),
    MAIL_FROM = os.getenv("MAIL_FROM"),
    MAIL_PORT = 587,
    MAIL_SERVER = os.getenv("MAIL_SERVER"),
    MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME"),
    MAIL_STARTTLS = True,
    MAIL_SSL_TLS = False,
    USE_CREDENTIALS = True,
    VALIDATE_CERTS = True
)

class TransactionData(BaseModel):
    """Données de transaction reçues"""
    transaction_id: str
    amount: float
    merchant: str
    timestamp: Optional[str] = None
    user_email: str
    fraud_score: Optional[float] = None
    is_fraud: bool = False



class EmailSchema(BaseModel):
    email: Union[EmailStr, List[EmailStr]]
    
    @validator('email')
    def validate_email(cls, v):
        # Convertir un seul email en liste
        if isinstance(v, str):
            return [v]
        return v




#============================= Helper Functions ============================#

def getModelRunID():
    # Récupérer la dernière version du modèle
    client = mlflow.MlflowClient()

    models = client.search_registered_models()
    best_model = sorted(
        models,
        key=lambda x: x.creation_timestamp,
        reverse=True
    )[0]
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

    print(f"[INFO] Latest registered model: {best_model_name}")

    latest = client.get_latest_versions(
        best_model_name, stages=["None"]
    )
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

class Transaction(BaseModel):
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



#============================= Lifespan for FastAPI app ============================#

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    
    global model
    try:
        # Configuration MLFlow avec authentification
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://davidrambeau-bloc3-mlflow.hf.space")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
                
        # Charger le modèle
        modelRunID = getModelRunID()
        print(f"Model runid: {modelRunID.latest_versions[0].run_id}")
        print(f"Model name: {modelRunID.latest_versions[0].name}")

        logged_model = os.getenv("MODEL_URI", f"runs:/{modelRunID.latest_versions[0].run_id}/{EXPERIMENT_NAME}")
        model = mlflow.pyfunc.load_model(logged_model)
        print(f"✅ Model loaded successfully from {logged_model}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
    
    yield
    
    print("Shutting down...")

#============================ FastAPI Application ============================#




app = FastAPI( title="LBFraud Detection API", version="1.0.0",lifespan=lifespan, debug=True )

#============================ Mail Middleware ============================#
@app.post("/send-fraud-alert")
async def send_fraud_alert(email: EmailSchema):
    # Ajouter du logging
    print(f"Email reçu: {email.email}")
    print(f"Type: {type(email.email)}")

@app.post("/email")
async def simple_send(recipient: str, transaction: TransactionData):
    """Envoie un email de fraude en arrière-plan"""
    try:
        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 2px solid #d32f2f;">
                    <h2 style="color: #d32f2f;">⚠️ Alerte de Fraude Détectée</h2>
                    
                    <p><strong>Une transaction suspecte a été détectée sur votre compte.</strong></p>
                    
                    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <h3 style="margin-top: 0;">Détails de la transaction :</h3>
                        <table style="width: 100%;">
                            <tr>
                                <td><strong>ID Transaction:</strong></td>
                                <td>{transaction.transaction_id}</td>
                            </tr>
                            <tr>
                                <td><strong>Montant:</strong></td>
                                <td style="color: #d32f2f; font-size: 18px;">{transaction.amount} €</td>
                            </tr>
                            <tr>
                                <td><strong>Commerçant:</strong></td>
                                <td>{transaction.merchant}</td>
                            </tr>
                            <tr>
                                <td><strong>Date/Heure:</strong></td>
                                <td>{transaction.timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                            </tr>
                            {f'<tr><td><strong>Score de fraude:</strong></td><td>{transaction.fraud_score:.2%}</td></tr>' if transaction.fraud_score else ''}
                        </table>
                    </div>
                    
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <strong>🔒 Actions recommandées :</strong>
                        <ul>
                            <li>Si vous reconnaissez cette transaction, ignorez cet email</li>
                            <li>Sinon, bloquez votre carte immédiatement</li>
                            <li>Contactez notre support au 01 XX XX XX XX</li>
                            <li>Changez vos mots de passe</li>
                        </ul>
                    </div>
                    
                    <p style="color: #666; font-size: 12px; margin-top: 30px;">
                        Cet email a été généré automatiquement par notre système de détection de fraude.
                        Ne répondez pas à cet email.
                    </p>
                </div>
            </body>
        </html>
        """

        message = MessageSchema(
            subject=f"🚨 Alerte Fraude - Transaction {transaction.amount}€",
            recipients=[recipient],
            body=html,
            subtype=MessageType.html,
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"✅ Email envoyé à {recipient} pour transaction {transaction.transaction_id}")
        
    except Exception as e:
        print(f"❌ Erreur envoi email: {str(e)}")   
    # """Send a fraud alert email to the provided email address."""
    # html = "<b>This is a fraud alert email</b>"

    # message = MessageSchema(
    #     subject="Fastapi-Mail module",
    #     recipients=email.email,
    #     template_body=html,
    #     subtype=MessageType.html,
    #     alternative_body="This is a fraud alert email",
    #     multipart_subtype=MultipartSubtypeEnum.alternative,
    # )

    # fm = FastMail(conf)
    # await fm.send_message(message)
    # return JSONResponse(status_code=200, content={"message": "Une alerte de fraude a été envoyée"})
    # # message = Mail(
    # #     from_email='noreply@lbpfrauddetector.com',
    # #     to_emails='david.rambeau@gmail.com',
    # #     subject='🚨 Une alerte de fraude a été détectée !',
    # #     html_content='<strong>🚨 Alerte fraude détectée - Action requise immédiatement !!!!</strong>')
    # # try:
    # #     sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))

    # #     sg.set_sendgrid_data_residency("eu")
    # #     # sg.set_sendgrid_data_residency("eu")
    # #     # uncomment the above line if you are sending mail using a regional EU subuser
    # #     response = sg.send(message)
    # #     print(response.status_code)
    # #     print(response.body)
    # #     print(response.headers)
    # # except Exception as e:
    # #     print(e)    
    # # return {"message": f"Email sent successfully → Status code: {message}"}


#============================ API Endpoints ============================#

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

@app.get("/alert")
async def alert(message: str, background_tasks: BackgroundTasks):
    fake_transaction = TransactionData(
        transaction_id="TEST-001",
        amount=999.99,
        merchant="Test Store",
        user_email="david.rambeau@gmail.com",
        fraud_score=0.95,
        is_fraud=True
    )
    
    background_tasks.add_task(simple_send, "david.rambeau@gmail.com", fake_transaction)
    
    return {"message": "Email planifié", "recipient": "david.rambeau@gmail.com"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Make a fraud prediction for a single transaction.
    """

    # Convert Transaction object to DataFrame
    trans_dict = pd.DataFrame([transaction.model_dump()])
    trans_dict = Preprocessor(trans_dict)

    print(separator)
    print("🔮 Received transaction for prediction.")
    print(f"Transaction: {transaction}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convertir en DataFrame
        #df = pd.DataFrame([transaction.model_dump()])
        
        print(f"📊 DataFrame shape: {trans_dict.shape}")
        print(f"📊 DataFrame columns: {trans_dict.columns.tolist()}")
        print(f"📊 Data for prediction: {trans_dict.to_dict(orient='records')}")
        
        # ⚠️ CRITIQUE: Vérifier les colonnes attendues par le modèle
        print(f"📊 Model expected features: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Unknown'}")
        
        # Prédiction
        prediction = model.predict(trans_dict)
        
        print(f"✅ Prediction result: {prediction}")
        
        alert_message = "🚨 Alerte de fraude détectée!" if prediction[0] == 1 else "✅ Transaction légitime."
        alert(alert_message)

        return PredictionResponse(
            prediction=int(prediction[0]),
            is_fraud=bool(prediction[0] == 1),
            trans_num=transaction.trans_num,
            amt=transaction.amt,
            category=transaction.category,        
        )
        
    except Exception as e:
        print(f"❌ ERREUR DÉTAILLÉE:")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

#============================ MLflow Endpoints ============================#    

  

# class ExperimentInfo(BaseModel):
#     experiment_id: str
#     name: str
#     artifact_location: str
#     lifecycle_stage: str
#     total_runs: int
#     last_update: Optional[str] = None

# class MLflowTestResponse(BaseModel):
#     status: str
#     tracking_uri: str
#     connection_success: bool
#     total_experiments: int
#     experiments: List[ExperimentInfo]
#     error: Optional[str] = None




# @app.get("/mlflow-test", response_model=MLflowTestResponse)
# async def test_mlflow_connection():
#     """
#     Teste la connexion à MLflow et liste toutes les expériences disponibles
#     """
#     tracking_uri = mlflow.get_tracking_uri()
    
#     try:
#         client = MlflowClient()
        
#         # Récupérer toutes les expériences
#         experiments = client.search_experiments()
        
#         # Construire la liste des expériences avec détails
#         experiment_list = []
#         for exp in experiments:
#             # Récupérer les runs de l'expérience
#             runs = client.search_runs(
#                 experiment_ids=[exp.experiment_id],
#                 max_results=1000
#             )
            
#             # Trouver la dernière mise à jour
#             last_update = None
#             if runs:
#                 last_update = max(run.info.start_time for run in runs)
#                 last_update = pd.to_datetime(last_update, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            
#             experiment_list.append(
#                 ExperimentInfo(
#                     experiment_id=exp.experiment_id,
#                     name=exp.name,
#                     artifact_location=exp.artifact_location,
#                     lifecycle_stage=exp.lifecycle_stage,
#                     total_runs=len(runs),
#                     last_update=last_update
#                 )
#             )
        
#         return MLflowTestResponse(
#             status="connected",
#             tracking_uri=tracking_uri,
#             connection_success=True,
#             total_experiments=len(experiments),
#             experiments=experiment_list
#         )
        
#     except Exception as e:
#         return MLflowTestResponse(
#             status="error",
#             tracking_uri=tracking_uri,
#             connection_success=False,
#             total_experiments=0,
#             experiments=[],
#             error=str(e)
#         )

# @app.get("/mlflow-experiments")
# async def list_experiments(
#     active_only: bool = True,
#     limit: Optional[int] = None
# ):
#     """
#     Liste toutes les expériences MLflow avec option de filtrage
    
#     - **active_only**: Afficher uniquement les expériences actives (non supprimées)
#     - **limit**: Nombre maximum d'expériences à retourner
#     """
#     try:
#         client = MlflowClient()
        
#         # Filtrer les expériences
#         filter_string = "lifecycle_stage = 'active'" if active_only else None
#         experiments = client.search_experiments(filter_string=filter_string)
        
#         # Limiter le nombre de résultats
#         if limit:
#             experiments = experiments[:limit]
        
#         # Formater les résultats
#         results = []
#         for exp in experiments:
#             runs = client.search_runs(experiment_ids=[exp.experiment_id])
#             results.append({
#                 "experiment_id": exp.experiment_id,
#                 "name": exp.name,
#                 "total_runs": len(runs),
#                 "lifecycle_stage": exp.lifecycle_stage,
#                 "artifact_location": exp.artifact_location
#             })
        
#         return {
#             "status": "success",
#             "total_experiments": len(results),
#             "experiments": results
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching experiments: {str(e)}")

# @app.get("/mlflow-experiment/{experiment_id}")
# async def get_experiment_detail(
#     experiment_id: str,
#     max_runs: int = 10
# ):
#     """
#     Récupère les détails d'une expérience spécifique avec ses runs
    
#     - **experiment_id**: ID de l'expérience
#     - **max_runs**: Nombre maximum de runs à retourner
#     """
#     try:
#         client = MlflowClient()
        
#         # Récupérer l'expérience
#         experiment = client.get_experiment(experiment_id)
        
#         if not experiment:
#             raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
#         # Récupérer les runs
#         runs = client.search_runs(
#             experiment_ids=[experiment_id],
#             max_results=max_runs,
#             order_by=["start_time DESC"]
#         )
        
#         # Formater les runs
#         runs_data = []
#         for run in runs:
#             runs_data.append({
#                 "run_id": run.info.run_id,
#                 "run_name": run.data.tags.get("mlflow.runName", "N/A"),
#                 "status": run.info.status,
#                 "start_time": pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
#                 "metrics": run.data.metrics,
#                 "params": run.data.params,
#                 "tags": run.data.tags
#             })
        
#         return {
#             "status": "success",
#             "experiment": {
#                 "experiment_id": experiment.experiment_id,
#                 "name": experiment.name,
#                 "artifact_location": experiment.artifact_location,
#                 "lifecycle_stage": experiment.lifecycle_stage,
#                 "total_runs": len(client.search_runs(experiment_ids=[experiment_id]))
#             },
#             "runs": runs_data,
#             "returned_runs": len(runs_data)
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching experiment details: {str(e)}")