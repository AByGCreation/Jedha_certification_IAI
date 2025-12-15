from flask import Flask, json, render_template, request, jsonify, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import mlflow
import requests
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from dotenv import load_dotenv , find_dotenv
import boto3

env_path = find_dotenv()
load_dotenv(env_path, override=True)

separator = ("="*80)

"""
Flask Transaction Simulator with Real-time Fraud Detection
Simulates credit card transactions and instantly detects fraud using MLFlow model
"""
# Global model variable
loaded_model = None
bucket_name = 'bucket-laposte-david'
s3_client = boto3.client('s3')


EXPERIMENT_NAME = "LBPFraudDetector"
# Configuration
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://davidrambeau-bloc3-mlflow.hf.space")
mlflow.set_tracking_uri(mlflow_tracking_uri)
        

BACKEND_STORE_URI = os.getenv("BACKEND_STORE_URI")
simulationAPI_URL = os.getenv("API_URL", "https://sdacelo-real-time-fraud-detection.hf.space/current-transactions")
fast_API_URL = os.getenv("FAST_API_URL", "https://davidrambeau-bloc3-fastapi.hf.space/")
#fast_API_URL = os.getenv("FAST_API_URL", "http://localhost:7860/")
MLFLOW_TRACKING_URI = mlflow_tracking_uri



app = Flask(__name__, template_folder='templates')
CORS(app)


def getMLFlowLastModel():
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



# Set MLFlow tracking


# Load model at startup
# loaded_model = None

# def load_fraud_model():
#     """Load the fraud detection model from MLFlow"""
#     global loaded_model
#     MODEL_URI = getMLFlowLastModel()

#     print(f"{separator}\n📦 Loading Fraud Detection Model\n{separator}")
#     print(f"Loading model from {MODEL_URI}...")
#     print(f"Experiment Name: {EXPERIMENT_NAME}")
#     print(f"MLFlow Tracking URI: {mlflow.get_tracking_uri()}")
#     print(f"Model Run ID: {MODEL_URI.latest_versions[0].run_id}")
   
#     print(separator)

#     try:
        

#         logged_model = f'runs:/{MODEL_URI.latest_versions[0].run_id}/{EXPERIMENT_NAME}'
#         print(f"Model URI: {logged_model}")
#         loaded_model = mlflow.pyfunc.load_model(logged_model)        
    
#         print(f"✅ Model loaded successfully from {MODEL_URI}")
#         return True
#     except Exception as e:
#         print(f"❌ Error loading model: {str(e)}")
#         return False


@app.route('/get_api_data', methods=['GET'])
def getTransactionFromAPI(API_URL=simulationAPI_URL, s3_client=s3_client, bucket_name=bucket_name):
    """Get transaction from API and return as flat dict"""
    
    r = requests.get(API_URL, timeout=30)

    if r.status_code == 200:
        # Double parsing car l'API encode 2 fois
        donnees = json.loads(r.text)
        donnees = json.loads(donnees)
        
        # Convertir le format DataFrame en dict plat
        if 'columns' in donnees and 'data' in donnees:
            columns = donnees['columns']
            data = donnees['data'][0]  # Première ligne
            transaction_dict = dict(zip(columns, data))
        else:
            transaction_dict = donnees
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'outputs/transaction_{timestamp}.json'
        
        json_data = json.dumps(transaction_dict, ensure_ascii=False, indent=2)
        
        # Upload to S3
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"✅ Données enregistrées sur S3: s3://{bucket_name}/{file_name}")
            
        except Exception as e:
            print(f"❌ Erreur S3: {str(e)}")
        
        print(f"✅ Transaction keys: {list(transaction_dict.keys())}")
        return transaction_dict
    
    else:
        print(f"❌ Erreur API Simulation: status code {r.status_code}")
        return None
    
def predict_fraud(transaction_dict):
    """Predict if transaction is fraud using FastAPI"""
    global loaded_model

  
    print(separator)
    print("🔍 Predicting fraud for transaction...")
    print(f"Transaction keys: {list(transaction_dict.keys())}")
     
    try:
        api_url = f"{fast_API_URL}predict"
        
        response = requests.post(
            api_url,
            json=transaction_dict,  # ✅ Déjà au bon format
            headers={'accept': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            print(f"✅ Prediction: {prediction}")
            return prediction, None
        else:
            error_detail = response.json() if response.status_code == 422 else response.text
            return None, f"API call failed [{response.status_code}]: {error_detail}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"
    
def save_transaction_to_db(transaction_dict, prediction):
    """Save transaction with prediction to database"""
    if not BACKEND_STORE_URI:
        return False, "Database not configured"

    try:
        engine = create_engine(BACKEND_STORE_URI)
        db_transaction = {}
        
        # Colonnes attendues
        db_columns = [
            'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 
            'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 
            'city_pop', 'job', 'dob', 'trans_num', 'merch_lat', 'merch_long', 
            'is_fraud'
        ]
        
        # Copier les colonnes existantes
        for col in db_columns:
            if col in transaction_dict:
                db_transaction[col] = transaction_dict[col]
        
     
        if 'current_time' in transaction_dict:
            timestamp_ms = transaction_dict['current_time']
            # Convertir millisecondes → secondes → datetime
            db_transaction['trans_date_trans_time'] = datetime.fromtimestamp(timestamp_ms / 1000)
        
        # Ajouter métadonnées
        db_transaction['prediction'] = prediction
        db_transaction['created_at'] = datetime.now()

        print(f"🔍 DB Transaction keys: {list(db_transaction.keys())}")
        print(f"🔍 trans_date_trans_time type: {type(db_transaction.get('trans_date_trans_time'))}")
        print(f"🔍 trans_date_trans_time value: {db_transaction.get('trans_date_trans_time')}")

        # Convert to DataFrame and save
        df = pd.DataFrame([db_transaction])
        df.to_sql('currentTransaction', engine, if_exists='append', index=False)

        print(f"✅ Transaction saved to database")
        return True, None
        
    except Exception as e:
        if 'UniqueViolation' in str(e) or 'duplicate key' in str(e):
            print(f"ℹ️ Transaction already exists - skipping")
            return True, "Already exists (skipped)"        
        import traceback
        traceback.print_exc()
        return False, str(e)

@app.route('/health')
def health():
    """Health check endpoint"""
    #model = getMLFlowLastModel()
    return jsonify({
        'status': 'healthy',
        # 'model_loaded': model is not None,
        # 'model_name': model.name if model else None,
        # 'model_version': model.latest_versions[0].version if model else None,
        # 'model_runid': model.latest_versions[0].run_id if model else None,
        # 'model_ExperimentName': EXPERIMENT_NAME,
        # 'model_tags': model.tags if model else None,
        # 'mlflow_uri': MLFLOW_TRACKING_URI
    })

@app.route('/simulate', methods=['POST'])
def simulate_transaction():
    """Simulate a transaction and get fraud prediction"""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        #force_fraud = data.get('force_fraud', False)

        # Generate transaction
        transaction = getTransactionFromAPI()#generate_random_transaction(is_fraud=force_fraud)
        if transaction is None:
            return jsonify({
                'success': False,
                'error': 'Failed to get transaction from API',
                'error_code': 'API_001'
            }), 500
        
        # Predict fraud
        prediction, error = predict_fraud(transaction)

        if error:
            return jsonify({
                'success': False,
                'error': f"Prediction error: {error}",
                'error_code': 'PRED_001'
            }), 500

        # Save to database
        db_success = False
        db_error = None
        try:
            db_success, db_error = save_transaction_to_db(transaction, prediction)
        except Exception as db_exception:
            db_error = str(db_exception)
            print(f"⚠️ DB Error (non-blocking): {db_error}")

        transaction_display = {
            'id': transaction['trans_num'],
            'amount': transaction['amt'],
            'category': transaction['category'],
            'cc_num': transaction['cc_num'],
            'merchant': transaction.get('merchant', 'N/A'),
            'city': transaction.get('city', 'N/A'),
            'state': transaction.get('state', 'N/A'),
            'date': str(transaction.get('current_time', 'N/A')),
            'predicted_fraud': prediction,
            'is_fraud': bool(prediction == 1)
        }

        response = {
            'success': True,
            'count': 0,
            'accuracy': 0,
            'transaction': transaction_display,
            'transactions': transaction_display,
            'prediction': {
                'is_fraud': bool(prediction == 1),
                'confidence': 'high' if prediction == 1 else 'low',
                'actual_fraud': transaction.get('is_fraud', 0)
            },
            'saved_to_db': db_success
        }

        print(f"✅ Simulation response: {response}")

        if db_error:
            response['db_warning'] = f"Database save failed: {db_error}"
            print(f"⚠️ DB Warning: {db_error}")
        print(f"✅ Simulation response: Success={response['success']}, Prediction={prediction}")

        return jsonify(response), 200 

    except KeyError as e:
        print(f"❌ Missing key: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Missing field: {str(e)}',
            'error_code': 'KEY_001'
        }), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'GEN_001'
        }), 500


@app.route('/')
def index():
    """Main page"""
    #global loaded_model
    #load_fraud_model()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860, debug=True)


