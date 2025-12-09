from dotenv import load_dotenv, find_dotenv
import pandas as pd
import requests
import json
import mlflow
from sqlalchemy import create_engine
import os
import boto3
from datetime import datetime
import sys

env_path = find_dotenv()
load_dotenv(env_path, override=True)

#current_path = os.path.dirname(os.path.abspath(__file__))
#project_path = os.path.abspath(os.path.join(current_path, "..", "..")) + "/"
#sys.path.append(os.path.join(current_path, 'libraries'))
current_path = os.path.dirname(os.path.abspath(__file__))
# Define current_path FIRST

sys.path.insert(0, os.path.join(current_path, 'libraries'))

# NOW use current_path
project_path = os.path.abspath(os.path.join(current_path, "..", "..")) + "/"


from config import API_URL as API_URL, bucket_name as bucket_name, EXPERIMENT_NAME as EXPERIMENT_NAME, HF_connectionCSV as HF_connectionCSV, HF_connectionURL as HF_connectionURL, current_path as current_path, debug as debug, inputDBFormat as inputDBFormat, localDB_connectionURL as localDB_connectionURL, localDB_tableName as localDB_tableName, local_connectionURL as local_connectionURL, mlFlowLocal as mlFlowLocal, mlflow_tracking_uri as mlflow_tracking_uri, modelPrefix as modelPrefix, neonDB_connectionURL as neonDB_connectionURL, neonDB_fraudTableName as neonDB_fraudTableName, os as os, pd as pd, plottingEDA as plottingEDA, project_path as project_path, samplingSize as samplingSize, separator as separator, sys as sys
from plotters import MarkerCluster as MarkerCluster, drawCorrelationMatrix as drawCorrelationMatrix, folium as folium, go as go, jedhaCMInverted as jedhaCMInverted, jedhaColor_black as jedhaColor_black, jedhaColor_blue as jedhaColor_blue, jedhaColor_violet as jedhaColor_violet, jedha_bg_color as jedha_bg_color, jedha_font as jedha_font, jedha_grid_color as jedha_grid_color, pd as pd, plotFeatureDistributions as plotFeatureDistributions, plt as plt, saveMap as saveMap, sns as sns
from graphics import colors as colors, go as go, jedhaCM as jedhaCM, jedhaCMInverted as jedhaCMInverted, jedhaColor_black as jedhaColor_black, jedhaColor_blue as jedhaColor_blue, jedhaColor_blueLight as jedhaColor_blueLight, jedhaColor_violet as jedhaColor_violet, jedhaColor_white as jedhaColor_white, jedha_bg_color as jedha_bg_color, jedha_colors as jedha_colors, jedha_font as jedha_font, jedha_grid_color as jedha_grid_color, np as np, pio as pio, platform as platform 
from dataLoader import cfg as cfg, create_engine as create_engine, dataSourceLoader as dataSourceLoader, pd as pd, sqlite3 as sqlite3
from converters import haversine as haversine
from preprocessor import Preprocessor as Preprocessor
# ######################################### #
# EXTRACT FUNCTION
# connect API to get real-time transactions #
# save raw data to S3 bucket in json        #
# ######################################### #

def extract(API_URL, s3_client, bucket_name):
    r = requests.get(API_URL)
    if r.status_code == 200:
        # Double parsing car l'API encode 2 fois
        donnees = json.loads(r.text)
        donnees = json.loads(donnees)
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        file_name = f'outputs/transaction_{timestamp}.json'
        
        # Convert data to JSON string
        json_data = json.dumps(donnees, ensure_ascii=False, indent=2)
        
        # Upload to S3
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"Données enregistrées avec succès sur S3: s3://{bucket_name}/{file_name}")
            return donnees
        except Exception as e:
            print(f"Erreur lors de l'enregistrement sur S3: {str(e)}")
            return None
    else:
        print(f"Erreur API: status code {r.status_code}")



# ############################################ #
#            TRANSFORM FUNCTION                #
# process data to fit model input format       #
# save transformed data with prediction to S3  #
# ############################################ #

def transform(datas, s3_client, bucket_name):

    # Prepare data to stick input format of model 
    df = pd.DataFrame(datas['data'], columns=datas['columns']  )

    print("✅ Preprocessing in progress !")
    # Predict on a Pandas DataFrame.
    df_processed = Preprocessor(df)
    data = df_processed.drop(columns=['is_fraud'])

    prediction = loaded_model.predict(data)
    
    df['prediction'] = prediction

    # Save transaction with prediction to csv file    
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'outputs/transaction_{timestamp}.csv'
    
    # Convert data to CSV string
    csv_data = df.to_csv(index=False)
    
    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=csv_data,
            ContentType='application/csv'
        )
        print(f"Données enregistrées avec succès sur S3: s3://{bucket_name}/{file_name}")
        return df
    except Exception as e:
        print(f"Erreur lors de l'enregistrement sur S3: {str(e)}")



    # Alert system

    if (prediction[0] == 1):
        print("⚠️ ⚠️ ⚠️ Fraud detected! ⚠️ ⚠️ ⚠️")
    else:
        print("✅ Transaction is legitimate.")

    # Add prediction to DataFrame
    



def load(df, engine):
    try:
        # Écrire tout le DataFrame
        df.to_sql('transactions', engine, if_exists='append', index=False)
        print(f"Données enregistrées avec succès sur {engine.url.database}.transactions")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement en base de données : {str(e)}")


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
            alias="staging",
            version=model_version,
        )
        print(f"[INFO] Alias 'staging' now points to version {model_version}")
        return best_model.latest_versions[0].run_id
    else:
        print("[WARN] Aucun modèle trouvé dans le registre.")


if __name__ == "__main__":

    # Set tracking URI 
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    runID = getModelRunID()
    print(f"Latest model version: {runID}")


    # Set model informations
    logged_model = f'runs:/{runID}/{EXPERIMENT_NAME}'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Set database connection string
    # Connection string pour Neon
    connection_string = os.getenv("BACKEND_STORE_URI")
    engine = create_engine(connection_string)

    # Initialiser le client S3
    s3_client = boto3.client('s3')
    # Définir le nom du bucket

    test = 9
    while test <10:
        print("================================")
        print("EXTRACT - calling API to generate transaction...")

        donnees = extract(API_URL, s3_client, bucket_name)

        if donnees:
            print("TRANSFORM - processing transaction data...")
            df = transform(donnees, s3_client, bucket_name)

            # Save transaction with prediction to database
            print("LOAD - saving transaction to database...")
            load(df, engine)
            wait_time = 10
        else :
            print(f"An error occurred")
            wait_time = 30
    
        print(f"waiting for {wait_time} seconds before next transaction...\n")
        import time
        time.sleep(wait_time)
        
        test += 1




        