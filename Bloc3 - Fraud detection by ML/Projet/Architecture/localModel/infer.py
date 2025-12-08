from dotenv import load_dotenv
load_dotenv()


import pandas as pd
import requests
import json
import mlflow
from sqlalchemy import create_engine
import os
import boto3
from datetime import datetime


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
        45file_name = f'bloc3/raw_data/transaction_{timestamp}.json'
        
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

def transform(donnees, s3_client, bucket_name):
    donnees_transaction = donnees['data']
    columns = donnees['columns']  
    # Prepare data to stick input format of model 
    df = pd.DataFrame(donnees_transaction, columns=columns)
    df['unix_time'] = df['current_time']/1000
    df['trans_date_trans_time'] = pd.to_datetime(df['current_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    df.drop(columns=['current_time'], inplace=True)
    df = df.astype({col: "float64" for col in df.select_dtypes(include=["int"]).columns})
    data = df.drop(columns=['is_fraud'])

    print("TRANSFORM - making prediction...")
    # Predict on a Pandas DataFrame.
    prediction = loaded_model.predict(pd.DataFrame(data))
    if (prediction[0] == 1):
        print("************ Fraud detected! *************")

    # Add prediction to DataFrame
    df['prediction'] = prediction

    # Save transaction with prediction to csv file    
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'bloc3/transformed_data/transaction_{timestamp}.csv'
    
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


# ############################################### #
#              LOAD FUNCTION                      #
# save processed data with prediction to database #
# ############################################### #

def load(df, engine):
    try:
        # Écrire tout le DataFrame
        df.to_sql('transactions', engine, if_exists='append', index=False)
        print(f"Données enregistrées avec succès sur {engine.url.database}.transactions")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement en base de données : {str(e)}")


# ############### #
#   MAIN SCRIPT   #
# ############### #

if __name__ == "__main__":

    # Set tracking URI 
    mlflow.set_tracking_uri("http://localhost:4000/")

    # Set model informations
    logged_model = 'runs:/27eaae4358774670acb197df4942a5bd/LBPFraudDetector'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Set database connection string
    # Connection string pour Neon
    connection_string = os.getenv("BACKEND_STORE_URI")
    engine = create_engine(connection_string)

    # Set API URL
    API_URL = "https://sdacelo-real-time-fraud-detection.hf.space/current-transactions"

    # Initialiser le client S3
    s3_client = boto3.client('s3')
    # Définir le nom du bucket
    bucket_name = 'bucket-laposte-david'
    test = 0
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




        