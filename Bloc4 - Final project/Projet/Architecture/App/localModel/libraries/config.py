
import os

import sys
import pandas as pd

debug = False  # True for debug mode (smaller dataset, configure size below in : samplingSize var), False for full dataset
plottingEDA = True # True to enable EDA plotting, False to disable
mlFlowLocal = False  # True for local mlflow, False for hosted mlflow

#============================================================================= 

samplingSize = 20000  # Number of rows to sample in debug mode, minimum 1000

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, "..", "..")) + "/"
sys.path.append(os.path.join(project_path, 'lib'))

neonDB_connectionURL = os.getenv('NEON_DB_URL', '')
neonDB_fraudTableName = "neondb"
HF_connectionURL = "https://huggingface.co/spaces/sdacelo/real-time-fraud-detection"
HF_connectionCSV = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
API_URL = "https://sdacelo-real-time-fraud-detection.hf.space/current-transactions"

local_connectionURL = os.path.abspath("./Bloc3 - Fraud detection by ML/Projet/datasSources/inputDataset/fraudTest.csv")  # absolute path
localDB_connectionURL = os.path.join(os.path.dirname(project_path), "datasSources", "inputDataset", "fraudTest.db")

localDB_tableName = "transactions"
inputDBFormat = "db"  # "csv" or "db" or neon or HF_CSV

separator = ("="*80)

#---- Name defined by user for project mlFlow ----

modelPrefix = "LBP_fraud_detector_"
EXPERIMENT_NAME = "LBPFraudDetector"
bucket_name = 'bucket-laposte-david'
aws_region = 'eu-north-1'

if mlFlowLocal == True:
    mlflow_tracking_uri = "http://localhost:4000/"
else:
    mlflow_tracking_uri = "https://davidrambeau-bloc3-mlflow.hf.space/"
    #mlflow_tracking_uri = "https://davidrambeau-bloc3-mlflow.hf.space/"
    


# https://jedhard.jfrog.io/artifactory/api/docker/jedha-docker
# jedhard.jfrog.io


# trackinguri = "https://gateway.storjshare.io"
# accessKeyId = "vodj6oqt5cnzzhqlw553kzfobba";
# secretAccessKey = "j3yfufqzxnay56hyo4qlp4bzg767zoikxfdkcj2zmbbgsqg6q6rf6";
# endpoint = "https://gateway.storjshare.io";
# bucketName = "lbpfrauddetector"; 



print("✅ Configuration load successful.")