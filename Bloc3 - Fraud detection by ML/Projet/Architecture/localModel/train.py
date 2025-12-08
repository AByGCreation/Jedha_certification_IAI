#=====================================================================
#=============================== IMPORTS ================================
#=====================================================================


# ==================== DATA MANIPULATION ====================
import argparse
from math import radians, cos, sin, asin, sqrt
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# ==================== VISUALIZATION ====================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import folium
from folium.plugins import MarkerCluster

# ==================== MACHINE LEARNING ====================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ==================== MLFLOW ====================
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ==================== UTILITIES ====================
import os
import warnings
import time
import datetime
from dotenv import find_dotenv, load_dotenv
import platform
import sys

env_path = find_dotenv()
load_dotenv(env_path, override=True)

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, "..", "..")) + "/"
sys.path.append(os.path.join(current_path, 'libraries'))

from config import EXPERIMENT_NAME as EXPERIMENT_NAME, HF_connectionCSV as HF_connectionCSV, HF_connectionURL as HF_connectionURL, current_path as current_path, debug as debug, inputDBFormat as inputDBFormat, localDB_connectionURL as localDB_connectionURL, localDB_tableName as localDB_tableName, local_connectionURL as local_connectionURL, mlFlowLocal as mlFlowLocal, mlflow_tracking_uri as mlflow_tracking_uri, modelPrefix as modelPrefix, neonDB_connectionURL as neonDB_connectionURL, neonDB_fraudTableName as neonDB_fraudTableName, os as os, pd as pd, plottingEDA as plottingEDA, project_path as project_path, samplingSize as samplingSize, separator as separator, sys as sys
from plotters import MarkerCluster as MarkerCluster, datetime as datetime, drawCorrelationMatrix as drawCorrelationMatrix, folium as folium, go as go, jedhaCMInverted as jedhaCMInverted, jedhaColor_black as jedhaColor_black, jedhaColor_blue as jedhaColor_blue, jedhaColor_violet as jedhaColor_violet, jedha_bg_color as jedha_bg_color, jedha_font as jedha_font, jedha_grid_color as jedha_grid_color, pd as pd, plotFeatureDistributions as plotFeatureDistributions, plt as plt, saveMap as saveMap, sns as sns
from graphics import colors as colors, go as go, jedhaCM as jedhaCM, jedhaCMInverted as jedhaCMInverted, jedhaColor_black as jedhaColor_black, jedhaColor_blue as jedhaColor_blue, jedhaColor_blueLight as jedhaColor_blueLight, jedhaColor_violet as jedhaColor_violet, jedhaColor_white as jedhaColor_white, jedha_bg_color as jedha_bg_color, jedha_colors as jedha_colors, jedha_font as jedha_font, jedha_grid_color as jedha_grid_color, np as np, pio as pio, platform as platform 
from dataLoader import cfg as cfg, create_engine as create_engine, dataSourceLoader as dataSourceLoader, pd as pd, sqlite3 as sqlite3

# ==================== SETTINGS ====================
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("✅ All libraries imported successfully!")

#============================================================================= 
#=============================== INITIALISATION ==============================
#============================================================================= 

#============================================================================= 
#============================== User PARAMETERS ================================
#============================================================================= 
dfRaw = pd.DataFrame()

# debug = True  # True for debug mode (smaller dataset, configure size below in : samplingSize var), False for full dataset
# plottingEDA = True # True to enable EDA plotting, False to disable
# mlFlowLocal = False  # True for local mlflow, False for hosted mlflow

# #============================================================================= 

# samplingSize = 2000  # Number of rows to sample in debug mode, minimum 1000



# neonDB_connectionURL = os.getenv('NEON_DB_URL', 'postgresql://neondb_owner:npg_UIrY18vhNmLE@ep-curly-sound-ag9a7x4l-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')
# neonDB_fraudTableName = "neondb"
# HF_connectionURL = "https://huggingface.co/spaces/sdacelo/real-time-fraud-detection"
# HF_connectionCSV = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
# local_connectionURL = os.path.abspath("./Bloc3 - Fraud detection by ML/Projet/datasSources/inputDataset/fraudTest.csv")  # absolute path
# localDB_connectionURL = os.path.join(os.path.dirname(project_path), "datasSources", "inputDataset", "fraudTest.db")

# localDB_tableName = "transactions"
# inputDBFormat = "db"  # "csv" or "db" or neon or HF_CSV

# dfRaw = pd.DataFrame()

# separator = ("="*80)

# #---- Name defined by user for project mlFlow ----

# modelPrefix = "LBP_fraud_detector_"
EXPERIMENT_NAME = "LBPFraudDetector"

# if mlFlowLocal == True:
#     mlflow_tracking_uri = "http://localhost:4000/"
# else:
#     mlflow_tracking_uri = "https://davidrambeau-bloc3-mlflow.hf.space/"


# https://jedhard.jfrog.io/artifactory/api/docker/jedha-docker
# jedhard.jfrog.io


# trackinguri = "https://gateway.storjshare.io"
# accessKeyId = "vodj6oqt5cnzzhqlw553kzfobba";
# secretAccessKey = "j3yfufqzxnay56hyo4qlp4bzg767zoikxfdkcj2zmbbgsqg6q6rf6";
# endpoint = "https://gateway.storjshare.io";
# bucketName = "lbpfrauddetector"; 


print("✅ Configuration load successful.")

#=====================================================================
#=============================== FUNCTIONS =============================
#=====================================================================

def logArrayToClipboard(array, array_name="Array"):
    """Log a DataFrame or Series statistics to clipboard.   
    Args:
        array (_type_): _description_
        array_name (str, optional): _description_. Defaults to "Array".
    """
    
    # Export basic statistics to clipboard (Excel/Word friendly)
    data_desc_rounded = array.round(2)
    data_desc_rounded.to_clipboard(excel=True)
    print(f"✅ {array_name} copied to clipboard.")


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
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

def dastaSourceLoader(inputDBFormat: str) -> pd.DataFrame|bool:
    """Load data from specified source format into a DataFrame.

    Args:
        inputDBFormat (str): The format of the data source ("csv", "db", "neon", "HF").

    Returns:
        pd.DataFrame: Loaded DataFrame or False on error.
    """
    dfRaw = pd.DataFrame()
    conn = None
    engine = None

    try:
        if inputDBFormat == "csv":
            dfRaw = pd.read_csv(local_connectionURL)
        elif inputDBFormat == "db":
            conn = sqlite3.connect(localDB_connectionURL)
            query = f"SELECT * FROM {localDB_tableName}"
            dfRaw = pd.read_sql_query(query, conn)
        elif inputDBFormat == "neon":
            engine = create_engine(neonDB_connectionURL)
            query = f"SELECT * FROM {neonDB_fraudTableName}"
            dfRaw = pd.read_sql_query(query, engine)
        elif inputDBFormat == "HF_CSV":
            dfRaw = pd.read_csv(HF_connectionCSV)
        else:
            print("❌ Invalid inputDBFormat. Please choose 'csv', 'db', 'neon', or 'HF_CSV'.")
            return False
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False
    finally:
        # Ensure connections are properly closed
        if conn is not None:
            conn.close()
        if engine is not None:
            engine.dispose()

    dfRaw = dfRaw.astype({col: "float64" for col in dfRaw.select_dtypes(include=["int"]).columns})

    return dfRaw


#=============================== Preprocessing ===========================

def Preprocessor(df : pd.DataFrame) -> pd.DataFrame:
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
    df.drop(columns=["Column1"], inplace=True)





    df['ccn_len'] = df['cc_num'].astype(str).str.len()
    df['bin'] = pd.to_numeric(df['cc_num'].astype(str).str[:6], errors='coerce')

    # Calculate the distance between customer and merchant locations (vectorized)
    lon1 = np.radians(df['long'].values)
    lat1 = np.radians(df['lat'].values)
    lon2 = np.radians(df['merch_long'].values)
    lat2 = np.radians(df['merch_lat'].values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['distance_km'] = c * 6371  # Earth radius in kilometers

    print(f"Distance calculated. Min: {df['distance_km'].min():.2f} km, "
        f"Max: {df['distance_km'].max():.2f} km, "
        f"Mean: {df['distance_km'].mean():.2f} km")

    # Convert dob to datetime and calculate age
    if not pd.api.types.is_datetime64_any_dtype(df['dob']):
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    df['age'] = ((pd.Timestamp.now() - df['dob']).dt.days // 365).astype('float32')
    df = df.sort_values(by='age', ascending=True)

    # Convert amt to numeric
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce').astype('float32')
    
    # Extract hour from transaction datetime
    df['trans_hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour.astype('float32')

    # Drop columns that are no longer needed (only if they exist)
   
    columns_to_drop = [
        'dob', 'trans_date_trans_time', 'unix_time', 'merchant', 'gender', 'state',
        'lat', 'long', 'merch_lat', 'merch_long', 'city', 'zip', 'city_pop', 'job', 'bin',
        'street', 'first', 'last', 'Column1', 'trans_num', "unamed: 0"
    ]
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr, annot=True, fmt=".2f", cmap=jedhaCMInverted, square=True)
    # plt.title("Matrice de corrélation des variables numériques")
    # #plt.show()
    # plt.savefig(current_path + '/outputs/Analysis_correlationMatrix_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
    # plt.close()

    drawCorrelationMatrix(df.drop(columns=["is_fraud"], inplace=False), title_suffix="_before_preprocessing", current_path=current_path)

    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
        print(f"Dropped columns: {existing_columns_to_drop}")
    else:
        print("No columns to drop (already removed or not present).")

    drawCorrelationMatrix(df.drop(columns=["is_fraud"], inplace=False), title_suffix="_after_preprocessing", current_path=current_path)

    return df

print("✅ UDF functions loaded successfully.")

#=====================================================================
#=============================== MAIN PROGRAM ===========================
#=====================================================================

if __name__ == "__main__":

    dfRaw = dataSourceLoader(inputDBFormat)
    print("✅ Dataframe successfully created from " + inputDBFormat + " format with " + str(len(dfRaw)) + " rows and " + str(len(dfRaw.columns)) + " columns.")

    #==EDA Prepa# Select dataset based on debug mode
    if debug:
        # Use stratified sampling to ensure both fraud and non-fraud cases are included
        fraud_cases = dfRaw[dfRaw['is_fraud'] == 1]
        non_fraud_cases = dfRaw[dfRaw['is_fraud'] == 0]

        # Calculate how many fraud cases to include (maintain approximate original ratio)
        fraud_ratio = len(fraud_cases) / len(dfRaw)
        n_fraud_samples = max(1, int(samplingSize * fraud_ratio))  # At least 1 fraud case
        n_non_fraud_samples = samplingSize - n_fraud_samples

        # Sample from each class
        fraud_sample = fraud_cases.sample(n=min(n_fraud_samples, len(fraud_cases)), random_state=42)
        non_fraud_sample = non_fraud_cases.sample(n=min(n_non_fraud_samples, len(non_fraud_cases)), random_state=42)

        # Combine and shuffle
        df = pd.concat([fraud_sample, non_fraud_sample], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Using debug mode with {len(df)} rows and {len(df.columns)} columns (Fraud: {len(fraud_sample)}, Non-fraud: {len(non_fraud_sample)})")
    else:
        df = dfRaw
        print(f"Using full mode with {len(df)} rows and {len(df.columns)} columns .")

    print()

    if debug:
        # Display initial dataframe info
        print(separator)
        print("INITIAL DATAFRAME")
        print(separator)
        print(f"Shape: {df.shape}")
        print()
        print("Columns in df:")
        print(df.columns.to_list())

    print("✅ Data preparation successfully completed.")

    #=== EDA Transform and optimize data  ===#
    # Data Type Transformation and Optimization
    print(separator)
    print("TRANSFORMING DATAFRAME WITH OPTIMIZED DATA TYPES")
    print(separator)
    print()

    # Store original memory usage
    original_memory = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Original memory usage: {original_memory:.2f} MB")
    print()

    # Create a copy to transform
    dfOptimized = df.copy()

    # STEP 1: Handle datetime columns FIRST
    print("Converting datetime columns...")

    # Define which columns are datetime columns
    datetime_columns = ['dob', 'trans_date_trans_time']

    datetimeConverter(df, ['dob'])
    datetimeConverter(df,  ['trans_date_trans_time'])

    print()

    # STEP 2: Apply other data type conversions
    type_conversions = {
        # Transaction identifiers
        'cc_num': 'int64',     # Credit card number as integer
        
        # Categorical columns with limited unique values
        'merchant': 'category',
        'category': 'category',
        'job': 'category',
        'gender': 'category',
        'city': 'category',
        'state': 'category',
        
        # Numeric columns - optimize size
        'amt': 'float32',  # Transaction amount
        'zip': 'int32',    # ZIP code
        'city_pop': 'int32',  # Population
        'unix_time': 'int64',  # Unix timestamp
        
        # Geographic coordinates
        'lat': 'float32',
        'long': 'float32',
        'merch_lat': 'float32',
        'merch_long': 'float32',
        
        # Boolean/Binary flags
        'is_fraud': 'int8',  # 0 or 1
    }

    print("Applying data type conversions...")
    print()

    # Apply conversions (skip datetime columns)
    for col, new_dtype in type_conversions.items():
        if col not in dfOptimized.columns:
            continue
            
        # Skip datetime columns
        if col in datetime_columns:
            continue
        
        try:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(dfOptimized[col]):
                print(f"⊘ {col}: Skipping (is datetime)")
                continue
            
            old_dtype = dfOptimized[col].dtype
            
            if new_dtype == 'category':
                dfOptimized[col] = dfOptimized[col].astype('category')
            elif new_dtype in ['float32', 'float64']:
                dfOptimized[col] = pd.to_numeric(dfOptimized[col], errors='coerce').astype(np.dtype(new_dtype))
            elif new_dtype in ['int8', 'int16', 'int32', 'int64']:
                # For integer conversions, fill NaN with 0 first
                dfOptimized[col] = pd.to_numeric(dfOptimized[col], errors='coerce').fillna(0).astype(np.dtype(new_dtype))
            else:
                dfOptimized[col] = dfOptimized[col].astype(np.dtype(new_dtype))
            
            print(f"✅  {col}: {old_dtype} → {new_dtype}")
        except Exception as e:
            print(f"✗ {col}: Failed - {str(e)[:80]}")
    
    # Clean up any remaining non-numeric values in numeric columns
    # for col in dfOptimized.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns:
    #     dfOptimized[col] = pd.to_numeric(dfOptimized[col], errors='coerce')
    #     if dfOptimized[col].dtype in ['float32', 'float64']:
    #         dfOptimized[col] = dfOptimized[col].fillna(dfOptimized[col].median())
    #     else:
    #         dfOptimized[col] = dfOptimized[col].fillna(0)

    print()
    print(separator)
    print("TRANSFORMATION COMPLETE")
    print(separator)
    print()

    # Calculate new memory usage
    optimized_memory = dfOptimized.memory_usage(deep=True).sum() / (1024**2)
    memory_saved = original_memory - optimized_memory
    memory_saved_pct = (memory_saved / original_memory) * 100 if original_memory > 0 else 0

    print(f"Original memory usage:  {original_memory:.2f} MB")
    print(f"Optimized memory usage: {optimized_memory:.2f} MB")
    print(f"Memory saved:           {memory_saved:.2f} MB ({memory_saved_pct:.1f}%)")
    print()

    # Display new data types
    print(separator)
    print("NEW DATA TYPES:")
    print(separator)
    print(dfOptimized.dtypes)
    print()

    # Verify data integrity
    print(separator)
    print("DATA INTEGRITY CHECK:")
    print(separator)
    print(f"Original shape:  {df.shape}")
    print(f"Optimized shape: {dfOptimized.shape}")
    print(f"Fraud count (original):  {df['is_fraud'].sum()}")
    print(f"Fraud count (optimized): {dfOptimized['is_fraud'].sum()}")
    print()

    # Replace the original dataframe
    df = dfOptimized

    print(df.head())

    print("✅ Dataframe successfully optimized and updated!")

    #======================================================================
    ## EDA - Preprocessing functions ##
    #======================================================================
    
    print(separator)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print(separator)
    print()






    # ============================================================================
    # STEP 1: Prepare Data for Modeling
    # ============================================================================


    # X, y split 
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Apply preprocessing
    X_train_after_dataset_processing = Preprocessor(df)


    plotFeatureDistributions(X_train_after_dataset_processing, current_path)



    categorical_features = X_train_after_dataset_processing.select_dtypes("object").columns # Select all the columns containing strings
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='error')

    numerical_feature_mask = ~X_train_after_dataset_processing.columns.isin(X_train_after_dataset_processing.select_dtypes("object").columns) # Select all the columns containing anything else than strings
    numerical_features = X_train_after_dataset_processing.columns[numerical_feature_mask]
    numerical_transformer = StandardScaler()

    feature_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_transformer", categorical_transformer, categorical_features),
            ("numerical_transformer", numerical_transformer, numerical_features)
        ]
    )

    model = Pipeline(steps=[
        #("Dates_preprocessing", date_preprocessor),
        ('features_preprocessing', feature_preprocessor),
        ("Regressor",RandomForestClassifier(n_estimators=100, min_samples_split=2))
    ])


    print("✅ Final cleanup before EDA complete.")

    print("Separating labels from features...")

    # Separate target variable Y from features X
    X = df.drop(columns=["is_fraud"], inplace=False)
    Y = df["is_fraud"]

    print(f"Y shape: {Y.shape}")
    print(f"X shape: {X.shape}")
    print(f"Unique classes in Y: {Y.unique()}")
    print()



    #======================================================================
    ##  Model training and evaluation ##
    #======================================================================

    print(separator)
    print("MODEL TRAINING AND EVALUATION")
    print(separator)
    print()


    # Check if we have sufficient samples for stratified split
    if len(Y.unique()) < 2:
        print("⚠️ ERROR: Only one class present in target variable. Cannot train models.")
        print(f"Please increase samplingSize to ensure at least 20 fraud cases.")
    else:
        min_samples_per_class = Y.value_counts().min()
        
        if min_samples_per_class < 2:
            print(f"⚠️ ERROR: Insufficient samples for stratified split.")
            print(f"Minimum samples per class: {min_samples_per_class}")
        else:
            print("✅  Sufficient samples for stratified split")
            print(f"Fraud cases: {Y.sum()}, Non-fraud cases: {len(Y) - Y.sum()}")
            print()
            
            # ====================================================================
            # STEP 2: Train/Test Split
            # ====================================================================
            
            print("Splitting data into training and testing sets...")
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, 
                test_size=0.2, 
                random_state=42, 
                stratify=Y
            )
            
            print(separator)
            print(X_test.info())
            print(separator)


            print(f"Training set size: {X_train.shape[0]} (Fraud: {Y_train.sum()}, Non-fraud: {len(Y_train) - Y_train.sum()})")
            print(f"Test set size: {X_test.shape[0]} (Fraud: {Y_test.sum()}, Non-fraud: {len(Y_test) - Y_test.sum()})")
            print()
            
            # ====================================================================
            # STEP 3: Prepare Features for Modeling
            # ====================================================================
            
            print("Encoding features for machine learning...")

            # Prepare features
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()

            # Convert datetime to numeric if exists (days since epoch)
            # if 'dob' in X_train_processed.columns:
            #     if pd.api.types.is_datetime64_any_dtype(X_train_processed['dob']):
            #         X_train_processed['dob'] = (X_train_processed['dob'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
            #         X_test_processed['dob'] = (X_test_processed['dob'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
            
            # Encode categorical features with handling for unseen labels
            label_encoders = {}
            unknown_counts = {}
            
            for col in X_train_processed.columns:
                if X_train_processed[col].dtype == 'category' or X_train_processed[col].dtype == 'object':
                    # Fit encoder on training data
                    label_encoders[col] = LabelEncoder()
                    X_train_processed[col] = label_encoders[col].fit_transform(X_train_processed[col].astype(str))
                    
                    # Transform test data with handling for unseen labels
                    # Get unique values in test set
                    test_values = X_test_processed[col].astype(str)
                    
                    # Find values in test that weren't in train
                    unseen_mask = ~test_values.isin(label_encoders[col].classes_)
                    unseen_count = unseen_mask.sum()
                    
                    if unseen_count > 0:
                        unknown_counts[col] = unseen_count
                        # Replace unseen values with the most common value from training
                        most_common_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else label_encoders[col].classes_[0]
                        test_values[unseen_mask] = str(most_common_value)
                    
                    # Transform test data
                    X_test_processed[col] = label_encoders[col].transform(test_values)
            
            # Report unknown categories
            if unknown_counts:
                print(f"⚠️ Unknown categories found in test set:")
                for col, count in unknown_counts.items():
                    print(f"  {col}: {count} unseen values (replaced with most common training value)")
                print()

            # Ensure all features are numeric
            X_train_processed = X_train_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_processed = X_test_processed.apply(pd.to_numeric, errors='coerce').fillna(0)

            print(f"✅  Features encoded")
            print(f"  Train shape: {X_train_processed.shape}")
            print(f"  Test shape: {X_test_processed.shape}")
            print()
            
            # ====================================================================
            # STEP 4: Train Multiple Models
            # ====================================================================
            
            print("Training models...")
            print(separator)
            
            # Define models to test
            models_to_train = {
                'LogisticRegression_100': LogisticRegression(max_iter=100, random_state=42, class_weight='balanced'),
                'LogisticRegression_200': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
                #'SVC': SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)
            }
            
            # Store results
            results = {}

            for model_name, model in models_to_train.items():
                print(f"\n{model_name}:")
                print("-" * 40)
                
                # Train the model
                start_time = time.time()
                model.fit(X_train_processed, Y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                y_train_pred = model.predict(X_train_processed)
                y_test_pred = model.predict(X_test_processed)
                
                # Calculate metrics
                train_accuracy = accuracy_score(Y_train, y_train_pred)
                test_accuracy = accuracy_score(Y_test, y_test_pred)
                train_f1 = f1_score(Y_train, y_train_pred)
                test_f1 = f1_score(Y_test, y_test_pred)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'train_time': train_time,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred
                }
                
                # Print results
                print(f"Training time: {train_time:.2f}s")
                print(f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
                print(f"Train F1 Score: {train_f1:.4f} | Test F1 Score: {test_f1:.4f}")
                
                # Visualize confusion matrices for this model
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Training confusion matrix
                disp1 = ConfusionMatrixDisplay.from_predictions(
                    Y_train, y_train_pred, ax=ax1, cmap=jedhaCM, values_format='d'
                )


                ax1.set_title(f"{model_name} - Training Set", color=jedhaColor_black, fontsize=12, fontweight='bold', fontname={jedha_font})
                ax1.set_facecolor(jedha_bg_color)
                ax1.xaxis.label.set_color(jedhaColor_black)
                ax1.yaxis.label.set_color(jedhaColor_black)
                ax1.set_xticklabels(['Pas Fraude', 'Fraude'])
                ax1.set_yticklabels(['Pas Fraude', 'Fraude'])
                ax1.tick_params(colors=jedhaColor_black)

                # Test confusion matrix
                disp2 = ConfusionMatrixDisplay.from_predictions(
                    Y_test, y_test_pred, ax=ax2, cmap=jedhaCM, values_format='d'
                )
                ax2.set_title(f"{model_name} - Test Set", color=jedhaColor_black, fontsize=12, fontweight='bold')
                ax2.set_facecolor(jedha_bg_color)
                ax2.xaxis.label.set_color(jedhaColor_black)
                ax2.yaxis.label.set_color(jedhaColor_black)
                ax2.set_xticklabels(['Pas Fraude', 'Fraude'])
                ax2.set_yticklabels(['Pas Fraude', 'Fraude'])
                ax2.tick_params(colors=jedhaColor_black)
               
                fig.patch.set_facecolor(jedha_bg_color)
                plt.tight_layout()
                plt.savefig(current_path + '/outputs/Results_confusionMatrix_' + model_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
                plt.close(fig)

            print("\n" + separator)
            print("✅ Model training complete!")
            print()
            
            # ====================================================================
            # STEP 5: Model Comparison
            # ====================================================================
            
            print("\n" + separator)
            print("MODEL COMPARISON")
            print(separator)
            print()
            
            # Create comparison dataframe
            results_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Train Accuracy': [r['train_accuracy'] for r in results.values()],
                'Test Accuracy': [r['test_accuracy'] for r in results.values()],
                'Train F1': [r['train_f1'] for r in results.values()],
                'Test F1': [r['test_f1'] for r in results.values()],
                'Time (s)': [r['train_time'] for r in results.values()]
            })

            # Sort by Test F1 Score
            results_df = results_df.sort_values('Test F1', ascending=False)
            
            print(results_df)
            print()

            # Visualize model comparison
            if 1 :
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.patch.set_facecolor(jedha_bg_color)
                
                # 1. Accuracy Comparison (Train vs Test)
                ax = axes[0, 0]
                x = np.arange(len(results_df))
                width = 0.35
                
                ax.bar(x - width/2, results_df['Train Accuracy'], width, 
                    label='Train', color=jedhaColor_blue, alpha=0.8)
                ax.bar(x + width/2, results_df['Test Accuracy'], width, 
                    label='Test', color=jedhaColor_violet, alpha=0.8)
                
                ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
                ax.set_ylabel('Accuracy', fontweight='bold', color=jedhaColor_black)
                ax.set_title('Accuracy Comparison: Train vs Test', fontweight='bold', color=jedhaColor_black)
                ax.set_xticks(x)
                ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
                ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
                ax.set_facecolor(jedha_bg_color)
                ax.tick_params(colors=jedhaColor_black)
                ax.grid(True, alpha=0.3, color=jedhaColor_black)
                for spine in ax.spines.values():
                    spine.set_color(jedhaColor_black)
                
                # 2. F1 Score Comparison (Train vs Test)
                ax = axes[0, 1]
                ax.bar(x - width/2, results_df['Train F1'], width, 
                    label='Train', color=jedhaColor_blue, alpha=0.8)
                ax.bar(x + width/2, results_df['Test F1'], width, 
                    label='Test', color=jedhaColor_violet, alpha=0.8)
                
                ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
                ax.set_ylabel('F1 Score', fontweight='bold', color=jedhaColor_black)
                ax.set_title('F1 Score Comparison: Train vs Test', fontweight='bold', color=jedhaColor_black)
                ax.set_xticks(x)
                ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
                ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
                ax.set_facecolor(jedha_bg_color)
                ax.tick_params(colors=jedhaColor_black)
                ax.grid(True, alpha=0.3, color=jedhaColor_black)
                for spine in ax.spines.values():
                    spine.set_color(jedhaColor_black)
                
                # 3. Overfitting Detection (Accuracy Gap)
                ax = axes[1, 0]
                accuracy_gap = results_df['Train Accuracy'] - results_df['Test Accuracy']
                colors_gap = [jedhaColor_violet if gap > 0.05 else jedhaColor_blue for gap in accuracy_gap]
                
                ax.bar(results_df['Model'], accuracy_gap, color=colors_gap, alpha=0.8)
                ax.axhline(y=0, color=jedhaColor_black, linestyle='-', linewidth=0.5)
                ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
                
                ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
                ax.set_ylabel('Train-Test Accuracy', fontweight='bold', color=jedhaColor_black)
                ax.set_title('Detection Overfitting ', fontweight='bold', color=jedhaColor_black)
                ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
                ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
                ax.set_facecolor(jedha_bg_color)
                ax.tick_params(colors=jedhaColor_black)
                ax.grid(True, alpha=0.3, color=jedhaColor_black)
                for spine in ax.spines.values():
                    spine.set_color(jedhaColor_black)
                
                # 4. Training Time Comparison
                ax = axes[1, 1]
                ax.bar(results_df['Model'], results_df['Time (s)'], color=jedhaColor_blue, alpha=0.8)
                
                ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
                ax.set_ylabel('Training Time (seconds)', fontweight='bold', color=jedhaColor_black)
                ax.set_title('Training Time Comparison', fontweight='bold', color=jedhaColor_black)
                ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
                ax.set_facecolor(jedha_bg_color)
                ax.tick_params(colors=jedhaColor_black)
                ax.grid(True, alpha=0.3, color=jedhaColor_black)
                for spine in ax.spines.values():
                    spine.set_color(jedhaColor_black)
                
                plt.suptitle('Comparaison des modeles',
                            fontsize=16, fontweight='bold', color=jedhaColor_black, y=0.995)
                plt.tight_layout()
                plt.savefig(current_path + '/outputs/Results_model_performance_comparison_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
                plt.close(fig)
            
            # ====================================================================
            # STEP 6: Best Model Analysis
            # ====================================================================
            
            print("\n" + separator)
            print("BEST MODEL DETAILED ANALYSIS")
            print(separator)
            
            best_model_name = results_df.iloc[0]['Model']
            best_result = results[best_model_name]
            
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   Test F1 Score: {best_result['test_f1']:.4f}")
            print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
            print()
            
            # ROC Curve for best model
            fig, ax = plt.subplots(figsize=(10, 8))
            RocCurveDisplay.from_estimator(
                best_result['model'], 
                X_test_processed, 
                Y_test, 
                ax=ax, 
                color=jedhaColor_violet,
                lw=3
            )
            ax.set_facecolor(jedha_bg_color)
            fig.patch.set_facecolor(jedha_bg_color)
            ax.set_title(f"ROC Curve - {best_model_name} (Test Set)", 
                        fontsize=14, fontweight='bold', color=jedhaColor_black)
            ax.xaxis.label.set_color(jedhaColor_black)
            ax.yaxis.label.set_color(jedhaColor_black)
            ax.tick_params(colors=jedhaColor_black)
            for spine in ax.spines.values():
                spine.set_color(jedhaColor_black)
            ax.legend(facecolor=jedha_bg_color, labelcolor=jedhaColor_black)
            ax.grid(True, alpha=0.3, color=jedhaColor_black)
            plt.tight_layout()
            plt.savefig(current_path + '/outputs/Results_roc_curve_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
            plt.close(fig)
            
            # Classification Report
            print("\nClassification Report (Test Set):")
            print(separator)
            print(classification_report(Y_test, best_result['y_test_pred'], 
                                    target_names=['Non-Fraud', 'Fraud']))
            
            # Feature importance (if RandomForest)
            if best_model_name == 'RandomForest':
                print("\nFeatures les plus importantes dans la creation du Random Forest:")
                print(separator)
                
                feature_importance = pd.DataFrame({
                    'feature': X_train_processed.columns,
                    'importance': best_result['model'].feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                print(feature_importance)
                
                # Visualize feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['feature'], feature_importance['importance'], 
                    color=jedhaColor_violet, alpha=0.8)
                ax.set_xlabel('Importance', fontweight='bold', color=jedhaColor_black)
                ax.set_ylabel('Feature', fontweight='bold', color=jedhaColor_black)
                ax.set_title('Features les plus importantes dans la creation du Random Forest', 
                            fontweight='bold', color=jedhaColor_black)
                ax.set_facecolor(jedha_bg_color)
                fig.patch.set_facecolor(jedha_bg_color)
                ax.tick_params(colors=jedhaColor_black)
                ax.grid(True, alpha=0.3, color=jedhaColor_black, axis='x')
                for spine in ax.spines.values():
                    spine.set_color(jedhaColor_black)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(current_path + '/outputs/Results_feature_importance_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
                plt.close(fig)
            
    print("\n✅ Model evaluation complete!")
    print(separator)



    # ============================================================================
    # MLflow Configuration and Experiment Setup
    # ============================================================================

    print(separator)
    print("MLFLOW EXPERIMENT TRACKING")
    print(separator)
    print()

    # Set tracking URI (MLflow server)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set experiment info
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get experiment metadata
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    

    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print()

    # ============================================================================
    # Log Best Model to MLflow
    # ============================================================================
        # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False) # We won't log models right away
    # Parse arguments given in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", default=3)
    parser.add_argument("--min_samples_split", default=5)
    args = parser.parse_args()

    if 'best_model_name' in locals():
        
        # Start MLflow run
        print(f"Logging best model to MLflow: {best_model_name}")
        print()
        
        # Start MLflow run

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{best_model_name}") as run:
       # with mlflow.start_run(experiment_id = experiment.experiment_id) as run:
            
            # ====================================================================
            # Log Parameters
            # ====================================================================
            
            print("Logging parameters...")
            
            # Get model parameters
            model_params = best_result['model'].get_params()
            
            # Log general parameters
            
            mlflow.log_param("model_type", best_model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("debug_mode", debug)
            mlflow.log_param("sampling_size", samplingSize if debug else len(dfRaw))
            
            # Log model-specific parameters
            for param_name, param_value in model_params.items():
                try:
                    # Skip complex objects
                    if isinstance(param_value, (int, float, str, bool)) or param_value is None:
                        mlflow.log_param(f"model_{param_name}", param_value)
                except Exception as e:
                    print(f"  ⚠️ Could not log parameter {param_name}: {e}")
            
            print("✅  Parameters logged")
            print()


            # ====================================================================
            # Log Tags
            # ====================================================================
            
            print("Logging tags...")
            
            try:
                mlflow.set_tag("status", "staging")
                mlflow.set_tag("model_family", "LBPFraudDetection")
                mlflow.set_tag("model_name", best_model_name)
                mlflow.set_tag("data_source", inputDBFormat)
                
                print("✅  Tags logged")
            except Exception as e:
                print(f"⚠️ Could not log tags: {e}")
            
            print()

            # ====================================================================
            # Log Metrics
            # ====================================================================
            
            print("Logging metrics...")
            
            # Training metrics
            mlflow.log_metric("train_accuracy", best_result['train_accuracy'])
            mlflow.log_metric("train_f1_score", best_result['train_f1'])
            
            # Test metrics (most important)
            mlflow.log_metric("test_accuracy", best_result['test_accuracy'])
            mlflow.log_metric("test_f1_score", best_result['test_f1'])
            
            # Performance metrics
            mlflow.log_metric("training_time_seconds", best_result['train_time'])
            
            # Dataset metrics
            mlflow.log_metric("total_samples", len(X))
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("fraud_ratio", Y.sum() / len(Y))
            
            print("✅  Metrics logged")
            print()
            
            # ====================================================================
            # Log Model
            # ====================================================================
            
            print("Logging model artifact...")
            
            try:
                # Create predictions for signature inference
                predictions = best_result['model'].predict(X_train_processed)
                
                # Infer model signature
                signature = infer_signature(X_train_processed, predictions)
                
                # Calculate actual scores for logging
                train_score = best_result['model'].score(X_train_processed, Y_train)
                test_score = best_result['model'].score(X_test_processed, Y_test)

                mlflow.log_metric("train_score", train_score)
                mlflow.log_metric("test_score", test_score)
            except Exception as e:
                print(f"⚠️ Could not calculate signature: {e}")
                signature = None

            # Log the model to MLflow (skip if S3 not configured)
            try:
                mlflow.sklearn.log_model(
                    sk_model=best_result['model'],
                    artifact_path=f"{EXPERIMENT_NAME}",
                    registered_model_name=f"{best_model_name}",
                    signature=signature,
                    input_example=X_train_processed.head(5)
                )



                print("✅  Model artifact logged to MLflow")
            except Exception as artifact_error:
                
                print("⚠️ S3 credentials not configured. Skipping model artifact upload.")
                print("  Metrics and parameters have been logged successfully.")

                        
            print()

            # ====================================================================
            # Display Run Information
            # ====================================================================
            
            print(separator)
            print("MLFLOW RUN SUMMARY")
            print(separator)
            print(f"Run ID: {run.info.run_id}")
            print(f"Experiment ID: {run.info.experiment_id}")
            print(f"Model: {best_model_name}")
            print(f"Test F1 Score: {best_result['test_f1']:.4f}")
            print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
            print()
            print(f"View run at: {mlflow_tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}")
            print("=" * 80)
            
        print()
        print("✅ MLflow logging complete!")
        print()
        print("Note: If you see warnings about artifacts or model logging,")
        print("this is usually due to S3 bucket configuration. Metrics and")
        print("parameters are always logged to the MLflow tracking server.")
        
    else:
        print("⚠️ No trained models found. Please run the model training cell first.")
        print("Variables 'best_model_name' and 'best_result' are required.")




# Récupérer la dernière version du modèle
client = mlflow.MlflowClient()
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
else:
    print("[WARN] Aucun modèle trouvé dans le registre.")