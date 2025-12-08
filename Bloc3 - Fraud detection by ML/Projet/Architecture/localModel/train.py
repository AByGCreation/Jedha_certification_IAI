#=====================================================================
#=============================== IMPORTS ================================
#=====================================================================


# ==================== DATA MANIPULATION ====================
from math import radians, cos, sin, asin, sqrt
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# ==================== VISUALIZATION ====================
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import folium
from folium.plugins import MarkerCluster

# ==================== MACHINE LEARNING ====================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.base import BaseEstimator, TransformerMixin


# ==================== MLFLOW ====================
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ==================== UTILITIES ====================
import os
import warnings
import time
import datetime
import json
from dotenv import load_dotenv
from IPython.core.magic import register_cell_magic
import platform
from sklearn.metrics import roc_auc_score

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

debug = False  # True for debug mode (smaller dataset, configure size below in : samplingSize var), False for full dataset
plottingEDA = True # True to enable EDA plotting, False to disable
mlFlowLocal = False  # True for local mlflow, False for hosted mlflow

#============================================================================= 

samplingSize = 20000  # Number of rows to sample in debug mode, minimum 1000

neonDB_connectionURL = 'postgresql://neondb_owner:npg_UIrY18vhNmLE@ep-curly-sound-ag9a7x4l-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
neonDB_fraudTableName = "neondb"
HF_connectionURL = "https://huggingface.co/spaces/sdacelo/real-time-fraud-detection"
HF_connectionCSV = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
local_connectionURL = os.path.abspath("./Bloc3 - Fraud detection by ML/Projet/datasSources/inputDataset/fraudTest.csv")  # absolute path
localDB_connectionURL = os.path.abspath("./Bloc3 - Fraud detection by ML/Projet/datasSources/inputDataset/fraudTest.db")  # absolute path

localDB_tableName = "transactions"
inputDBFormat = "db"  # "csv" or "db" or neon or HF_CSV

current_path = os.path.dirname(os.path.abspath(__file__))

dfRaw = pd.DataFrame()

separator = ("="*80)

#---- Name defined by user for project mlFlow ----

modelPrefix = "LBP_fraud_detector_"
EXPERIMENT_NAME = "LBPFraudDetector"

if mlFlowLocal == True:
    mlflow_tracking_uri = "http://localhost:4000/"
else:
    mlflow_tracking_uri = "https://davidrambeau-bloc3-mlflow.hf.space/"

#---- Jedha Colors for plots ----

jedhaColor_violet = '#8409FF'
jedhaColor_blue = '#3AE5FF'
jedhaColor_blueLight = '#89C2FF'
jedhaColor_white = '#DFF4F5'
jedhaColor_black = '#170035'

jedha_bg_color = jedhaColor_white
jedha_grid_color = jedhaColor_black

if platform.system() == "Darwin":
    jedha_font = "Avenir Next"
else:
    jedha_font = "Avenir Next, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol"

# Plotly Jedha Template
pio.templates["jedha_template"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=jedha_font, color=jedhaColor_black),
        title=dict(x=0.5, xanchor="center", font=dict(size=24, color=jedhaColor_black)),
        plot_bgcolor=jedha_bg_color,
        paper_bgcolor=jedha_bg_color,
        xaxis=dict(
            gridcolor=jedha_grid_color,
            zerolinecolor=jedha_grid_color,
            linecolor=jedha_grid_color,
            ticks="outside",
            tickcolor=jedha_grid_color,
        ),
        yaxis=dict(
            gridcolor=jedha_grid_color,
            zerolinecolor=jedha_grid_color,
            linecolor=jedha_grid_color,
            ticks="outside",
            tickcolor=jedha_grid_color,
        ),
        legend=dict(
            bgcolor=jedha_bg_color,
            bordercolor=jedha_grid_color,
            borderwidth=1,
        ),
    )
)
pio.templates.default = "jedha_template"

colors = np.array([(132, 9, 255), (223,244,245), (58, 229, 255)])/255.
jedhaCM = matplotlib.colors.LinearSegmentedColormap.from_list('Jedha Scale', colors)
jedhaCMInverted = matplotlib.colors.LinearSegmentedColormap.from_list('Jedha Scale', colors)

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

def saveMap(df, nbPoint=None, outputPath=''):
    """Save a map with merchant locations and transaction clusters.

    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
        nbPoint (int, optional): Number of points to plot. Defaults to None.
        outputPath (str, optional): Path to save the map HTML file. Defaults to ''.
    """
    # ~15min pour l'ensemble des points un fichier de 500mo
    
    # Center map on mean latitude and longitude of merchant locations
    center_lat = df['merch_lat'].astype(float).mean()
    center_lon = df['merch_long'].astype(float).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron', control_scale=True, width='100%', height='100%', max_bounds=True)

    # Add merchant locations as points
    
    if nbPoint:
        dfTemp = df.head(nbPoint)
    else:
        dfTemp = df

    # Group by merchant and count number of transactions and frauds
    merchant_stats = dfTemp.groupby('merchant').agg(
        total_transactions=('is_fraud', 'size'),
        fraud_count=('is_fraud', 'sum')
    ).reset_index()

    # Draw points for merchant locations on the map
    # Create separate marker clusters for fraud and legitimate transactions

    fraud_cluster = MarkerCluster(name='Transactions frauduleuses').add_to(m)
    legit_cluster = MarkerCluster(name='Transactions légitimes').add_to(m)

    for idx, row in dfTemp.iterrows():
        lat = float(row['merch_lat'])
        lon = float(row['merch_long'])
        merchant = row['merchant']
        total_tx = merchant_stats.loc[merchant_stats['merchant'] == merchant, 'total_transactions'].values[0]
        fraud_tx = merchant_stats.loc[merchant_stats['merchant'] == merchant, 'fraud_count'].values[0]
        popup_text = (
            f"<b>Vendeur</b>: {merchant}<br>"
            f"<b>Montant</b>: {row['amt']}$ <br>"
            f"<b>Fraude</b>: {row['is_fraud']}<br>"
            f"<b>Nombre total de transactions</b>: {total_tx}<br>"
            f"<b>Nombre de transactions frauduleuses</b>: {fraud_tx}"
        )
        if row['is_fraud'] == 1:
            icon = folium.Icon(color='purple', icon='exclamation-sign', prefix='glyphicon')
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=icon
            ).add_to(fraud_cluster)
        else:
            icon = folium.Icon(color='lightblue', icon='ok-sign', prefix='glyphicon')
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=icon
            ).add_to(legit_cluster)

    # Add layer control to toggle clusters
    folium.LayerControl().add_to(m)

    # Add legend to the map
    legend_html = f'''
     <div id="customLegend" style="
         position: fixed; 
         bottom: 50px; left: 50px; width: 200px; height: 90px; 
         background-color: white; z-index:9999; font-size:14px;
         border:2px solid grey; border-radius:8px; padding: 10px;">
         <b>Légende</b><br>
         <i class="glyphicon glyphicon-exclamation-sign" style="color:{jedhaColor_violet}"></i> Transaction frauduleuse<br>
         <i class="glyphicon glyphicon-ok-sign" style="color:{jedhaColor_blue}"></i> Transaction légitime
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


    m.save(outputPath, close_file=False)
    print(f"✅ Map saved to {outputPath}")

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
                    print(f"✓ {col}: converted to datetime64")
                else:
                    print(f"⊘ {col}: already datetime64")
            except Exception as e:
                print(f"✗ {col}: Failed to convert ({e})")

def dataSourceLoader(inputDBFormat: str) -> pd.DataFrame|bool:

    """Load data from specified source format into a DataFrame.

    Args:
        inputDBFormat (str): The format of the data source ("csv", "db", "neon", "HF").

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    dfRaw = pd.DataFrame()
    try:
        if inputDBFormat == "csv":
            dfRaw = pd.read_csv(local_connectionURL)
        elif inputDBFormat == "db":

            conn = sqlite3.connect(localDB_connectionURL)
            query = f"SELECT * FROM {localDB_tableName}"
            dfRaw = pd.read_sql_query(query, conn)
            conn.close()
        elif inputDBFormat == "neon":
            engine = create_engine(neonDB_connectionURL)
            query = f"SELECT * FROM {neonDB_fraudTableName}"
            dfRaw = pd.read_sql_query(query, engine)
        elif inputDBFormat == "HF_CSV":
            dfRaw = pd.read_csv(HF_connectionCSV)
        else:
            print("❌ Invalid inputDBFormat. Please choose 'csv', 'db', 'neon', or 'HF_CSV'.")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False


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

    # Calculate the distance between customer and merchant locations
    df['distance_km'] = df.apply(
        lambda row: haversine(row['long'], row['lat'], row['merch_long'], row['merch_lat']), 
        axis=1
    )
    print(f"Distance calculated. Min: {df['distance_km'].min():.2f} km, "
        f"Max: {df['distance_km'].max():.2f} km, "
        f"Mean: {df['distance_km'].mean():.2f} km")

    # Convert dob to datetime and calculate age
    if not pd.api.types.is_datetime64_any_dtype(df['dob']):
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365
    df = df.sort_values(by='age', ascending=True)

    # Convert amt to numeric
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce')
    
    # Extract hour from transaction datetime
    df['trans_hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour

    # Drop columns that are no longer needed (only if they exist)
   


    columns_to_drop = [
        'dob', 'trans_date_trans_time', 'unix_time', 
        'lat', 'long', 'merch_lat', 'merch_long', 
        'street', 'first', 'last', 'Column1', 'trans_num', "unamed: 0"
    ]
    




    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    




    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=jedhaCMInverted, square=True)
    plt.title("Matrice de corrélation des variables numériques")
    #plt.show()
    plt.savefig(current_path + '/outputs/Analysis_correlationMatrix_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')









    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
        print(f"Dropped columns: {existing_columns_to_drop}")
    else:
        print("No columns to drop (already removed or not present).")
    
    return df

#=============================== CUSTOM TRANSFORMERS ============================= #
# class DistanceCalculator(BaseEstimator, TransformerMixin):
#     """
#     Custom transformer to calculate the great circle distance between customer and merchant locations.
    
#     This transformer prevents data leakage by computing distances independently for each sample
#     without using any global statistics.
    
#     Attributes
#     ----------
#     None (stateless transformer)
    
#     Methods
#     -------
#     fit(X, y=None)
#         No-op operation as this transformer is stateless.
#     transform(X)
#         Calculates the distance between customer (lat, long) and merchant (merch_lat, merch_long).
    
#     Examples
#     --------
#     >>> from sklearn.pipeline import Pipeline
#     >>> pipeline = Pipeline([
#     ...     ('distance', DistanceCalculator()),
#     ...     ('classifier', RandomForestClassifier())
#     ... ])
#     >>> pipeline.fit(X_train, y_train)
#     """
    
#     def fit(self, X, y=None):
#         """
#         Fit method (no-op for stateless transformer).
        
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Input features.
#         y : array-like, optional
#             Target variable (ignored).
            
#         Returns
#         -------
#         self : object
#             Returns self for method chaining.
#         """
#         return self
    
#     def transform(self, X):
#         """
#         Calculate distance between customer and merchant locations.
        
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Input features containing 'lat', 'long', 'merch_lat', 'merch_long' columns.
            
#         Returns
#         -------
#         X_transformed : pd.DataFrame
#             DataFrame with added 'distance_km' column.
#         """
#         X = X.copy()
#         X['distance_km'] = X.apply(
#             lambda row: haversine(row['long'], row['lat'], row['merch_long'], row['merch_lat']), 
#             axis=1
#         )
#         return X

# class AgeCalculator(BaseEstimator, TransformerMixin):
#     """
#     Custom transformer to calculate customer age from date of birth.
    
#     This transformer computes age at the time of transformation, ensuring
#     consistent age calculation for both training and test data.
    
#     Attributes
#     ----------
#     reference_date_ : pd.Timestamp
#         The reference date used for age calculation (set during fit).
    
#     Methods
#     -------
#     fit(X, y=None)
#         Stores the reference date for age calculation.
#     transform(X)
#         Calculates age from 'dob' column using the stored reference date.
    
#     Examples
#     --------
#     >>> from sklearn.pipeline import Pipeline
#     >>> pipeline = Pipeline([
#     ...     ('age', AgeCalculator()),
#     ...     ('classifier', RandomForestClassifier())
#     ... ])
#     >>> pipeline.fit(X_train, y_train)
#     """
    
#     def fit(self, X, y=None):
#         """
#         Store reference date for consistent age calculation.
        
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Input features.
#         y : array-like, optional
#             Target variable (ignored).
            
#         Returns
#         -------
#         self : object
#             Returns self for method chaining.
#         """
#         self.reference_date_ = pd.Timestamp.now()
#         return self
    
#     def transform(self, X):
#         """
#         Calculate customer age from date of birth.
        
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Input features containing 'dob' column.
            
#         Returns
#         -------
#         X_transformed : pd.DataFrame
#             DataFrame with added 'age' column.
#         """
#         X = X.copy()
        
#         # Ensure dob is datetime
#         if not pd.api.types.is_datetime64_any_dtype(X['dob']):
#             X['dob'] = pd.to_datetime(X['dob'], errors='coerce')
        
#         # Calculate age
#         X['age'] = (self.reference_date_ - X['dob']).dt.days // 365
        
#         return X

# class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
#     """
#     Custom transformer to extract temporal features from transaction datetime.
    
#     Extracts hour, day of week, and month from transaction timestamp to capture
#     temporal patterns in fraudulent behavior.
    
#     Attributes
#     ----------
#     None (stateless transformer)
    
#     Methods
#     -------
#     fit(X, y=None)
#         No-op operation as this transformer is stateless.
#     transform(X)
#         Extracts time-based features from 'trans_date_trans_time' column.
    
#     Examples
#     --------
#     >>> from sklearn.pipeline import Pipeline
#     >>> pipeline = Pipeline([
#     ...     ('time_features', TimeFeatureExtractor()),
#     ...     ('classifier', RandomForestClassifier())
#     ... ])
#     >>> pipeline.fit(X_train, y_train)
#     """
    
#     def fit(self, X, y=None):
#         """
#         Fit method (no-op for stateless transformer).
        
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Input features.
#         y : array-like, optional
#             Target variable (ignored).
            
#         Returns
#         -------
#         self : object
#             Returns self for method chaining.
#         """
#         return self
    
#     def transform(self, X):
#         """
#         Extract temporal features from transaction datetime.
        
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Input features containing 'trans_date_trans_time' column.
            
#         Returns
#         -------
#         X_transformed : pd.DataFrame
#             DataFrame with added 'trans_hour', 'trans_day', 'trans_month' columns.
#         """
#         X = X.copy()
        
#         # Ensure datetime format
#         trans_dt = pd.to_datetime(X['trans_date_trans_time'], errors='coerce')
        
#         # Extract temporal features
#         X['trans_hour'] = trans_dt.dt.hour
#         X['trans_day'] = trans_dt.dt.dayofweek  # 0=Monday, 6=Sunday
#         X['trans_month'] = trans_dt.dt.month
        
#         return X

# class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns from DataFrame.
    
    This transformer removes columns that are not needed for modeling,
    such as identifiers, raw datetime fields, or redundant features.
    
    Attributes
    ----------
    columns_to_drop : list
        List of column names to remove from the DataFrame.
    
    Methods
    -------
    fit(X, y=None)
        No-op operation as this transformer is stateless.
    transform(X)
        Removes specified columns from the DataFrame.
    
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> dropper = ColumnDropper(['dob', 'trans_date_trans_time'])
    >>> pipeline = Pipeline([
    ...     ('drop_cols', dropper),
    ...     ('classifier', RandomForestClassifier())
    ... ])
    >>> pipeline.fit(X_train, y_train)
    """
    
    def __init__(self, columns_to_drop):
        """
        Initialize the ColumnDropper transformer.
        
        Parameters
        ----------
        columns_to_drop : list
            List of column names to drop from the DataFrame.
        """
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        """
        Fit method (no-op for stateless transformer).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : array-like, optional
            Target variable (ignored).
            
        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        return self
    
    def transform(self, X):
        """
        Drop specified columns from DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
            
        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame with specified columns removed.
        """
        X = X.copy()
        
        # Only drop columns that exist in the DataFrame
        existing_columns = [col for col in self.columns_to_drop if col in X.columns]
        
        if existing_columns:
            X = X.drop(columns=existing_columns)
        
        return X

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
            
            print(f"✓ {col}: {old_dtype} → {new_dtype}")
        except Exception as e:
            print(f"✗ {col}: Failed - {str(e)[:80]}")

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




    # Apply preprocessing
    df = Preprocessor(df)
        
    print("✅ Final cleanup before EDA complete.")
    print()

    print(df.head())

    print("Data types of key columns:")
    print(f"  age: {df['age'].dtype}")
    print(f"  amt: {df['amt'].dtype}")
    print()


    #======================================================================
    ## EDA - Plotting datas ##
    #======================================================================
    if plottingEDA == True:
        # Univariate analysis - Distribution of numeric variables
        print("Generating distributions for numeric features...")
        num_features = ["age", "amt", "trans_hour", "distance_km", "city_pop", "ccn_len"]

        for f in num_features:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot non-fraud distribution
            """
            sns.histplot(
                data=df[df['is_fraud'] == 0],
                x=f,
                bins=50,
                label='Non-Fraude',
                color=jedhaColor_blue,
                alpha=0.7,
                ax=ax
                )
        
            """
            
            # Plot fraud distribution
            sns.histplot(
                data=df[df['is_fraud'] == 1],
                x=f,
                bins=50,
                label='Fraude',
                color=jedhaColor_violet,
                alpha=0.75,
                ax=ax
            )
            
            ax.set_title(f'Distribution - {f}', fontweight='bold', color=jedhaColor_black)
            ax.set_xlabel(f.capitalize(), color=jedhaColor_black)
            ax.set_ylabel('Frequency', color=jedhaColor_black)
            ax.set_facecolor(jedha_bg_color)
            fig.patch.set_facecolor(jedha_bg_color)
            ax.tick_params(colors=jedhaColor_black)
            ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
            plt.tight_layout()

            plt.savefig(current_path + f"/outputs/Analysis_distribution_{f}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

            # fig.show()
           # fig.write_image(current_path + f"/outputs/distribution_{f}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

        print("✅ EDA analysis complete.")


        # Distribution of transaction amounts by fraud status

        print("Distribution des fraudes:")
        print(df['is_fraud'].value_counts())
        print()
        #print(df['is_fraud'].describe())
        print()

        # Visualize transaction amounts: Normal vs Fraudulent
        fig = go.Figure()
        # Correlation heatmap for numeric features

        # numeric_cols = df.select_dtypes(include=['number']).columns
        # corr = df[numeric_cols].corr()

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr, annot=True, fmt=".2f", cmap=jedhaCMInverted, square=True)
        # plt.title("Matrice de corrélation des variables numériques")
        # #plt.show()
        # plt.savefig(current_path + '/outputs/correlationMatrix_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')




        # Normal transactions
        """ fig.add_trace(go.Histogram(
            x=df[df['is_fraud']==0]['amt'],
            name='Transactions conformes',
            nbinsx=50,
            opacity=0.7,
            marker_color=jedhaColor_blue
        )) """

        # Fraud transactions
        # fig.add_trace(go.Histogram(
        #     x=df[df['is_fraud']==1]['amt'],
        #     name='Transactions frauduleuses',
        #     nbinsx=50,
        #     opacity=0.7,
        #     marker_color=jedhaColor_violet
        # ))

        # fig.update_layout(
        #     title='Distribution des montants de transactions: Conformes vs Frauduleuses',
        #     xaxis_title='Montant',
        #     yaxis_title='Fréquence',
        #     barmode='overlay',
        #     height=500,
        #     width=1000
        # )

        # #fig.show()
        # fig.write_image(current_path + '/outputs/Analysis_transaction_amount_distribution_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')


        # Visualize fraud distribution by hour
        fig = go.Figure()

        # Normal transactions by hour
        """ fig.add_trace(go.Histogram(
            x=df[df['is_fraud']==0]['trans_hour'],
            name='Transactions conformes',
            nbinsx=24,
            opacity=0.7,
            marker_color=jedhaColor_blue
        )) """

        # Fraud transactions by hour
        # fig.add_trace(go.Histogram(
        #     x=df[df['is_fraud']==1]['trans_hour'],
        #     name='Transactions frauduleuses',
        #     nbinsx=24,
        #     opacity=0.7,
        #     marker_color=jedhaColor_violet
        # ))

        # fig.update_layout(
        #     title='Distribution des transactions par heure: Conformes vs Frauduleuses',
        #     xaxis_title='Heure de la journée',
        #     yaxis_title='Fréquence',
        #     barmode='overlay',
        #     height=500,
        #     width=1000
        # )

        #fig.show()

        #fig.write_image(current_path + '/outputs/transaction_distribution_by_hour_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')

        # Visualize fraud distribution by category
        fig = go.Figure()

        # Fraud transactions by category
        fraud_by_category = df[df['is_fraud']==1].groupby('category').size().reset_index(name='count')
        fraud_by_category = fraud_by_category.sort_values('count', ascending=False)

        fig.add_trace(go.Bar(
            x=fraud_by_category['category'],
            y=fraud_by_category['count'],
            name='Transactions frauduleuses',
            marker_color=jedhaColor_violet
        ))

        fig.update_layout(
            title='Distribution des transactions frauduleuses par catégorie',
            xaxis_title='Catégorie',
            yaxis_title='Nombre de fraudes',
            height=500,
            width=1000
        )


        # Add Pareto curve (cumulative percentage)
        fraud_by_category['cumulative'] = fraud_by_category['count'].cumsum()
        fraud_by_category['cumulative_pct'] = fraud_by_category['cumulative'] / fraud_by_category['count'].sum() * 100

        fig.add_trace(go.Scatter(
            x=fraud_by_category['category'],
            y=fraud_by_category['cumulative_pct'],
            name='Courbe de Pareto (%)',
            mode='lines+markers',
            marker_color=jedhaColor_blue,
            yaxis='y2'
        ))

        fig.update_layout(
            yaxis2=dict(
                title='Pourcentage cumulatif (%)',
                overlaying='y',
                side='right',
                range=[0, 100],
                showgrid=False,
            )
        )


        #fig.show()
        fig.write_image(current_path + '/outputs/Analysis_transaction_fraud_distribution_by_category_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')


        print("✅ Fraud distribution analysis complete.")


    #======================================================================
    ##  Model training and evaluation ##
    #======================================================================

    print(separator)
    print("MODEL TRAINING AND EVALUATION")
    print(separator)
    print()

    # ============================================================================
    # STEP 1: Prepare Data for Modeling
    # ============================================================================

    print("Separating labels from features...")

    # Separate target variable Y from features X
    X = df.drop(columns=["is_fraud"], inplace=False)
    Y = df["is_fraud"]

    print(f"Y shape: {Y.shape}")
    print(f"X shape: {X.shape}")
    print(f"Unique classes in Y: {Y.unique()}")
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
            print("✓ Sufficient samples for stratified split")
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
            if 'dob' in X_train_processed.columns:
                if pd.api.types.is_datetime64_any_dtype(X_train_processed['dob']):
                    X_train_processed['dob'] = (X_train_processed['dob'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                    X_test_processed['dob'] = (X_test_processed['dob'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
            
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
                print(f"\n⚠️ Unknown categories found in test set:")
                for col, count in unknown_counts.items():
                    print(f"  {col}: {count} unseen values (replaced with most common training value)")
                print()
            
            # Ensure all features are numeric
            X_train_processed = X_train_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_processed = X_test_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            print(f"✓ Features encoded")
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
                'LogisticRegression_200': LogisticRegression(max_iter=200, random_state=42, class_weight='balanced'),
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

                #plt.colorbar(disp1.im_, ax=ax1, label='Count', format='%d')

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
                #plt.colorbar(disp2.im_, ax=ax2, label='Count', format='%d')

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
            #plt.show()
            plt.savefig(current_path + '/outputs/Results_model_performance_comparison_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
            
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
            #plt.show()
            plt.savefig(current_path + '/outputs/Results_roc_curve_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
            
            # Classification Report
            print("\nClassification Report (Test Set):")
            print(separator)
            print(classification_report(Y_test, best_result['y_test_pred'], 
                                    target_names=['Non-Fraud', 'Fraud']))
            
            # Feature importance (if RandomForest)
            if best_model_name == 'RandomForest':
                print("\nFeatures les plus importantes dans l'analyse:")
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
                ax.set_title('Top 10 Feature Importances - RandomForest', 
                            fontweight='bold', color=jedhaColor_black)
                ax.set_facecolor(jedha_bg_color)
                fig.patch.set_facecolor(jedha_bg_color)
                ax.tick_params(colors=jedhaColor_black)
                ax.grid(True, alpha=0.3, color=jedhaColor_black, axis='x')
                for spine in ax.spines.values():
                    spine.set_color(jedhaColor_black)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                #plt.show()
                plt.savefig(current_path + '/outputs/Results_feature_importance_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
            
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

    if 'best_model_name' in locals() and 'best_result' in locals():
        
        print(f"Logging best model to MLflow: {best_model_name}")
        print()
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{best_model_name}_production") as run:
            
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
            
            print("✓ Parameters logged")
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
            
            print("✓ Metrics logged")
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
                
                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=best_result['model'],
                    name=f"{EXPERIMENT_NAME}_model",
                    registered_model_name=f"{best_model_name}_production",
                    signature=signature,
                    input_example=X_train_processed.head(5)
                )
                
                print("✓ Model artifact logged")
            except Exception as e:
                print(f"⚠️ Could not log model artifact: {e}")
                print("  This is usually due to S3 bucket permissions or connectivity issues.")
                print("  Metrics and parameters were still logged successfully.")
            
            print()
            
            # ====================================================================
            # Log Additional Artifacts
            # ====================================================================
            
            print("Logging additional artifacts...")
            
            try:
                # Save feature names and encoders info
                feature_info = {
                    'features': list(X_train_processed.columns),
                    'n_features': X_train_processed.shape[1],
                    'encoded_columns': list(label_encoders.keys()) if 'label_encoders' in locals() else []
                }
                
                # Create temporary file
                temp_file = 'feature_info.json'
                with open(temp_file, 'w') as f:
                    json.dump(feature_info, f, indent=2)
                
                # Try to log artifact
                mlflow.log_artifact(temp_file)
                
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                print("✓ Additional artifacts logged")
            except Exception as e:
                print(f"⚠️ Could not log additional artifacts: {e}")
                print("  This is usually due to S3 bucket permissions or connectivity issues.")
                
                # Clean up temporary file even if logging failed
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            print()
            
            # ====================================================================
            # Log Tags
            # ====================================================================
            
            print("Logging tags...")
            
            try:
                mlflow.set_tag("model_family", "fraud_detection")
                mlflow.set_tag("data_source", inputDBFormat)
                mlflow.set_tag("best_model", "true")
                mlflow.set_tag("production_ready", "true")
                mlflow.set_tag("preprocessing", "manual")
                
                print("✓ Tags logged")
            except Exception as e:
                print(f"⚠️ Could not log tags: {e}")
            
            print()
            
            # ====================================================================
            # Display Run Information
            # ====================================================================
            
            print("=" * 80)
            print("MLFLOW RUN SUMMARY")
            print("=" * 80)
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
