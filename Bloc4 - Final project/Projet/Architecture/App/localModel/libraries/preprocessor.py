import pandas as pd
import datetime
from converters import haversine
from plotters import drawCorrelationMatrix
import os


current_path = os.path.dirname(os.path.abspath(__file__))

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
