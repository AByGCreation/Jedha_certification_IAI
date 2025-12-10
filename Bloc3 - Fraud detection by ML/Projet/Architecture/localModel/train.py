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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
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

from config import EXPERIMENT_NAME as EXPERIMENT_NAME, HF_connectionCSV as HF_connectionCSV, HF_connectionURL as HF_connectionURL, current_path as current_path, debug as debug, inputDBFormat as inputDBFormat, localDB_connectionURL as localDB_connectionURL, localDB_tableName as localDB_tableName, local_connectionURL as local_connectionURL, mlFlowLocal as mlFlowLocal, mlflow_tracking_uri as mlflow_tracking_uri, modelPrefix as modelPrefix, neonDB_connectionURL as neonDB_connectionURL, neonDB_fraudTableName as neonDB_fraudTableName, os as os, pd as pd, plottingEDA as plottingEDA, project_path as project_path, samplingSize as samplingSize, separator as separator, API_URL as API_URL, bucket_name as bucket_name, aws_region as aws_region
from plotters import MarkerCluster as MarkerCluster, datetime as datetime, drawCorrelationMatrix as drawCorrelationMatrix, folium as folium, go as go, jedhaCMInverted as jedhaCMInverted, jedhaColor_black as jedhaColor_black, jedhaColor_blue as jedhaColor_blue, jedhaColor_violet as jedhaColor_violet, jedha_bg_color as jedha_bg_color, jedha_font as jedha_font, jedha_grid_color as jedha_grid_color, pd as pd, plotFeatureDistributions as plotFeatureDistributions, plt as plt, saveMap as saveMap, sns as sns, draw_confusion_matrices as draw_confusion_matrices, draw_model_comparison as draw_model_comparison
from graphics import colors as colors, go as go, jedhaCM as jedhaCM, jedhaCMInverted as jedhaCMInverted, jedhaColor_black as jedhaColor_black, jedhaColor_blue as jedhaColor_blue, jedhaColor_blueLight as jedhaColor_blueLight, jedhaColor_violet as jedhaColor_violet, jedhaColor_white as jedhaColor_white, jedha_bg_color as jedha_bg_color, jedha_colors as jedha_colors, jedha_font as jedha_font, jedha_grid_color as jedha_grid_color, np as np, pio as pio, platform as platform 
from dataLoader import cfg as cfg, create_engine as create_engine, dataSourceLoader as dataSourceLoader, pd as pd, sqlite3 as sqlite3
from converters import haversine as haversine, datetimeConverter as datetimeConverter
from preprocessor import Preprocessor as Preprocessor
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


# modelPrefix = "LBP_fraud_detector_"
EXPERIMENT_NAME = "LBPFraudDetector"

print("✅ Configuration load successful.")
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
        
        print(separator)
        print("INITIAL DATAFRAME")
        print(separator)
        print(f"Shape: {df.shape}")
        print()
        print("Columns in df:")
        print(df.columns.to_list())

        print(f"🐛 Debug mode: {len(df)} rows, {len(df.columns)} columns (Fraud: {len(fraud_sample)}, Non-fraud: {len(non_fraud_sample)})")
    else:
        df = dfRaw
        print(f"Using full mode with {len(df)} rows and {len(df.columns)} columns .")

    print(separator)
    print("✅ Data importing successfully completed.")
    print(separator)
    print()
    print(separator)
    # Store original memory usage
    original_memory = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"💾 Original memory usage: {original_memory:.2f} MB")
    print(separator)
    print()

    #======================================================================
    ## EDA - Preprocessing functions ##
    #======================================================================
    
    print(separator)
    print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
    print(separator)
    print()

    # ============================================================================
    # STEP 1: Prepare Data for Modeling
    # ============================================================================





    df = Preprocessor(df)

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    one_hot_encoder = Pipeline(steps=[('OHE', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    num_encoder = Pipeline(steps=[('numerical', StandardScaler())])

    categorical_features = (X_train_processed.select_dtypes("object").columns.tolist()) 
    numerical_features = X_train_processed.columns[~X_train_processed.columns.isin(X_train_processed.select_dtypes("object").columns)]

    preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', one_hot_encoder, categorical_features),
        ('numerical', num_encoder, numerical_features),
    ]
    )

    print("✅  Features encoded")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")
    print(f"Train head:{X_train_processed.head()}")
    print(f"Test head:{X_test_processed.head()}")
    print()

    pipelines = {
    "Random Forest": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())]),
    #"Gradient Boosting": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier())]),
    "Logistic Regression_100": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=100))]),
    "Logistic Regression_1000": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
    }

    modelsResults = {}
    
    for model_name, model in pipelines.items():

        # Train the model
        print(separator)
        print(f"💪 Training → {model_name}:")
        print(separator)

        start_time = time.time()
        model.fit(X_train_processed, y_train)
        train_time = time.time() - start_time

        # Make predictions
        print(separator)
        print(f"🔮 Predicting...")
        print(separator)

        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)


        modelsResults[model_name] = {
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
        print(f"⏱️ Training time: {train_time:.2f}s")
        print(f"✅ Train Accuracy: {train_accuracy:.4f}")
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"📈 Train F1: {train_f1:.4f}")
        print(f"📈 Test F1: {test_f1:.4f}")
        
        print(separator)
        print(f"📊 Dashboard for {model_name}...")
        print(separator)
        # Visualize confusion matrices for this model
        
        draw_confusion_matrices(model_name, y_train, y_train_pred, y_test, y_test_pred)
        print("✅ Dashboard saved for " + model_name + ".")

        print(separator)
        print("✅ Model training complete!")
        print(separator)
        
        # ====================================================================
        # STEP 5: Model Comparison
        # ====================================================================
        
        print(separator)
        print("🔍 MODEL COMPARISON")
        print(separator)
        print()
        
        # Create comparison dataframe
        results_df = pd.DataFrame({
            'Model': list(modelsResults.keys()),
            'Train Accuracy': [r['train_accuracy'] for r in modelsResults.values()],
            'Test Accuracy': [r['test_accuracy'] for r in modelsResults.values()],
            'Train F1': [r['train_f1'] for r in modelsResults.values()],
            'Test F1': [r['test_f1'] for r in modelsResults.values()],
            'Time (s)': [r['train_time'] for r in modelsResults.values()]
        })

        # Sort by Test F1 Score
        results_df = results_df.sort_values('Test F1', ascending=False)
        print("Model Comparison Results:")
        draw_model_comparison(results_df)
        # ====================================================================
        # STEP 6: Best Model Analysis
        # ====================================================================
        
        print("\n" + separator)
        print("BEST MODEL DETAILED ANALYSIS")
        print(separator)
        
        best_model_name = results_df.iloc[0]['Model']
        best_result = modelsResults[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test F1 Score: {best_result['test_f1']:.4f}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
        print()
        
        # ROC Curve for best model
        fig, ax = plt.subplots(figsize=(10, 8))
        RocCurveDisplay.from_estimator(
            best_result['model'], 
            X_test_processed, 
            y_test, 
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
        print(classification_report(y_test, best_result['y_test_pred'], 
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
            mlflow.log_metric("fraud_ratio", y.sum() / len(y))
            
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
                train_score = best_result['model'].score(X_train_processed, y_train)
                test_score = best_result['model'].score(X_test_processed, y_test)

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



                print(f"✅  Model artifact {best_model_name} logged to MLflow")
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
            print(separator)
            
        print()
        print("✅ MLflow logging complete!")
        print()
        print(separator)
        print(f"https://{aws_region}.console.aws.amazon.com/s3/buckets/{bucket_name}?prefix={EXPERIMENT_NAME}/{run.info.run_id}/&region={aws_region} ")

        
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

print("...Done!")
print(f"---Total training time: {time.time()-start_time}")

        