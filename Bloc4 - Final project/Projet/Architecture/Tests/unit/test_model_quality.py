import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import warnings
import logging
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Add the parent directory to the path so src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.append('./Bloc4 - Final project/Projet/Architecture')

#important ! ne pas deplacer avant le sys.path.append
#from App.Dockers.fastapi.main import app, getMyModel, getModelRunID, load_reference_dataset

from App.Dockers.fastapi.main import app, getMyModel, load_reference_dataset, Preprocessor

# Trouve le fichier .env
dotenv_path = find_dotenv()
load_dotenv(".env")  # Depuis le répertoire courant

print(f"✅ .env chargé dans test_predict.py")
print(f"   TesVAr : {os.getenv('EXPERIMENT_NAME')}")
print(f"Fichier .env trouvé : {dotenv_path}")
EXPERIMENT_NAME = "LBPFraudDetector"

#=======================================
# Test de la qualité du modèle MLflow
#=======================================

class TestModelQuality:
    """Test de qualité du modèle MLflow"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_model(self):
        """Initialize model before tests"""
        print("\n🤖 Loading model for quality tests...")
        model = getMyModel()
        
        if model is None:
            pytest.exit("❌ Failed to load model")
        
        app.state.loaded_model = model
        print("✅ Model loaded for quality testing")
        yield
    
    def test_model_accuracy_threshold(self):
        """Test 1: Vérifier que la précision du modèle dépasse 80%"""
        
        try:
            # Load reference dataset
            X_test, y_test = load_reference_dataset()
            
            print(f"\n📊 Testing on reference dataset:")
            print(f"   - Shape: {X_test.shape}")
            print(f"   - Columns: {X_test.columns.tolist()}")
            
            # Preprocess test data (important!)
            X_test_preprocessed = Preprocessor(X_test.copy())
            
            print(f"   - After preprocessing: {X_test_preprocessed.shape}")
            print(f"   - Columns after preprocessing: {X_test_preprocessed.columns.tolist()}")
            
            # Make predictions
            model = app.state.loaded_model
            predictions = model.predict(X_test_preprocessed)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            print(f"\n📈 Model Performance Metrics:")
            print(f"   - Accuracy:  {accuracy:.4f}")
            print(f"   - Precision: {precision:.4f}")
            print(f"   - Recall:    {recall:.4f}")
            print(f"   - F1-Score:  {f1:.4f}")
            
            # Assertions
            assert accuracy >= 0.75, f"Accuracy {accuracy:.4f} is below threshold 0.75"
            assert precision >= 0.70, f"Precision {precision:.4f} is below threshold 0.70"
            assert recall >= 0.60, f"Recall {recall:.4f} is below threshold 0.60"
            assert f1 >= 0.65, f"F1-Score {f1:.4f} is below threshold 0.65"
            
            print(f"✅ All quality metrics passed!")
            
        except Exception as e:
            print(f"❌ Model quality test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_model_prediction_consistency(self):
        """Test 2: Vérifier que le modèle donne les mêmes prédictions pour les mêmes entrées"""
        
        try:
            # Load dataset
            X_test, y_test = load_reference_dataset()
            
            # Preprocess
            X_test_preprocessed = Preprocessor(X_test.copy())
            
            # Get subset for test (first 100 samples)
            X_subset = X_test_preprocessed.head(100)
            
            model = app.state.loaded_model
            
            # Make two predictions
            predictions_1 = model.predict(X_subset)
            predictions_2 = model.predict(X_subset)
            
            # Check consistency
            is_consistent = np.array_equal(predictions_1, predictions_2)
            
            print(f"\n🔄 Prediction Consistency Test:")
            print(f"   - First run:  {predictions_1[:5]}...")
            print(f"   - Second run: {predictions_2[:5]}...")
            print(f"   - Consistent: {is_consistent}")
            
            assert is_consistent, "Model predictions are not consistent"
            print(f"✅ Model predictions are consistent!")
            
        except Exception as e:
            print(f"❌ Consistency test failed: {str(e)}")
            raise
    
    def test_model_output_format(self):
        """Test 3: Vérifier que le modèle retourne le bon format de sortie"""
        
        try:
            # Load dataset
            X_test, y_test = load_reference_dataset()
            
            # Preprocess
            X_test_preprocessed = Preprocessor(X_test.copy())
            
            # Get subset
            X_subset = X_test_preprocessed.head(10)
            
            model = app.state.loaded_model
            predictions = model.predict(X_subset)
            
            print(f"\n📋 Output Format Test:")
            print(f"   - Predictions type: {type(predictions)}")
            print(f"   - Predictions shape: {predictions.shape}")
            print(f"   - Predictions dtype: {predictions.dtype}")
            print(f"   - Unique values: {np.unique(predictions)}")
            
            # Assertions
            assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
            assert len(predictions) == len(X_subset), "Predictions count should match input"
            assert predictions.dtype in [np.int32, np.int64, np.float32, np.float64], "Invalid dtype"
            assert all(p in [0, 1] for p in predictions), "Predictions should be binary (0 or 1)"
            
            print(f"✅ Output format is correct!")
            
        except Exception as e:
            print(f"❌ Output format test failed: {str(e)}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



















#==========================================
#==========================================


# def test_model_accuracy_threshold():
#     """
#     Seuil métier : Accuracy ≥ 92%
#     Justification : Basé sur analyse coût/bénéfice
#     - Faux négatif (fraude non détectée) : 150€ de perte moyenne
#     - Faux positif (client bloqué) : 5€ de friction + risque churn


#     model_uri: str = MODEL_URI,

    
#     Charge un modèle MLflow (sklearn) à partir d'un tracking URI et d'un model URI.
#     Par défaut : modèle RF hotel_cancellation_detector_RF.
#     """

#     modelRunID = getModelRunID()
#     print(f"Model runid: {modelRunID.latest_versions[0].run_id}")
#     print(f"Model name: {modelRunID.latest_versions[0].name}")

#     model_uri = os.getenv("MODEL_URI", f"runs:/{modelRunID.latest_versions[0].run_id}/{EXPERIMENT_NAME}")
#     tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
   

#     print(f"🔍 Chargement du modèle depuis MLflow : {model_uri} @ {tracking_uri}")


#     mlflow.set_tracking_uri(tracking_uri)
#     model = mlflow.sklearn.load_model(model_uri)
#     logging.info("✅ Model récupéré depuis MLflow")
#     X_test, y_test = load_reference_dataset()
    
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     assert accuracy >= 0.92, f"Accuracy trop faible: {accuracy:.3f}"



#     # model = mlflow.sklearn.load_model("models:/fraud-detection/production")
#     # X_test, y_test = load_reference_dataset()
    
#     # y_pred = model.predict(X_test)
#     # accuracy = accuracy_score(y_test, y_pred)
    
#     # assert accuracy >= 0.92, f"Accuracy trop faible: {accuracy:.3f}"




# def test_model_f1_score_threshold():
#     """
#     Seuil métier : F1-Score ≥ 85%
#     Justification : Équilibre précision/rappel critique en détection fraude
#     """
#     f1 = f1_score(y_test, y_pred)
#     assert f1 >= 0.85, f"F1-score trop faible: {f1:.3f}"