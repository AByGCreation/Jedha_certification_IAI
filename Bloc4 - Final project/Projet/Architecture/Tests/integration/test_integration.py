
import pytest

class TestIntegration:
    """Tests d'intégration FastAPI"""
    
    def test_01_health_check(self, client):
        """Test 1: Health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"✅ Health: {data}")
    
    def test_02_predict_with_model(self, client, require_model):
        """Test 2: Prediction with loaded model"""
        transaction = {
            "cc_num": 1234567890123456,
            "merchant": "test_store",
            "category": "personal_care",
            "amt": 150.0,
            "first": "John",
            "last": "Doe",
            "gender": "M",
            "street": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip": 62701,
            "lat": 39.7817,
            "long": -89.6501,
            "city_pop": 116250,
            "job": "Engineer",
            "dob": "1980-01-15",
            "trans_num": "tx_test_001",
            "merch_lat": 39.7817,
            "merch_long": -89.6501,
            "is_fraud": 0,
            "current_time": 1702310400
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        print(f"✅ Prediction: {data['prediction']}")



# import sys
# import os
# import pytest
# from pathlib import Path
# from fastapi.testclient import TestClient
# from datetime import datetime
# from dotenv import load_dotenv, find_dotenv
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Setup paths
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# sys.path.append('./Bloc4 - Final project/Projet/Architecture')

# # Load environment
# dotenv_path = find_dotenv()
# load_dotenv(".env")

# print(f"✅ .env chargé dans test_predict.py")
# print(f"   TesVAr : {os.getenv('MAIL_FROM_NAME')}")
# print(f"Fichier .env trouvé : {dotenv_path}")

# from App.Dockers.fastapi.main import app, getMyModel, getModelRunID


# class TestIntegration:
#     """Test d'intégration complet de l'application FastAPI"""
    
#     def test_01_app_startup(self):
#         """Test 1: Simple test that doesn't need model"""
#         assert True, "Basic test should pass"
    
#     @pytest.mark.skipif(
#         not pytest.importorskip("sys").modules.get('App.Dockers.fastapi.main'),
#         reason="FastAPI app not available"
#     )
#     def test_02_with_model(self, model_available):
#         """Test 2: Test that requires model"""
#         if not model_available:
#             pytest.skip("Model not available")
        
#         assert True, "Model-dependent test"
    
#     @pytest.fixture(scope="class", autouse=True)
#     def setup_app(self):
#         """Initialize app with loaded model before all tests"""
#         print("\n🚀 Setting up integration test...")
#         model = getMyModel()
        
#         if model is None:
#             pytest.exit("❌ Impossible de charger le modèle MLflow")
        
#         app.state.loaded_model = model
#         print("✅ Model loaded in app.state")
#         yield
#         print("✅ Integration test complete")
    
#     @pytest.fixture
#     def client(self):
#         """Create test client"""
#         return TestClient(app)
    
#     def test_01_app_startup_and_health(self, client):
#         """Test 1: Vérifier que l'app démarre et que le health check fonctionne"""
#         response = client.get("/health")
        
#         assert response.status_code == 200, f"Expected 200, got {response.status_code}"
#         data = response.json()
        
#         assert "status" in data, "Missing 'status' field"
#         assert data["status"] == "healthy", f"Expected status='healthy', got {data['status']}"
#         assert "model_loaded" in data, "Missing 'model_loaded' field"
#         assert data["model_loaded"] is True, "Model should be loaded"
        
#         print(f"✅ Health check passed: {data}")
    
#     # def test_02_root_endpoint(self, client):
#     #     """Test 2: Vérifier que l'endpoint root retourne les métadonnées"""
#     #     response = client.get("/")
        
#     #     assert response.status_code == 200
#     #     data = response.json()
        
#     #     assert "message" in data
#     #     assert "version" in data
#     #     assert "endpoints" in data
#     #     assert "health" in data["endpoints"]
#     #     assert "predict" in data["endpoints"]
        
#     #     print(f"✅ Root endpoint working: {data['message']}")
    
#     # def test_03_model_registry_loaded(self, client):
#     #     """Test 3: Vérifier que le modèle est correctement enregistré"""
#     #     model_run_id = getModelRunID()
        
#     #     assert model_run_id is not None, "Model RunID should not be None"
#     #     assert hasattr(model_run_id, 'name'), "Model should have 'name' attribute"
#     #     assert hasattr(model_run_id, 'latest_versions'), "Model should have 'latest_versions' attribute"
        
#     #     print(f"✅ Model registry: {model_run_id.name} (v{model_run_id.latest_versions[0].version})")
    
#     # def test_04_predict_fraud_transaction(self, client):
#     #     """Test 4: Prédiction pour une transaction suspecte (fraude)"""
#     #     fraud_transaction = {
#     #         "cc_num": 9999567890123456,
#     #         "merchant": "fraud_Suspect_Store",
#     #         "category": "personal_care",
#     #         "amt": 9999.0,
#     #         "first": "Jane",
#     #         "last": "Smith",
#     #         "gender": "F",
#     #         "street": "456 Elm St",
#     #         "city": "Lagos",
#     #         "state": "NG",
#     #         "zip": 12345,
#     #         "lat": 6.5244,
#     #         "long": 3.3792,
#     #         "city_pop": 21000000,
#     #         "job": "Unknown",
#     #         "dob": "1990-05-20",
#     #         "trans_num": "tx_fraud_001",
#     #         "merch_lat": 6.5244,
#     #         "merch_long": 3.3792,
#     #         "is_fraud": 1,
#     #         "current_time": 1702396800
#     #     }
        
#     #     response = client.post("/predict", json=fraud_transaction)
        
#     #     assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
#     #     data = response.json()
        
#     #     assert "prediction" in data, f"Missing 'prediction' field in {data}"
#     #     assert "is_fraud" in data, f"Missing 'is_fraud' field in {data}"
#     #     assert isinstance(data["is_fraud"], bool), f"'is_fraud' should be bool, got {type(data['is_fraud'])}"
#     #     assert "trans_num" in data
#     #     assert data["trans_num"] == "tx_fraud_001"
        
#     #     print(f"✅ Fraud prediction: {data['prediction']} (is_fraud={data['is_fraud']})")
    
#     # def test_05_predict_legitimate_transaction(self, client):
#     #     """Test 5: Prédiction pour une transaction légitime"""
#     #     legitimate_transaction = {
#     #         "cc_num": 1234567890123456,
#     #         "merchant": "trusted_Kirlin and Sons",
#     #         "category": "personal_care",
#     #         "amt": 50.0,
#     #         "first": "John",
#     #         "last": "Doe",
#     #         "gender": "M",
#     #         "street": "123 Main St",
#     #         "city": "Springfield",
#     #         "state": "IL",
#     #         "zip": 62701,
#     #         "lat": 39.7817,
#     #         "long": -89.6501,
#     #         "city_pop": 116250,
#     #         "job": "Engineer",
#     #         "dob": "1980-01-15",
#     #         "trans_num": "tx_legit_001",
#     #         "merch_lat": 39.7817,
#     #         "merch_long": -89.6501,
#     #         "is_fraud": 0,
#     #         "current_time": 1702396800
#     #     }
        
#     #     response = client.post("/predict", json=legitimate_transaction)
        
#     #     assert response.status_code == 200, f"Expected 200, got {response.status_code}"
#     #     data = response.json()
        
#     #     assert "prediction" in data
#     #     assert "is_fraud" in data
#     #     assert isinstance(data["is_fraud"], bool)
#     #     assert data["trans_num"] == "tx_legit_001"
        
#     #     print(f"✅ Legitimate prediction: {data['prediction']} (is_fraud={data['is_fraud']})")
    
#     # def test_06_predict_response_contains_metadata(self, client):
#     #     """Test 6: Vérifier que la réponse contient toutes les métadonnées"""
#     #     transaction = {
#     #         "cc_num": 5555567890123456,
#     #         "merchant": "test_store",
#     #         "category": "grocery_pos",
#     #         "amt": 100.0,
#     #         "first": "Test",
#     #         "last": "User",
#     #         "gender": "M",
#     #         "street": "789 Test Ave",
#     #         "city": "New York",
#     #         "state": "NY",
#     #         "zip": 10001,
#     #         "lat": 40.7128,
#     #         "long": -74.0060,
#     #         "city_pop": 8000000,
#     #         "job": "Developer",
#     #         "dob": "1985-03-10",
#     #         "trans_num": "tx_metadata_001",
#     #         "merch_lat": 40.7128,
#     #         "merch_long": -74.0060,
#     #         "is_fraud": 0,
#     #         "current_time": 1702396800
#     #     }
        
#     #     response = client.post("/predict", json=transaction)
        
#     #     assert response.status_code == 200
#     #     data = response.json()
        
#     #     # Vérifier tous les champs de réponse
#     #     required_fields = ["prediction", "is_fraud", "trans_num", "amt", "category"]
#     #     for field in required_fields:
#     #         assert field in data, f"Missing required field: {field}"
        
#     #     # Vérifier que les métadonnées correspondent
#     #     assert data["amt"] == 100.0
#     #     assert data["category"] == "grocery_pos"
#     #     assert data["trans_num"] == "tx_metadata_001"
#     #     assert isinstance(data["prediction"], int)
#     #     assert data["prediction"] in [0, 1]
        
#     #     print(f"✅ Response metadata valid: {data}")


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "--tb=short"])



# # def test_prediction_logged_in_database():
# #     """
# #     Conformité :
# #     - PCI-DSS Req. 10 : Audit trail obligatoire
# #     - RGPD Article 22 : Droit à l'explication
# #     """
# #     # Appeler l'API avec une transaction test
# #     response = client.post("/predict", json=test_transaction)
    
# #     # Vérifier l'enregistrement en base
# #     log_entry = db.query("SELECT * FROM predictions_log WHERE transaction_id = ?")
    
# #     assert log_entry is not None, "Prédiction non tracée"
# #     assert log_entry['model_version'] is not None
# #     assert log_entry['fraud_score'] is not None
# #     assert log_entry['input_features'] is not None  # Pour explicabilité
# #     assert log_entry['response_time_ms'] is not None

# # def test_log_retention_policy():
#     # """
#     # RGPD Article 5.1.e : Limitation de conservation
#     # Politique : Purge automatique après 2 ans
#     # """
#     # old_logs = db.query("SELECT COUNT(*) FROM predictions_log WHERE timestamp < NOW() - INTERVAL '2 years'")
#     # assert old_logs == 0, "Logs anciens non purgés (violation RGPD)"
