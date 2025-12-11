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

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Add the parent directory to the path so src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.append('./Bloc4 - Final project/Projet/Architecture')

#important ! ne pas deplacer avant le sys.path.append
from App.Dockers.fastapi.main import app, getMyModel, getModelRunID


# Trouve le fichier .env
dotenv_path = find_dotenv()
load_dotenv(".env")  # Depuis le répertoire courant

print(f"✅ .env chargé dans test_predict.py")
print(f"   TesVAr : {os.getenv('MAIL_FROM_NAME')}")
print(f"Fichier .env trouvé : {dotenv_path}")
    

#=======================================
# FIXTURE POUR CHARGER LE MODÈLE AVANT LES TESTS
#=======================================

@pytest.fixture(scope="session", autouse=True)
def setup_model_for_tests():
    """Force le chargement du modèle dans l'app avant les tests"""

    print("\n🤖 Setup : Chargement du modèle...")
    model = getMyModel()
    
    if model is None:
        pytest.exit("Aucun modèle MLflow disponible. Lance : python register_test_model.py")
    else:
        print(f"✅ Modèle chargé : {model}")

    
    # Forcer la variable globale
    app.state.loaded_model = model
    
    print("✅ Modèle injecté dans l'app\n")
    yield

#=======================================
# TESTS
#=======================================

client = TestClient(app)

def test_fastapi_health():
    """Test que l'endpoint /health retourne 200"""
    response = client.get("/health")
    assert response.status_code == 200

def test_getModelRunID():
    """Test que l'endpoint /getModelRunID retourne un run_id"""

    
    # Appeler la fonction
    best_model = getModelRunID()
    
    # Vérifications
    assert best_model is not None, "Aucun modèle trouvé"
    assert hasattr(best_model, 'name'), "Le modèle n'a pas d'attribut 'name'"
    assert hasattr(best_model, 'latest_versions'), "Pas de versions"
    assert len(best_model.latest_versions) > 0, "Aucune version disponible"
    
    print(f"✅ Modèle trouvé : {best_model.name}")
    print(f"✅ Version : {best_model.latest_versions[0].version}")
    print(f"✅ Run ID : {best_model.latest_versions[0].run_id}")
    
# def test_predict_returns_200():

#     """Test que l'endpoint /predict retourne 200"""
#     response = client.post("/predict", json={
#         "cc_num": 1234567890123456,
#         "merchant": "fraud_Kirlin and Sons",
#         "category": "personal_care",
#         "amt": 150.0,
#         "first": "John",
#         "last": "Doe",
#         "gender": "M",
#         "street": "123 Main St",
#         "city": "Springfield",
#         "state": "IL",
#         "zip": 62701,
#         "lat": 39.7817,
#         "long": -89.6501,
#         "city_pop": 116250,
#         "job": "Engineer",
#         "dob": "1980-01-15",
#         "trans_num": "tx_test_001",
#         "merch_lat": 39.7817,
#         "merch_long": -89.6501,
#         "is_fraud": 0,
#         "current_time": 1702310400
#     })
#     assert response.status_code == 200



def test_predict_returns_prediction():
    """Test que la réponse contient une prédiction"""
    response = client.post("/predict", json={
        "cc_num": 9999567890123456,
        "merchant": "fraud_Suspect_Store",
        "category": "personal_care",
        "amt": 9999.0,
        "first": "Jane",
        "last": "Smith",
        "gender": "F",
        "street": "456 Elm St",
        "city": "Lagos",
        "state": "NG",
        "zip": 12345,
        "lat": 6.5244,
        "long": 3.3792,
        "city_pop": 21000000,
        "job": "Unknown",
        "dob": "1990-05-20",
        "trans_num": "tx_test_002",
        "merch_lat": 6.5244,
        "merch_long": 3.3792,
        "is_fraud": 1,
        "current_time": 1702396800
    })
    data = response.json()

    print(f"Response data: {data}")

   # assert "prediction" in data
   # assert data["prediction"] in [0, 1]
    assert "is_fraud" in data
    assert isinstance(data["is_fraud"], bool)



# @pytest.fixture(scope="session", autouse=True)
# def generate_test_report(request):
#     """Generate a custom test report after all tests run"""
#     yield
    
#     # Collect test results
#     report_data = {
#         "timestamp": datetime.now().isoformat(),
#         "total_tests": len(request.session.items),
#         "passed": 0,
#         "failed": 0,
#         "errors": 0,
#         "skipped": 0,
#         "test_details": []
#     }
    
#     for item in request.session.items:
#         outcome = request.node.stash.get(item, None)
#         report_data["test_details"].append({
#             "test": item.name,
#             "status": "passed" if outcome else "failed"
#         })
    
#     # Save report
#     report_path = Path("test_results/test_report.json")
#     with open(report_path, "w") as f:
#         json.dump(report_data, f, indent=2)
    
#     print(f"\n📊 Test report saved to: {report_path}")









# def test_predict_high_amount_suspicious():
    # """Test qu'un montant élevé est considéré comme suspect"""
    # response = client.post("/predict", json={
    #     "cc_num": 5555567890123456,
    #     "merchant": "fraud_HighRisk",
    #     "category": "shopping_net",
    #     "amt": 15000.0,
    #     "first": "Bob",
    #     "last": "Fraudster",
    #     "gender": "M",
    #     "street": "789 Shady Lane",
    #     "city": "Risky City",
    #     "state": "TX",
    #     "zip": 75001,
    #     "lat": 32.7767,
    #     "long": -96.7970,
    #     "city_pop": 1340000,
    #     "job": "Scammer",
    #     "dob": "1985-03-10",
    #     "trans_num": "tx_test_003",
    #     "merch_lat": 32.7767,
    #     "merch_long": -96.7970,
    #     "is_fraud": 1,
    #     "current_time": 1702483200
    # })
    # data = response.json()
    # assert response.status_code == 200
    # assert "prediction" in data