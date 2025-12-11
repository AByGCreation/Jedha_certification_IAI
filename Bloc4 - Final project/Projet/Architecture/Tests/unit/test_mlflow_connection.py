# test_mlflow_connection.py
import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = mlflow.MlflowClient()

print("🔍 Connexion MLflow...")
print(f"   URI: {os.getenv('MLFLOW_TRACKING_URI')}")

# Lister tous les modèles
models = client.search_registered_models()
print(f"\n📦 Nombre de modèles enregistrés : {len(models)}")

if models:
    for model in models:
        print(f"\n  Modèle: {model.name}")
        print(f"  Versions: {len(model.latest_versions)}")
else:
    print("\n❌ AUCUN MODÈLE TROUVÉ dans le registre MLflow")
    print("   → Vérifie que tu as bien enregistré un modèle avec mlflow.register_model()")