def test_model_accuracy_threshold():
    """
    Seuil métier : Accuracy ≥ 92%
    Justification : Basé sur analyse coût/bénéfice
    - Faux négatif (fraude non détectée) : 150€ de perte moyenne
    - Faux positif (client bloqué) : 5€ de friction + risque churn
    """
    model = mlflow.sklearn.load_model("models:/fraud-detection/production")
    X_test, y_test = load_reference_dataset()
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy >= 0.92, f"Accuracy trop faible: {accuracy:.3f}"

def test_model_f1_score_threshold():
    """
    Seuil métier : F1-Score ≥ 85%
    Justification : Équilibre précision/rappel critique en détection fraude
    """
    f1 = f1_score(y_test, y_pred)
    assert f1 >= 0.85, f"F1-score trop faible: {f1:.3f}"