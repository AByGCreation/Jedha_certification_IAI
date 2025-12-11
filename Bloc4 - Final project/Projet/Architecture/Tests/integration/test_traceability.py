def test_prediction_logged_in_database():
    """
    Conformité :
    - PCI-DSS Req. 10 : Audit trail obligatoire
    - RGPD Article 22 : Droit à l'explication
    """
    # Appeler l'API avec une transaction test
    response = client.post("/predict", json=test_transaction)
    
    # Vérifier l'enregistrement en base
    log_entry = db.query("SELECT * FROM predictions_log WHERE transaction_id = ?")
    
    assert log_entry is not None, "Prédiction non tracée"
    assert log_entry['model_version'] is not None
    assert log_entry['fraud_score'] is not None
    assert log_entry['input_features'] is not None  # Pour explicabilité
    assert log_entry['response_time_ms'] is not None

def test_log_retention_policy():
    """
    RGPD Article 5.1.e : Limitation de conservation
    Politique : Purge automatique après 2 ans
    """
    old_logs = db.query("SELECT COUNT(*) FROM predictions_log WHERE timestamp < NOW() - INTERVAL '2 years'")
    assert old_logs == 0, "Logs anciens non purgés (violation RGPD)"