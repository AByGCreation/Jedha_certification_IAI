import pytest

class TestModelQuality:
    """Test de qualité du modèle MLflow"""
    
    def test_model_loaded(self, require_model, model):
        """Test 1: Model is loaded"""
        assert model is not None, "Model should be loaded"
        assert hasattr(model, 'predict'), "Model should have predict method"
    
    def test_model_can_predict(self, require_model, model):
        """Test 2: Model can make predictions"""
        import pandas as pd
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'cc_num': [1234567890123456],
            'merchant': ['test'],
            'category': ['test'],
            'amt': [100.0],
            'first': ['John'],
            'last': ['Doe'],
            'gender': ['M'],
            'street': ['123 Main'],
            'city': ['NYC'],
            'state': ['NY'],
            'zip': [10001],
            'lat': [40.7128],
            'long': [-74.0060],
            'city_pop': [8000000],
            'job': ['Engineer'],
            'dob': ['1990-01-01'],
            'trans_num': ['tx_001'],
            'merch_lat': [40.7128],
            'merch_long': [-74.0060],
            'is_fraud': [0],
            'current_time': [1702396800]
        })
        
        from App.Dockers.fastapi.main import Preprocessor
        
        preprocessed = Preprocessor(dummy_data.copy())
        prediction = model.predict(preprocessed)
        
        assert prediction is not None, "Prediction should not be None"
        assert len(prediction) > 0, "Prediction should have results"
        assert prediction[0] in [0, 1], "Prediction should be binary"