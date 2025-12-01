"""
Fraud Detection ML Model
Pre-trained Logistic Regression model for fraud detection
"""

import joblib
import numpy as np
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FraudDetector:
    """Fraud detection using pre-trained ML model"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            model_path = "/mlflow/model/logistic_regression_fraud.pkl"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("✅ ML model loaded successfully")
            else:
                logger.warning("⚠️ No ML model found, using rule-based detection")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, amount: float, hour: int, velocity_1h: int, 
                avg_amount_user: float, user_total_tx: int) -> Dict[str, Any]:
        """
        Predict fraud probability
        
        Features:
        - amount: Transaction amount
        - hour: Hour of day (0-23)
        - velocity_1h: Number of transactions in last hour
        - avg_amount_user: User's average transaction amount
        - user_total_tx: User's total transaction count
        """
        
        # Calculate features
        features = self._engineer_features(amount, hour, velocity_1h, avg_amount_user, user_total_tx)
        
        if self.model:
            # Use ML model
            try:
                X = np.array([features])
                probability = self.model.predict_proba(X)[0][1]  # Probability of fraud class
                fraud_score = int(probability * 100)
                is_fraud = probability >= 0.70  # 70% threshold
                
                return {
                    "fraud_score": fraud_score,
                    "probability": probability,
                    "is_fraud": is_fraud,
                    "model": "logistic_regression"
                }
            except Exception as e:
                logger.error(f"ML prediction error: {e}")
                return self._rule_based_detection(amount, hour, velocity_1h, avg_amount_user, user_total_tx)
        else:
            # Fallback to rule-based
            return self._rule_based_detection(amount, hour, velocity_1h, avg_amount_user, user_total_tx)
    
    def _engineer_features(self, amount, hour, velocity_1h, avg_amount_user, user_total_tx):
        """Engineer features for ML model"""
        
        # Amount deviation from user average
        if avg_amount_user > 0:
            amount_deviation = (amount - avg_amount_user) / avg_amount_user
        else:
            amount_deviation = 0
        
        # High amount flag
        high_amount = 1 if amount > 500 else 0
        
        # Unusual hour (3 AM - 6 AM)
        unusual_hour = 1 if 3 <= hour <= 6 else 0
        
        # High velocity
        high_velocity = 1 if velocity_1h >= 5 else 0
        
        # New user
        new_user = 1 if user_total_tx < 5 else 0
        
        # Feature vector (must match training features)
        features = [
            amount,
            hour,
            velocity_1h,
            avg_amount_user,
            user_total_tx,
            amount_deviation,
            high_amount,
            unusual_hour,
            high_velocity,
            new_user
        ]
        
        return features
    
    def _rule_based_detection(self, amount, hour, velocity_1h, avg_amount_user, user_total_tx):
        """Rule-based fraud detection as fallback"""
        
        fraud_score = 0
        
        # Rule 1: High amount (>500)
        if amount > 500:
            fraud_score += 30
        
        # Rule 2: Very high amount (>1000)
        if amount > 1000:
            fraud_score += 20
        
        # Rule 3: Unusual amount for user
        if avg_amount_user > 0 and amount > avg_amount_user * 3:
            fraud_score += 25
        
        # Rule 4: High velocity
        if velocity_1h >= 5:
            fraud_score += 35
        elif velocity_1h >= 3:
            fraud_score += 20
        
        # Rule 5: Unusual hour
        if 3 <= hour <= 6:
            fraud_score += 15
        
        # Rule 6: New user with high amount
        if user_total_tx < 5 and amount > 300:
            fraud_score += 20
        
        is_fraud = fraud_score >= 70
        probability = min(fraud_score / 100.0, 1.0)
        
        return {
            "fraud_score": min(fraud_score, 100),
            "probability": probability,
            "is_fraud": is_fraud,
            "model": "rule_based"
        }
    
    def close(self):
        """Cleanup resources"""
        self.model = None
        logger.info("Fraud detector closed")
