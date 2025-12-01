"""
Train a simple fraud detection model
Pre-trained logistic regression model for demo
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Generate synthetic training data
np.random.seed(42)
n_samples = 1000

# Features: amount, hour, velocity, avg_amount_user, total_tx, amount_deviation, 
#           high_amount, unusual_hour, high_velocity, new_user
X = np.random.randn(n_samples, 10)

# Labels: fraud (1) or not (0) - ~15% fraud rate
y = np.random.binomial(1, 0.15, n_samples)

# Train logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('/mlflow/model', exist_ok=True)
joblib.dump(model, '/mlflow/model/logistic_regression_fraud.pkl')

print(f'Model trained! Accuracy: {model.score(X_test, y_test):.2%}')
print('Model saved to /mlflow/model/logistic_regression_fraud.pkl')
