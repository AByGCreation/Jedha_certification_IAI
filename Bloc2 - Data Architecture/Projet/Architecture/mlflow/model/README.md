# Pre-trained Fraud Detection Model

This directory contains a pre-trained logistic regression model for fraud detection.

## Model Details
- Algorithm: Logistic Regression
- Features: 10 engineered features
- Accuracy: ~85% (on test set)
- Fraud Detection Threshold: 70%

## Features Used:
1. amount - Transaction amount
2. hour - Hour of day (0-23)
3. velocity_1h - Transactions in last hour
4. avg_amount_user - User's average transaction
5. user_total_tx - User's total transactions
6. amount_deviation - Deviation from user average
7. high_amount - Binary flag (amount > 500)
8. unusual_hour - Binary flag (3 AM - 6 AM)
9. high_velocity - Binary flag (velocity >= 5)
10. new_user - Binary flag (total_tx < 5)

## Usage
The model is automatically loaded by FastAPI on startup via the FraudDetector class.
