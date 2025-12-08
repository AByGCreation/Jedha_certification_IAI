"""
Flask Transaction Simulator with Real-time Fraud Detection
Simulates credit card transactions and instantly detects fraud using MLFlow model
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import mlflow
import requests
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:4000")
MODEL_URI = os.getenv("MODEL_URI", "runs:/bf781e6a105445afa07d064f8f1f30a3/fraud_detector")
BACKEND_STORE_URI = os.getenv("BACKEND_STORE_URI")

# Set MLFlow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model at startup
model = None

def load_fraud_model():
    """Load the fraud detection model from MLFlow"""
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"✅ Model loaded successfully from {MODEL_URI}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Transaction categories
CATEGORIES = [
    'gas_transport', 'grocery_pos', 'home', 'grocery_net', 'shopping_pos',
    'misc_pos', 'entertainment', 'food_dining', 'personal_care',
    'health_fitness', 'misc_net', 'shopping_net', 'kids_pets', 'travel'
]

# Sample merchant names
MERCHANTS = [
    'fraud_Rippin, Kub and Mann', 'fraud_Heller, Gutmann and Zieme',
    'fraud_Lind-Buckridge', 'fraud_Kirlin and Sons', 'fraud_Sporer-Keebler',
    'fraud_Haley Group', 'fraud_Johnston-Casper', 'fraud_Barrows-Heller'
]

# Sample US cities with coordinates
CITIES = [
    {'city': 'New York', 'state': 'NY', 'lat': 40.7128, 'long': -74.0060, 'pop': 8336817},
    {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0522, 'long': -118.2437, 'pop': 3979576},
    {'city': 'Chicago', 'state': 'IL', 'lat': 41.8781, 'long': -87.6298, 'pop': 2693976},
    {'city': 'Houston', 'state': 'TX', 'lat': 29.7604, 'long': -95.3698, 'pop': 2320268},
    {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.4484, 'long': -112.0740, 'pop': 1680992},
    {'city': 'Miami', 'state': 'FL', 'lat': 25.7617, 'long': -80.1918, 'pop': 467963},
]

def generate_random_transaction(is_fraud=False):


    # Select city
    city = random.choice(CITIES)

    # Generate transaction details
    category = random.choice(CATEGORIES)
    merchant = random.choice(MERCHANTS)

    # Generate amount (fraud transactions tend to be higher)
    if is_fraud:
        amt = round(random.uniform(100, 1000), 2)
    else:
        amt = round(random.uniform(5, 200), 2)

    # Customer location (near city center)
    lat = city['lat'] + random.uniform(-0.5, 0.5)
    long = city['long'] + random.uniform(-0.5, 0.5)

    # Merchant location (if fraud, further away)
    if is_fraud:
        merch_lat = city['lat'] + random.uniform(-2, 2)
        merch_long = city['long'] + random.uniform(-2, 2)
    else:
        merch_lat = city['lat'] + random.uniform(-0.2, 0.2)
        merch_long = city['long'] + random.uniform(-0.2, 0.2)

    # Generate timestamp
    trans_date_trans_time = datetime.now()

    # Calculate age (20-70 years old)
    age = random.randint(20, 70)
    dob = datetime.now() - timedelta(days=age*365)

    transaction = {
        'cc_num': random.randint(1000000000000000, 9999999999999999),
        'merchant': merchant,
        'category': category,
        'amt': amt,
        'first': random.choice(['John', 'Jane', 'Bob', 'Alice', 'Mike', 'Sarah']),
        'last': random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']),
        'gender': random.choice(['M', 'F']),
        'street': f"{random.randint(1, 9999)} Main St",
        'city': city['city'],
        'state': city['state'],
        'zip': random.randint(10000, 99999),
        'lat': lat,
        'long': long,
        'city_pop': city['pop'],
        'job': random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager']),
        'dob': dob.strftime('%Y-%m-%d'),
        'trans_date_trans_time': trans_date_trans_time.strftime('%Y-%m-%d %H:%M:%S'),
        'trans_num': f"TXN{random.randint(1000000, 9999999)}",
        'unix_time': int(trans_date_trans_time.timestamp()),
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'is_fraud': 1 if is_fraud else 0
    }

    return transaction

def predict_fraud(transaction_dict):
    """Predict if transaction is fraud using MLFlow model"""
    global model

    if model is None:
        if not load_fraud_model():
            return None, "Model not loaded"

    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_dict])

        # Make prediction
        prediction = model.predict(df)

        return int(prediction[0]), None
    except Exception as e:
        return None, str(e)

def save_transaction_to_db(transaction_dict, prediction):
    """Save transaction with prediction to database"""
    if not BACKEND_STORE_URI:
        return False, "Database not configured"

    try:
        engine = create_engine(BACKEND_STORE_URI)

        # Add prediction to transaction
        transaction_dict['prediction'] = prediction
        transaction_dict['created_at'] = datetime.now()

        # Convert to DataFrame and save
        df = pd.DataFrame([transaction_dict])
        df.to_sql('transactions', engine, if_exists='append', index=False)

        return True, None
    except Exception as e:
        return False, str(e)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mlflow_uri': MLFLOW_TRACKING_URI
    })

@app.route('/simulate', methods=['POST'])
def simulate_transaction():
    """Simulate a transaction and get fraud prediction"""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        force_fraud = data.get('force_fraud', False)

        # Generate transaction
        transaction = generate_random_transaction(is_fraud=force_fraud)

        # Predict fraud
        prediction, error = predict_fraud(transaction)

        if error:
            return jsonify({
                'success': False,
                'error': f"Prediction error: {error}"
            }), 500

        # Save to database
        db_success, db_error = save_transaction_to_db(transaction, prediction)

        # Prepare response
        response = {
            'success': True,
            'transaction': {
                'id': transaction['trans_num'],
                'amount': transaction['amt'],
                'merchant': transaction['merchant'],
                'category': transaction['category'],
                'date': transaction['trans_date_trans_time'],
                'city': transaction['city'],
                'state': transaction['state'],
            },
            'prediction': {
                'is_fraud': bool(prediction == 1),
                'confidence': 'high' if prediction == 1 else 'low',
                'actual_fraud': transaction['is_fraud']
            },
            'saved_to_db': db_success
        }

        if db_error:
            response['db_error'] = db_error

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/simulate/batch', methods=['POST'])
def simulate_batch():
    """Simulate multiple transactions"""
    try:
        data = request.get_json() or {}
        count = data.get('count', 10)
        fraud_rate = data.get('fraud_rate', 0.1)  # 10% fraud by default

        results = []

        for _ in range(count):
            # Decide if this should be fraud
            is_fraud = random.random() < fraud_rate

            # Generate transaction
            transaction = generate_random_transaction(is_fraud=is_fraud)

            # Predict
            prediction, error = predict_fraud(transaction)

            if error:
                continue

            # Save to database
            save_transaction_to_db(transaction, prediction)

            results.append({
                'id': transaction['trans_num'],
                'amount': transaction['amt'],
                'category': transaction['category'],
                'actual_fraud': transaction['is_fraud'],
                'predicted_fraud': prediction
            })

        # Calculate accuracy
        correct = sum(1 for r in results if r['actual_fraud'] == r['predicted_fraud'])
        accuracy = (correct / len(results)) * 100 if results else 0

        return jsonify({
            'success': True,
            'count': len(results),
            'accuracy': round(accuracy, 2),
            'transactions': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load model at startup
    load_fraud_model()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
