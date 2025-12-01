"""
Flask Web Interface for Stripe Fraud Detection
Bootstrap UI for transaction simulation
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import logging
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route("/api/simulate", methods=["POST"])
def simulate_transaction():
    """Simulate a transaction"""
    try:
        data = request.json
        
        # Call FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/api/transactions",
            json=data,
            timeout=30
        )
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling FastAPI: {e}")
        return jsonify({"error": "Backend service unavailable"}), 503
    except Exception as e:
        logger.error(f"Error simulating transaction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/transactions/<transaction_id>")
def get_transaction(transaction_id):
    """Get transaction details"""
    try:
        response = requests.get(
            f"{FASTAPI_URL}/api/transactions/{transaction_id}",
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error fetching transaction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/transactions")
def list_transactions():
    """List recent transactions"""
    try:
        limit = request.args.get("limit", 20, type=int)
        offset = request.args.get("offset", 0, type=int)
        is_fraud = request.args.get("is_fraud", None)
        
        params = {"limit": limit, "offset": offset}
        if is_fraud is not None:
            params["is_fraud"] = is_fraud.lower() == "true"
        
        # Increased timeout for large transaction queries
        response = requests.get(
            f"{FASTAPI_URL}/api/transactions",
            params=params,
            timeout=60
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error listing transactions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def get_status():
    """Get backend status"""
    try:
        # Increased timeout for slow status queries
        response = requests.get(f"{FASTAPI_URL}/status", timeout=90)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        logger.error("FastAPI status check timed out after 90 seconds")
        return jsonify({"error": "Backend status check timeout"}), 504
    except Exception as e:
        logger.error(f"Error checking backend status: {e}")
        return jsonify({"error": "Backend unavailable"}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
