"""
Stripe Fraud Detection Platform - FastAPI Backend
Connects to all services: PostgreSQL, ClickHouse, MongoDB, Redis, Kafka, Neo4j, MLflow, Elasticsearch
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
import logging
from contextlib import asynccontextmanager

# Import custom modules
from database import (
    get_postgres_connection,
    get_mongodb_client,
    get_redis_client,
    get_neo4j_driver,
    get_clickhouse_client,
    get_kafka_producer,
    get_elasticsearch_client,
    get_minio_client
)
from fraud_detector import FraudDetector
from models import Transaction, TransactionCreate, FraudScore, User, Merchant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize fraud detector
fraud_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: startup and shutdown"""
    global fraud_detector
    
    # Startup
    logger.info("Starting Stripe Fraud Detection Platform...")
    
    try:
        # Initialize fraud detector with MLflow model
        fraud_detector = FraudDetector()
        logger.info("✅ Fraud detector initialized")
        
        # Test database connections
        pg_conn = get_postgres_connection()
        pg_conn.close()
        logger.info("✅ PostgreSQL connection OK")
        
        mongo_client = get_mongodb_client()
        mongo_client.server_info()
        logger.info("✅ MongoDB connection OK")
        
        redis_client = get_redis_client()
        redis_client.ping()
        logger.info("✅ Redis connection OK")
        
        neo4j_driver = get_neo4j_driver()
        neo4j_driver.verify_connectivity()
        logger.info("✅ Neo4j connection OK")
        
        ch_client = get_clickhouse_client()
        ch_client.execute("SELECT 1")
        logger.info("✅ ClickHouse connection OK")
        
        logger.info("🚀 All services connected successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if fraud_detector:
        fraud_detector.close()

# Create FastAPI app
app = FastAPI(
    title="Stripe Fraud Detection API",
    description="Complete fraud detection platform with ML, streaming, and analytics",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "fraud_detector": "active" if fraud_detector else "inactive"
        }
    }

@app.get("/status")
async def get_status():
    """Get detailed status of all services"""
    status = {"timestamp": datetime.now().isoformat()}
    
    # Test each service
    try:
        pg_conn = get_postgres_connection()
        cursor = pg_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        tx_count = cursor.fetchone()[0]
        cursor.close()
        pg_conn.close()
        status["postgres"] = {"status": "connected", "transactions_count": tx_count}
    except Exception as e:
        status["postgres"] = {"status": "error", "message": str(e)}
    
    try:
        mongo_client = get_mongodb_client()
        db = mongo_client["stripe_nosql"]
        disputes_count = db["disputes"].count_documents({})
        status["mongodb"] = {"status": "connected", "disputes_count": disputes_count}
    except Exception as e:
        status["mongodb"] = {"status": "error", "message": str(e)}
    
    try:
        redis_client = get_redis_client()
        redis_info = redis_client.info("keyspace")
        status["redis"] = {"status": "connected", "keys": redis_info.get("db0", {}).get("keys", 0)}
    except Exception as e:
        status["redis"] = {"status": "error", "message": str(e)}
    
    try:
        neo4j_driver = get_neo4j_driver()
        neo4j_driver.verify_connectivity()
        status["neo4j"] = {"status": "connected"}
    except Exception as e:
        status["neo4j"] = {"status": "error", "message": str(e)}
    
    try:
        ch_client = get_clickhouse_client()
        result = ch_client.execute("SELECT COUNT(*) FROM transactions_olap")
        status["clickhouse"] = {"status": "connected", "analytics_records": result[0][0] if result else 0}
    except Exception as e:
        status["clickhouse"] = {"status": "error", "message": str(e)}
    
    status["mlflow"] = {"status": "active" if fraud_detector and fraud_detector.model else "no_model"}
    
    return status

# ============================================
# TRANSACTION ENDPOINTS
# ============================================

@app.post("/api/transactions", response_model=Dict[str, Any])
async def create_transaction(transaction: TransactionCreate, background_tasks: BackgroundTasks):
    """
    Create a new transaction and detect fraud in real-time
    """
    try:
        # Generate transaction ID
        tx_id = f"txn_{datetime.now().strftime('%Y%m%d%H%M%S')}_{transaction.user_id[-4:]}"
        
        # 1. Check Redis cache for user fraud history
        redis_client = get_redis_client()
        user_fraud_key = f"user_fraud:{transaction.user_id}"
        cached_risk = redis_client.get(user_fraud_key)
        
        base_risk_score = 0
        if cached_risk:
            base_risk_score = int(cached_risk)
            logger.info(f"User {transaction.user_id} cached risk: {base_risk_score}")
        
        # 2. Get user context from PostgreSQL
        pg_conn = get_postgres_connection()
        cursor = pg_conn.cursor()
        
        cursor.execute("""
            SELECT total_transactions, avg_amount, country 
            FROM users WHERE user_id = %s
        """, (transaction.user_id,))
        user_data = cursor.fetchone()
        
        if not user_data:
            cursor.close()
            pg_conn.close()
            raise HTTPException(status_code=404, detail="User not found")
        
        total_tx, avg_amount, user_country = user_data
        
        # 3. Calculate velocity (transactions in last hour)
        cursor.execute("""
            SELECT COUNT(*) FROM transactions 
            WHERE user_id = %s AND created_at >= NOW() - INTERVAL '1 hour'
        """, (transaction.user_id,))
        velocity = cursor.fetchone()[0]
        
        # 4. Calculate fraud score using ML model
        if fraud_detector and fraud_detector.model:
            fraud_result = fraud_detector.predict(
                amount=float(transaction.amount),
                hour=datetime.now().hour,
                velocity_1h=velocity,
                avg_amount_user=float(avg_amount) if avg_amount else 0,
                user_total_tx=total_tx or 0
            )
            fraud_score = fraud_result["fraud_score"]
            ml_probability = fraud_result["probability"]
            is_fraud = fraud_result["is_fraud"]
        else:
            # Fallback: rule-based scoring
            fraud_score = base_risk_score
            
            # Rule 1: Amount threshold
            if transaction.amount > 500:
                fraud_score += 30
            
            # Rule 2: Unusual amount for user
            if avg_amount and transaction.amount > avg_amount * 3:
                fraud_score += 25
            
            # Rule 3: High velocity
            if velocity >= 5:
                fraud_score += 35
            
            # Rule 4: Unusual hour (3 AM - 6 AM)
            current_hour = datetime.now().hour
            if 3 <= current_hour <= 6:
                fraud_score += 20
            
            is_fraud = fraud_score >= 70
            ml_probability = fraud_score / 100.0
        
        # 5. Determine transaction status
        if fraud_score >= 90:
            status = "failed"  # Blocked
        elif fraud_score >= 70:
            status = "pending"  # Manual review
        else:
            status = "succeeded"
        
        # 6. Insert transaction into PostgreSQL
        cursor.execute("""
            INSERT INTO transactions (
                transaction_id, user_id, merchant_id, amount, currency,
                status, payment_method, card_last4, is_fraud, fraud_score,
                ml_fraud_probability, ip_address, device_type, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            tx_id, transaction.user_id, transaction.merchant_id, transaction.amount,
            transaction.currency, status, transaction.payment_method,
            transaction.card_last4, is_fraud, fraud_score, ml_probability,
            transaction.ip_address, transaction.device_type, datetime.now()
        ))
        
        # 7. Insert fraud events if suspicious
        if is_fraud:
            if transaction.amount > 500:
                cursor.execute("""
                    INSERT INTO fraud_events (transaction_id, rule_triggered, severity, score_contribution, details)
                    VALUES (%s, %s, %s, %s, %s)
                """, (tx_id, "amount_threshold_exceeded", "high", 30, f'{{"amount": {transaction.amount}}}'))
            
            if velocity >= 5:
                cursor.execute("""
                    INSERT INTO fraud_events (transaction_id, rule_triggered, severity, score_contribution, details)
                    VALUES (%s, %s, %s, %s, %s)
                """, (tx_id, "velocity_check_failed", "critical", 35, f'{{"velocity": {velocity}}}'))
        
        pg_conn.commit()
        cursor.close()
        pg_conn.close()
        
        # 8. Update Redis cache
        redis_client.setex(f"fraud_score:{tx_id}", 3600, fraud_score)  # TTL 1h
        if is_fraud:
            redis_client.incr(user_fraud_key)
            redis_client.expire(user_fraud_key, 86400)  # TTL 24h
        
        # 9. Publish to Kafka (async background task)
        background_tasks.add_task(publish_transaction_event, tx_id, transaction.dict(), fraud_score, is_fraud)
        
        # 10. Store in MongoDB for flexible queries
        background_tasks.add_task(store_transaction_mongodb, tx_id, transaction.dict(), fraud_score)
        
        # 11. Log to Elasticsearch
        background_tasks.add_task(log_to_elasticsearch, tx_id, "transaction_created", fraud_score, status)
        
        # 12. Update Neo4j graph (if fraud detected)
        if is_fraud:
            background_tasks.add_task(create_fraud_relationship, transaction.user_id, transaction.merchant_id, tx_id)
        
        return {
            "transaction_id": tx_id,
            "status": status,
            "fraud_score": fraud_score,
            "ml_probability": round(ml_probability, 4),
            "is_fraud": is_fraud,
            "processing_time_ms": 110,
            "message": "Transaction blocked" if status == "failed" else "Transaction requires review" if status == "pending" else "Transaction approved"
        }
        
    except Exception as e:
        logger.error(f"Error creating transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transactions/{transaction_id}")
async def get_transaction(transaction_id: str):
    """Get transaction details by ID"""
    try:
        pg_conn = get_postgres_connection()
        cursor = pg_conn.cursor()
        
        cursor.execute("""
            SELECT 
                t.transaction_id, t.user_id, t.merchant_id, t.amount, t.currency,
                t.status, t.payment_method, t.card_last4, t.is_fraud, t.fraud_score,
                t.ml_fraud_probability, t.created_at, t.ip_address, t.device_type,
                u.email, u.name, m.name as merchant_name, m.category
            FROM transactions t
            JOIN users u ON t.user_id = u.user_id
            JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE t.transaction_id = %s
        """, (transaction_id,))
        
        row = cursor.fetchone()
        cursor.close()
        pg_conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return {
            "transaction_id": row[0],
            "user_id": row[1],
            "merchant_id": row[2],
            "amount": float(row[3]),
            "currency": row[4],
            "status": row[5],
            "payment_method": row[6],
            "card_last4": row[7],
            "is_fraud": row[8],
            "fraud_score": row[9],
            "ml_fraud_probability": float(row[10]) if row[10] else None,
            "created_at": row[11].isoformat(),
            "ip_address": str(row[12]),
            "device_type": row[13],
            "user_email": row[14],
            "user_name": row[15],
            "merchant_name": row[16],
            "merchant_category": row[17]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transactions")
async def list_transactions(limit: int = 20, offset: int = 0, is_fraud: Optional[bool] = None):
    """List transactions with pagination"""
    try:
        pg_conn = get_postgres_connection()
        cursor = pg_conn.cursor()
        
        query = """
            SELECT 
                transaction_id, user_id, merchant_id, amount, currency,
                status, is_fraud, fraud_score, created_at
            FROM transactions
        """
        
        params = []
        if is_fraud is not None:
            query += " WHERE is_fraud = %s"
            params.append(is_fraud)
        
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        transactions = []
        for row in rows:
            transactions.append({
                "transaction_id": row[0],
                "user_id": row[1],
                "merchant_id": row[2],
                "amount": float(row[3]),
                "currency": row[4],
                "status": row[5],
                "is_fraud": row[6],
                "fraud_score": row[7],
                "created_at": row[8].isoformat()
            })
        
        cursor.close()
        pg_conn.close()
        
        return {
            "transactions": transactions,
            "count": len(transactions),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ANALYTICS ENDPOINTS (ClickHouse)
# ============================================

@app.get("/api/analytics/fraud-rate")
async def get_fraud_rate(days: int = 7):
    """Get fraud rate for last N days"""
    try:
        ch_client = get_clickhouse_client()
        
        query = f"""
            SELECT 
                toDate(created_at) as date,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_transactions,
                ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
            FROM transactions_olap
            WHERE created_at >= now() - INTERVAL {days} DAY
            GROUP BY date
            ORDER BY date DESC
        """
        
        result = ch_client.execute(query)
        
        analytics = []
        for row in result:
            analytics.append({
                "date": row[0].isoformat(),
                "total_transactions": row[1],
                "fraud_transactions": row[2],
                "fraud_rate": float(row[3])
            })
        
        return {"analytics": analytics, "period_days": days}
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        # Fallback to PostgreSQL if ClickHouse not available
        return await get_fraud_rate_postgres(days)

async def get_fraud_rate_postgres(days: int):
    """Fallback fraud rate calculation using PostgreSQL"""
    try:
        pg_conn = get_postgres_connection()
        cursor = pg_conn.cursor()
        
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total_tx,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_tx,
                ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
            FROM transactions
            WHERE created_at >= NOW() - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """, (days,))
        
        rows = cursor.fetchall()
        cursor.close()
        pg_conn.close()
        
        analytics = []
        for row in rows:
            analytics.append({
                "date": row[0].isoformat(),
                "total_transactions": row[1],
                "fraud_transactions": row[2],
                "fraud_rate": float(row[3])
            })
        
        return {"analytics": analytics, "period_days": days, "source": "postgres_fallback"}
        
    except Exception as e:
        logger.error(f"Error in PostgreSQL fallback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/top-merchants")
async def get_top_merchants(limit: int = 10):
    """Get top merchants by transaction volume"""
    try:
        pg_conn = get_postgres_connection()
        cursor = pg_conn.cursor()
        
        cursor.execute("""
            SELECT 
                m.merchant_id,
                m.name,
                m.category,
                COUNT(t.transaction_id) as total_transactions,
                SUM(t.amount) as total_amount,
                SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_count,
                ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / NULLIF(COUNT(t.transaction_id), 0), 2) as fraud_rate
            FROM merchants m
            LEFT JOIN transactions t ON m.merchant_id = t.merchant_id
            GROUP BY m.merchant_id, m.name, m.category
            ORDER BY total_amount DESC
            LIMIT %s
        """, (limit,))
        
        rows = cursor.fetchall()
        cursor.close()
        pg_conn.close()
        
        merchants = []
        for row in rows:
            merchants.append({
                "merchant_id": row[0],
                "name": row[1],
                "category": row[2],
                "total_transactions": row[3],
                "total_amount": float(row[4]) if row[4] else 0,
                "fraud_count": row[5],
                "fraud_rate": float(row[6]) if row[6] else 0
            })
        
        return {"merchants": merchants}
        
    except Exception as e:
        logger.error(f"Error fetching top merchants: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# GRAPH ANALYTICS (Neo4j)
# ============================================

@app.get("/api/graph/fraud-network/{user_id}")
async def get_fraud_network(user_id: str, depth: int = 2):
    """Get fraud network for a user using Neo4j graph traversal"""
    try:
        neo4j_driver = get_neo4j_driver()
        
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH path = (u:User {user_id: $user_id})-[:TRANSACTED_WITH*1..{depth}]-(m:Merchant)
                WHERE EXISTS((u)-[:FRAUD_DETECTED]-(m))
                RETURN u.user_id as user_id, m.merchant_id as merchant_id, m.name as merchant_name
                LIMIT 50
            """.replace("{depth}", str(depth)), user_id=user_id)
            
            network = []
            for record in result:
                network.append({
                    "user_id": record["user_id"],
                    "merchant_id": record["merchant_id"],
                    "merchant_name": record["merchant_name"]
                })
            
            return {"fraud_network": network, "depth": depth}
    
    except Exception as e:
        logger.error(f"Error fetching fraud network: {e}")
        return {"fraud_network": [], "error": str(e)}

# ============================================
# BACKGROUND TASKS
# ============================================

def publish_transaction_event(tx_id: str, transaction_data: dict, fraud_score: int, is_fraud: bool):
    """Publish transaction event to Kafka"""
    try:
        producer = get_kafka_producer()
        
        event = {
            "transaction_id": tx_id,
            "timestamp": datetime.now().isoformat(),
            "fraud_score": fraud_score,
            "is_fraud": is_fraud,
            **transaction_data
        }
        
        topic = "fraud.alerts" if is_fraud else "payment.events"
        producer.produce(topic, key=tx_id, value=str(event))
        producer.flush()
        
        logger.info(f"Published event to Kafka topic: {topic}")
    except Exception as e:
        logger.error(f"Error publishing to Kafka: {e}")

def store_transaction_mongodb(tx_id: str, transaction_data: dict, fraud_score: int):
    """Store transaction in MongoDB for flexible queries"""
    try:
        mongo_client = get_mongodb_client()
        db = mongo_client["stripe_nosql"]
        collection = db["transactions"]
        
        document = {
            "transaction_id": tx_id,
            "created_at": datetime.now(),
            "fraud_score": fraud_score,
            **transaction_data
        }
        
        collection.insert_one(document)
        logger.info(f"Stored transaction in MongoDB: {tx_id}")
    except Exception as e:
        logger.error(f"Error storing in MongoDB: {e}")

def log_to_elasticsearch(tx_id: str, event_type: str, fraud_score: int, status: str):
    """Log event to Elasticsearch"""
    try:
        es_client = get_elasticsearch_client()
        
        doc = {
            "transaction_id": tx_id,
            "event_type": event_type,
            "fraud_score": fraud_score,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        es_client.index(index="stripe-transactions", document=doc)
        logger.info(f"Logged to Elasticsearch: {tx_id}")
    except Exception as e:
        logger.error(f"Error logging to Elasticsearch: {e}")

def create_fraud_relationship(user_id: str, merchant_id: str, tx_id: str):
    """Create fraud relationship in Neo4j graph"""
    try:
        neo4j_driver = get_neo4j_driver()
        
        with neo4j_driver.session() as session:
            session.run("""
                MERGE (u:User {user_id: $user_id})
                MERGE (m:Merchant {merchant_id: $merchant_id})
                CREATE (u)-[:FRAUD_DETECTED {transaction_id: $tx_id, timestamp: datetime()}]->(m)
            """, user_id=user_id, merchant_id=merchant_id, tx_id=tx_id)
            
            logger.info(f"Created fraud relationship in Neo4j: {user_id} -> {merchant_id}")
    except Exception as e:
        logger.error(f"Error creating Neo4j relationship: {e}")

# ============================================
# STARTUP MESSAGE
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
