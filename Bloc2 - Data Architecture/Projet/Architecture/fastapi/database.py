"""
Database connection managers for all services
PostgreSQL, MongoDB, Redis, Neo4j, ClickHouse, Kafka, Elasticsearch, MinIO
"""

import os
import psycopg2
from pymongo import MongoClient
import redis
from neo4j import GraphDatabase
from clickhouse_driver import Client as ClickHouseClient
from confluent_kafka import Producer
from elasticsearch import Elasticsearch
from minio import Minio
import logging

logger = logging.getLogger(__name__)

# ============================================
# POSTGRESQL (OLTP)
# ============================================

def get_postgres_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=os.getenv("POSTGRES_PORT", 5432),
            user=os.getenv("POSTGRES_USER", "stripe_user"),
            password=os.getenv("POSTGRES_PASSWORD", "stripe_password"),
            database=os.getenv("POSTGRES_DB", "stripe_oltp")
        )
        return conn
    except Exception as e:
        logger.error(f"PostgreSQL connection error: {e}")
        raise

# ============================================
# MONGODB (NoSQL)
# ============================================

def get_mongodb_client():
    """Get MongoDB client"""
    try:
        connection_string = f"mongodb://{os.getenv('MONGODB_USER', 'stripe_user')}:{os.getenv('MONGODB_PASSWORD', 'stripe_password')}@{os.getenv('MONGODB_HOST', 'mongodb')}:{os.getenv('MONGODB_PORT', 27017)}/"
        client = MongoClient(connection_string)
        return client
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
        raise

# ============================================
# REDIS (Cache)
# ============================================

def get_redis_client():
    """Get Redis client"""
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=os.getenv("REDIS_PORT", 6379),
            password=os.getenv("REDIS_PASSWORD", "stripe_password"),
            decode_responses=True
        )
        return client
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        raise

# ============================================
# NEO4J (Graph Database)
# ============================================

def get_neo4j_driver():
    """Get Neo4j driver"""
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "stripe_password")
            )
        )
        return driver
    except Exception as e:
        logger.error(f"Neo4j connection error: {e}")
        raise

# ============================================
# CLICKHOUSE (OLAP)
# ============================================

def get_clickhouse_client():
    """Get ClickHouse client"""
    try:
        client = ClickHouseClient(
            host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
            port=os.getenv("CLICKHOUSE_PORT", 9000),
            user=os.getenv("CLICKHOUSE_USER", "stripe_user"),
            password=os.getenv("CLICKHOUSE_PASSWORD", "stripe_password"),
            database=os.getenv("CLICKHOUSE_DB", "stripe_olap")
        )
        return client
    except Exception as e:
        logger.error(f"ClickHouse connection error: {e}")
        raise

# ============================================
# KAFKA (Streaming)
# ============================================

def get_kafka_producer():
    """Get Kafka producer"""
    try:
        conf = {
            'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            'client.id': 'fastapi-producer'
        }
        producer = Producer(conf)
        return producer
    except Exception as e:
        logger.error(f"Kafka producer error: {e}")
        raise

# ============================================
# ELASTICSEARCH (Logs & Search)
# ============================================

def get_elasticsearch_client():
    """Get Elasticsearch client"""
    try:
        client = Elasticsearch(
            [f"http://{os.getenv('ELASTICSEARCH_HOST', 'elasticsearch')}:{os.getenv('ELASTICSEARCH_PORT', 9200)}"]
        )
        return client
    except Exception as e:
        logger.error(f"Elasticsearch connection error: {e}")
        raise

# ============================================
# MINIO (Object Storage)
# ============================================

def get_minio_client():
    """Get MinIO client"""
    try:
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "stripe_user"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "stripe_password"),
            secure=False
        )
        return client
    except Exception as e:
        logger.error(f"MinIO connection error: {e}")
        raise
