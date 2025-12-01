"""
ETL Pipeline: PostgreSQL (OLTP) → ClickHouse (OLAP)
Daily synchronization of transactions for analytics

Author: David Rambeau
Certification: AIA - Bloc 3 (Pipelines de données)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import psycopg2
from clickhouse_driver import Client
import logging

logger = logging.getLogger(__name__)

# Configuration
POSTGRES_CONN_ID = 'postgres_stripe'
BATCH_SIZE = 1000

default_args = {
    'owner': 'david_rambeau',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

def extract_transactions_from_postgres(**context):
    """
    EXTRACT: Récupérer les transactions depuis PostgreSQL
    """
    execution_date = context['execution_date']
    logger.info(f"Extracting transactions for date: {execution_date}")
    
    # Connexion PostgreSQL
    pg_conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='stripe_user',
        password='stripe_password',
        database='stripe_oltp'
    )
    
    cursor = pg_conn.cursor()
    
    # Extraction des transactions avec jointures
    query = """
        SELECT 
            t.transaction_id,
            t.user_id,
            t.merchant_id,
            t.amount,
            t.currency,
            t.status,
            t.payment_method,
            t.is_fraud,
            t.fraud_score,
            t.created_at,
            u.country as user_country,
            m.category as merchant_category
        FROM transactions t
        JOIN users u ON t.user_id = u.user_id
        JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE DATE(t.created_at) = DATE(%s)
        ORDER BY t.created_at
    """
    
    cursor.execute(query, (execution_date,))
    transactions = cursor.fetchall()
    
    cursor.close()
    pg_conn.close()
    
    logger.info(f"Extracted {len(transactions)} transactions")
    
    # Stocker dans XCom pour la prochaine tâche
    context['ti'].xcom_push(key='transactions', value=transactions)
    
    return len(transactions)

def transform_transactions(**context):
    """
    TRANSFORM: Transformer les données pour ClickHouse
    """
    transactions = context['ti'].xcom_pull(key='transactions', task_ids='extract_from_postgres')
    
    if not transactions:
        logger.warning("No transactions to transform")
        return 0
    
    # Transformation: conversion types, nettoyage, enrichissement
    transformed = []
    for tx in transactions:
        transformed.append({
            'transaction_id': tx[0],
            'user_id': tx[1],
            'merchant_id': tx[2],
            'amount': float(tx[3]),
            'currency': tx[4],
            'status': tx[5],
            'payment_method': tx[6],
            'is_fraud': 1 if tx[7] else 0,
            'fraud_score': tx[8] or 0,
            'created_at': tx[9],
            'country': tx[10],
            'merchant_category': tx[11]
        })
    
    logger.info(f"Transformed {len(transformed)} transactions")
    
    context['ti'].xcom_push(key='transformed_transactions', value=transformed)
    
    return len(transformed)

def load_to_clickhouse(**context):
    """
    LOAD: Charger les données dans ClickHouse
    """
    transactions = context['ti'].xcom_pull(key='transformed_transactions', task_ids='transform_transactions')
    
    if not transactions:
        logger.warning("No transactions to load")
        return 0
    
    # Connexion ClickHouse
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    # Vérifier si la table existe, sinon la créer
    ch_client.execute("""
        CREATE TABLE IF NOT EXISTS transactions_olap (
            transaction_id String,
            user_id String,
            merchant_id String,
            amount Decimal(15, 2),
            currency String,
            status String,
            payment_method String,
            is_fraud UInt8,
            fraud_score UInt8,
            created_at DateTime,
            country String,
            merchant_category String
        ) ENGINE = MergeTree()
        ORDER BY (created_at, transaction_id)
        PARTITION BY toYYYYMM(created_at)
    """)
    
    # Insertion par batch
    inserted = 0
    for i in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[i:i+BATCH_SIZE]
        
        ch_client.execute(
            """
            INSERT INTO transactions_olap 
            (transaction_id, user_id, merchant_id, amount, currency, status, 
             payment_method, is_fraud, fraud_score, created_at, country, merchant_category)
            VALUES
            """,
            batch
        )
        
        inserted += len(batch)
        logger.info(f"Inserted batch {i//BATCH_SIZE + 1}: {len(batch)} records")
    
    logger.info(f"Successfully loaded {inserted} transactions to ClickHouse")
    
    return inserted

def validate_data_quality(**context):
    """
    VALIDATE: Vérifier la qualité des données chargées
    """
    execution_date = context['execution_date']
    
    # Connexion PostgreSQL
    pg_conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='stripe_user',
        password='stripe_password',
        database='stripe_oltp'
    )
    pg_cursor = pg_conn.cursor()
    
    # Connexion ClickHouse
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    # Compter les enregistrements dans PostgreSQL
    pg_cursor.execute("""
        SELECT COUNT(*) FROM transactions 
        WHERE DATE(created_at) = DATE(%s)
    """, (execution_date,))
    pg_count = pg_cursor.fetchone()[0]
    
    # Compter les enregistrements dans ClickHouse
    ch_result = ch_client.execute("""
        SELECT COUNT(*) FROM transactions_olap 
        WHERE toDate(created_at) = %s
    """, (execution_date.date(),))
    ch_count = ch_result[0][0]
    
    pg_cursor.close()
    pg_conn.close()
    
    # Validation
    if pg_count != ch_count:
        raise ValueError(
            f"Data quality check FAILED: PostgreSQL has {pg_count} records, "
            f"ClickHouse has {ch_count} records for date {execution_date.date()}"
        )
    
    logger.info(f"Data quality check PASSED: {pg_count} records match in both databases")
    
    return True

def send_completion_notification(**context):
    """
    NOTIFY: Envoyer une notification de fin d'ETL
    """
    execution_date = context['execution_date']
    extracted = context['ti'].xcom_pull(key='return_value', task_ids='extract_from_postgres')
    loaded = context['ti'].xcom_pull(key='return_value', task_ids='load_to_clickhouse')
    
    message = f"""
    ETL Pipeline Completed Successfully!
    
    Date: {execution_date.date()}
    Extracted: {extracted} transactions
    Loaded: {loaded} transactions
    Status: ✅ SUCCESS
    
    ClickHouse OLAP database is up to date.
    """
    
    logger.info(message)
    
    # Ici vous pouvez ajouter l'envoi d'email, Slack, etc.
    # Pour la démo, on log simplement
    
    return True

# Définition du DAG
with DAG(
    'etl_postgres_to_clickhouse',
    default_args=default_args,
    description='Daily ETL: PostgreSQL (OLTP) → ClickHouse (OLAP)',
    schedule_interval='0 2 * * *',  # Tous les jours à 2h du matin
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'postgresql', 'clickhouse', 'daily'],
) as dag:
    
    # Task 1: Extract
    extract_task = PythonOperator(
        task_id='extract_from_postgres',
        python_callable=extract_transactions_from_postgres,
        provide_context=True,
    )
    
    # Task 2: Transform
    transform_task = PythonOperator(
        task_id='transform_transactions',
        python_callable=transform_transactions,
        provide_context=True,
    )
    
    # Task 3: Load
    load_task = PythonOperator(
        task_id='load_to_clickhouse',
        python_callable=load_to_clickhouse,
        provide_context=True,
    )
    
    # Task 4: Validate
    validate_task = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        provide_context=True,
    )
    
    # Task 5: Notify
    notify_task = PythonOperator(
        task_id='send_completion_notification',
        python_callable=send_completion_notification,
        provide_context=True,
    )
    
    # Définir le flux d'exécution
    extract_task >> transform_task >> load_task >> validate_task >> notify_task
