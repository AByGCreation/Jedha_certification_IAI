"""
Daily Aggregations Pipeline
Pre-compute analytics metrics for dashboards

Author: David Rambeau
Certification: AIA - Bloc 3
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from clickhouse_driver import Client
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'david_rambeau',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def aggregate_daily_stats(**context):
    """Agréger les statistiques quotidiennes"""
    
    execution_date = context['execution_date']
    
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    # Créer la table d'agrégation si elle n'existe pas
    ch_client.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date Date,
            total_transactions UInt32,
            total_amount Decimal(15, 2),
            avg_amount Decimal(15, 2),
            fraud_transactions UInt32,
            fraud_amount Decimal(15, 2),
            fraud_rate Decimal(5, 2),
            unique_users UInt32,
            unique_merchants UInt32
        ) ENGINE = MergeTree()
        ORDER BY date
    """)
    
    # Calculer les stats du jour
    result = ch_client.execute("""
        SELECT
            toDate(created_at) as date,
            COUNT(*) as total_transactions,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            SUM(is_fraud) as fraud_transactions,
            SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) as fraud_amount,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT merchant_id) as unique_merchants
        FROM transactions_olap
        WHERE toDate(created_at) = %s
        GROUP BY date
    """, (execution_date.date(),))
    
    if result:
        stats = result[0]
        
        # Insérer les statistiques
        ch_client.execute("""
            INSERT INTO daily_stats VALUES
        """, [stats])
        
        logger.info(f"Daily stats aggregated: {stats}")
        
        return {
            'date': str(stats[0]),
            'total_transactions': stats[1],
            'fraud_rate': float(stats[6])
        }
    else:
        logger.warning(f"No data found for {execution_date.date()}")
        return None

def aggregate_merchant_stats(**context):
    """Agréger les statistiques par marchand"""
    
    execution_date = context['execution_date']
    
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    # Créer la table si elle n'existe pas
    ch_client.execute("""
        CREATE TABLE IF NOT EXISTS merchant_daily_stats (
            date Date,
            merchant_id String,
            merchant_category String,
            total_transactions UInt32,
            total_amount Decimal(15, 2),
            fraud_transactions UInt32,
            fraud_rate Decimal(5, 2)
        ) ENGINE = MergeTree()
        ORDER BY (date, merchant_id)
    """)
    
    # Calculer les stats par marchand
    result = ch_client.execute("""
        SELECT
            toDate(created_at) as date,
            merchant_id,
            merchant_category,
            COUNT(*) as total_transactions,
            SUM(amount) as total_amount,
            SUM(is_fraud) as fraud_transactions,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate
        FROM transactions_olap
        WHERE toDate(created_at) = %s
        GROUP BY date, merchant_id, merchant_category
    """, (execution_date.date(),))
    
    if result:
        ch_client.execute("""
            INSERT INTO merchant_daily_stats VALUES
        """, result)
        
        logger.info(f"Merchant stats aggregated: {len(result)} merchants")
        return len(result)
    else:
        logger.warning("No merchant data found")
        return 0

def aggregate_user_stats(**context):
    """Agréger les statistiques par utilisateur"""
    
    execution_date = context['execution_date']
    
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    # Créer la table si elle n'existe pas
    ch_client.execute("""
        CREATE TABLE IF NOT EXISTS user_daily_stats (
            date Date,
            user_id String,
            country String,
            total_transactions UInt32,
            total_amount Decimal(15, 2),
            avg_amount Decimal(15, 2),
            fraud_transactions UInt32
        ) ENGINE = MergeTree()
        ORDER BY (date, user_id)
    """)
    
    # Calculer les stats par utilisateur
    result = ch_client.execute("""
        SELECT
            toDate(created_at) as date,
            user_id,
            country,
            COUNT(*) as total_transactions,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            SUM(is_fraud) as fraud_transactions
        FROM transactions_olap
        WHERE toDate(created_at) = %s
        GROUP BY date, user_id, country
    """, (execution_date.date(),))
    
    if result:
        ch_client.execute("""
            INSERT INTO user_daily_stats VALUES
        """, result)
        
        logger.info(f"User stats aggregated: {len(result)} users")
        return len(result)
    else:
        logger.warning("No user data found")
        return 0

def aggregate_hourly_patterns(**context):
    """Agréger les patterns horaires pour détection d'anomalies"""
    
    execution_date = context['execution_date']
    
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    # Créer la table si elle n'existe pas
    ch_client.execute("""
        CREATE TABLE IF NOT EXISTS hourly_patterns (
            date Date,
            hour UInt8,
            total_transactions UInt32,
            fraud_transactions UInt32,
            fraud_rate Decimal(5, 2),
            avg_amount Decimal(15, 2)
        ) ENGINE = MergeTree()
        ORDER BY (date, hour)
    """)
    
    # Calculer les patterns horaires
    result = ch_client.execute("""
        SELECT
            toDate(created_at) as date,
            toHour(created_at) as hour,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_transactions,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate,
            AVG(amount) as avg_amount
        FROM transactions_olap
        WHERE toDate(created_at) = %s
        GROUP BY date, hour
        ORDER BY hour
    """, (execution_date.date(),))
    
    if result:
        ch_client.execute("""
            INSERT INTO hourly_patterns VALUES
        """, result)
        
        # Identifier les heures suspectes (fraud_rate > 30%)
        suspicious_hours = [
            f"{row[1]:02d}h: {row[4]:.1f}% fraud"
            for row in result if row[4] > 30
        ]
        
        if suspicious_hours:
            logger.warning(f"Suspicious hours detected: {suspicious_hours}")
        
        logger.info(f"Hourly patterns aggregated: {len(result)} hours")
        return len(result)
    else:
        logger.warning("No hourly data found")
        return 0

def optimize_clickhouse_tables(**context):
    """Optimiser les tables ClickHouse"""
    
    ch_client = Client(
        host='clickhouse',
        port=9000,
        user='stripe_user',
        password='stripe_password',
        database='stripe_olap'
    )
    
    tables_to_optimize = [
        'transactions_olap',
        'daily_stats',
        'merchant_daily_stats',
        'user_daily_stats',
        'hourly_patterns'
    ]
    
    for table in tables_to_optimize:
        try:
            ch_client.execute(f"OPTIMIZE TABLE {table} FINAL")
            logger.info(f"✅ Optimized table: {table}")
        except Exception as e:
            logger.warning(f"⚠️  Could not optimize {table}: {e}")
    
    return f"Optimized {len(tables_to_optimize)} tables"

# Définition du DAG
with DAG(
    'daily_aggregations',
    default_args=default_args,
    description='Daily metrics aggregation for analytics dashboards',
    schedule_interval='0 3 * * *',  # Tous les jours à 3h (après l'ETL)
    start_date=days_ago(1),
    catchup=False,
    tags=['aggregation', 'analytics', 'clickhouse'],
) as dag:
    
    task_daily = PythonOperator(
        task_id='aggregate_daily_stats',
        python_callable=aggregate_daily_stats,
        provide_context=True,
    )
    
    task_merchant = PythonOperator(
        task_id='aggregate_merchant_stats',
        python_callable=aggregate_merchant_stats,
        provide_context=True,
    )
    
    task_user = PythonOperator(
        task_id='aggregate_user_stats',
        python_callable=aggregate_user_stats,
        provide_context=True,
    )
    
    task_hourly = PythonOperator(
        task_id='aggregate_hourly_patterns',
        python_callable=aggregate_hourly_patterns,
        provide_context=True,
    )
    
    task_optimize = PythonOperator(
        task_id='optimize_clickhouse_tables',
        python_callable=optimize_clickhouse_tables,
        provide_context=True,
    )
    
    # Agrégations en parallèle puis optimisation
    [task_daily, task_merchant, task_user, task_hourly] >> task_optimize
