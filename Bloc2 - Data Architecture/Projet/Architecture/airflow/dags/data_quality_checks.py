"""
Data Quality Checks Pipeline
Automated data validation and monitoring

Author: David Rambeau
Certification: AIA - Bloc 3
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import psycopg2
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'david_rambeau',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

def check_null_values(**context):
    """Vérifier l'absence de valeurs NULL dans les champs critiques"""
    
    pg_conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='stripe_user',
        password='stripe_password',
        database='stripe_oltp'
    )
    cursor = pg_conn.cursor()
    
    # Vérifier les NULLs dans les colonnes critiques
    checks = [
        ("transactions", "user_id"),
        ("transactions", "merchant_id"),
        ("transactions", "amount"),
        ("transactions", "created_at"),
        ("users", "user_id"),
        ("users", "email"),
        ("merchants", "merchant_id"),
    ]
    
    errors = []
    for table, column in checks:
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
        null_count = cursor.fetchone()[0]
        
        if null_count > 0:
            errors.append(f"{table}.{column} has {null_count} NULL values")
            logger.error(f"❌ {table}.{column}: {null_count} NULLs found")
        else:
            logger.info(f"✅ {table}.{column}: No NULLs")
    
    cursor.close()
    pg_conn.close()
    
    if errors:
        raise ValueError(f"NULL value check failed:\n" + "\n".join(errors))
    
    return "All NULL checks passed"

def check_data_types(**context):
    """Vérifier la cohérence des types de données"""
    
    pg_conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='stripe_user',
        password='stripe_password',
        database='stripe_oltp'
    )
    cursor = pg_conn.cursor()
    
    errors = []
    
    # Check 1: Amounts doivent être positifs
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE amount <= 0")
    negative_amounts = cursor.fetchone()[0]
    if negative_amounts > 0:
        errors.append(f"Found {negative_amounts} transactions with amount <= 0")
    
    # Check 2: Fraud score entre 0 et 100
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE fraud_score < 0 OR fraud_score > 100")
    invalid_scores = cursor.fetchone()[0]
    if invalid_scores > 0:
        errors.append(f"Found {invalid_scores} transactions with invalid fraud_score")
    
    # Check 3: Currency codes valides
    cursor.execute("""
        SELECT COUNT(*) FROM transactions 
        WHERE currency NOT IN ('EUR', 'USD', 'GBP', 'CAD', 'AUD')
    """)
    invalid_currency = cursor.fetchone()[0]
    if invalid_currency > 0:
        errors.append(f"Found {invalid_currency} transactions with invalid currency")
    
    # Check 4: Status valides
    cursor.execute("""
        SELECT COUNT(*) FROM transactions 
        WHERE status NOT IN ('pending', 'succeeded', 'failed', 'refunded', 'disputed')
    """)
    invalid_status = cursor.fetchone()[0]
    if invalid_status > 0:
        errors.append(f"Found {invalid_status} transactions with invalid status")
    
    cursor.close()
    pg_conn.close()
    
    if errors:
        raise ValueError(f"Data type validation failed:\n" + "\n".join(errors))
    
    logger.info("✅ All data type checks passed")
    return "All data type checks passed"

def check_referential_integrity(**context):
    """Vérifier l'intégrité référentielle"""
    
    pg_conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='stripe_user',
        password='stripe_password',
        database='stripe_oltp'
    )
    cursor = pg_conn.cursor()
    
    errors = []
    
    # Check 1: Transactions avec user_id inexistant
    cursor.execute("""
        SELECT COUNT(*) FROM transactions t 
        LEFT JOIN users u ON t.user_id = u.user_id 
        WHERE u.user_id IS NULL
    """)
    orphan_user = cursor.fetchone()[0]
    if orphan_user > 0:
        errors.append(f"Found {orphan_user} transactions with non-existent user_id")
    
    # Check 2: Transactions avec merchant_id inexistant
    cursor.execute("""
        SELECT COUNT(*) FROM transactions t 
        LEFT JOIN merchants m ON t.merchant_id = m.merchant_id 
        WHERE m.merchant_id IS NULL
    """)
    orphan_merchant = cursor.fetchone()[0]
    if orphan_merchant > 0:
        errors.append(f"Found {orphan_merchant} transactions with non-existent merchant_id")
    
    # Check 3: Fraud events avec transaction_id inexistant
    cursor.execute("""
        SELECT COUNT(*) FROM fraud_events fe 
        LEFT JOIN transactions t ON fe.transaction_id = t.transaction_id 
        WHERE t.transaction_id IS NULL
    """)
    orphan_fraud = cursor.fetchone()[0]
    if orphan_fraud > 0:
        errors.append(f"Found {orphan_fraud} fraud_events with non-existent transaction_id")
    
    cursor.close()
    pg_conn.close()
    
    if errors:
        raise ValueError(f"Referential integrity check failed:\n" + "\n".join(errors))
    
    logger.info("✅ All referential integrity checks passed")
    return "All referential integrity checks passed"

def check_business_rules(**context):
    """Vérifier les règles métier"""
    
    pg_conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='stripe_user',
        password='stripe_password',
        database='stripe_oltp'
    )
    cursor = pg_conn.cursor()
    
    warnings = []
    
    # Rule 1: Transactions frauduleuses doivent avoir fraud_score >= 70
    cursor.execute("""
        SELECT COUNT(*) FROM transactions 
        WHERE is_fraud = TRUE AND fraud_score < 70
    """)
    inconsistent_fraud = cursor.fetchone()[0]
    if inconsistent_fraud > 0:
        warnings.append(f"⚠️  {inconsistent_fraud} transactions marked as fraud with score < 70")
    
    # Rule 2: Transactions réussies ne doivent pas être frauduleuses
    cursor.execute("""
        SELECT COUNT(*) FROM transactions 
        WHERE status = 'succeeded' AND is_fraud = TRUE
    """)
    succeeded_fraud = cursor.fetchone()[0]
    if succeeded_fraud > 0:
        warnings.append(f"⚠️  {succeeded_fraud} succeeded transactions marked as fraud")
    
    # Rule 3: Vérifier le taux de fraude global (< 20%)
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count
        FROM transactions
        WHERE created_at >= NOW() - INTERVAL '30 days'
    """)
    result = cursor.fetchone()
    total, fraud_count = result
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0
    
    if fraud_rate > 20:
        warnings.append(f"⚠️  High fraud rate detected: {fraud_rate:.2f}% (last 30 days)")
    else:
        logger.info(f"✅ Fraud rate: {fraud_rate:.2f}% (within acceptable range)")
    
    cursor.close()
    pg_conn.close()
    
    if warnings:
        logger.warning("Business rule warnings:\n" + "\n".join(warnings))
    
    return f"Business rules checked. Warnings: {len(warnings)}"

def generate_quality_report(**context):
    """Générer un rapport de qualité des données"""
    
    null_check = context['ti'].xcom_pull(task_ids='check_null_values')
    type_check = context['ti'].xcom_pull(task_ids='check_data_types')
    integrity_check = context['ti'].xcom_pull(task_ids='check_referential_integrity')
    business_check = context['ti'].xcom_pull(task_ids='check_business_rules')
    
    report = f"""
    ═══════════════════════════════════════════════════════════
    DATA QUALITY REPORT
    Generated: {context['execution_date']}
    ═══════════════════════════════════════════════════════════
    
    ✅ NULL Values Check:          {null_check}
    ✅ Data Types Check:            {type_check}
    ✅ Referential Integrity Check: {integrity_check}
    ⚠️  Business Rules Check:       {business_check}
    
    Overall Status: PASSED
    
    ═══════════════════════════════════════════════════════════
    """
    
    logger.info(report)
    
    return report

# Définition du DAG
with DAG(
    'data_quality_checks',
    default_args=default_args,
    description='Automated data quality validation',
    schedule_interval='0 6 * * *',  # Tous les jours à 6h du matin
    start_date=days_ago(1),
    catchup=False,
    tags=['data-quality', 'validation', 'monitoring'],
) as dag:
    
    task_null = PythonOperator(
        task_id='check_null_values',
        python_callable=check_null_values,
        provide_context=True,
    )
    
    task_types = PythonOperator(
        task_id='check_data_types',
        python_callable=check_data_types,
        provide_context=True,
    )
    
    task_integrity = PythonOperator(
        task_id='check_referential_integrity',
        python_callable=check_referential_integrity,
        provide_context=True,
    )
    
    task_business = PythonOperator(
        task_id='check_business_rules',
        python_callable=check_business_rules,
        provide_context=True,
    )
    
    task_report = PythonOperator(
        task_id='generate_quality_report',
        python_callable=generate_quality_report,
        provide_context=True,
    )
    
    # Exécution en parallèle puis génération du rapport
    [task_null, task_types, task_integrity, task_business] >> task_report
