-- ============================================
-- CLICKHOUSE OLAP - Analytics Database
-- ============================================

CREATE DATABASE IF NOT EXISTS stripe_olap;

USE stripe_olap;

-- Table pour analytics (colonnar storage optimisé)
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
PARTITION BY toYYYYMM(created_at);

-- Vue agrégée par jour
CREATE MATERIALIZED VIEW IF NOT EXISTS fraud_stats_daily
ENGINE = SummingMergeTree()
ORDER BY (date, merchant_category)
AS SELECT
    toDate(created_at) as date,
    merchant_category,
    count() as total_transactions,
    sum(is_fraud) as fraud_transactions,
    sum(amount) as total_amount,
    avg(amount) as avg_amount
FROM transactions_olap
GROUP BY date, merchant_category;
