-- ============================================
-- STRIPE FRAUD DETECTION - POSTGRESQL OLTP
-- Schéma pour données transactionnelles
-- ============================================

-- Création de l'extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- TABLE: users
-- ============================================
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    country VARCHAR(2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_transactions INTEGER DEFAULT 0,
    avg_amount DECIMAL(15,2) DEFAULT 0.00,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_country ON users(country);
CREATE INDEX idx_users_created_at ON users(created_at);

-- ============================================
-- TABLE: merchants
-- ============================================
CREATE TABLE merchants (
    merchant_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    country VARCHAR(2) NOT NULL,
    risk_score INTEGER DEFAULT 50,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_merchants_category ON merchants(category);
CREATE INDEX idx_merchants_risk_score ON merchants(risk_score);

-- ============================================
-- TABLE: transactions
-- ============================================
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    merchant_id VARCHAR(50) NOT NULL REFERENCES merchants(merchant_id),
    amount DECIMAL(15,2) NOT NULL CHECK (amount >= 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'EUR',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    payment_method VARCHAR(50) NOT NULL,
    card_last4 VARCHAR(4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    device_type VARCHAR(50),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score INTEGER DEFAULT 0,
    ml_fraud_probability DECIMAL(5,4),
    CHECK (status IN ('pending', 'succeeded', 'failed', 'refunded', 'disputed'))
);

CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX idx_transactions_amount ON transactions(amount);

-- Partitionnement par date (mensuel) pour scalabilité
-- CREATE TABLE transactions_2024_11 PARTITION OF transactions
--     FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

-- ============================================
-- TABLE: fraud_events
-- ============================================
CREATE TABLE fraud_events (
    event_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL REFERENCES transactions(transaction_id),
    rule_triggered VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    score_contribution INTEGER NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (severity IN ('low', 'medium', 'high', 'critical'))
);

CREATE INDEX idx_fraud_events_transaction_id ON fraud_events(transaction_id);
CREATE INDEX idx_fraud_events_severity ON fraud_events(severity);
CREATE INDEX idx_fraud_events_created_at ON fraud_events(created_at DESC);

-- ============================================
-- TABLE: refunds
-- ============================================
CREATE TABLE refunds (
    refund_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL REFERENCES transactions(transaction_id),
    amount DECIMAL(15,2) NOT NULL CHECK (amount >= 0),
    reason VARCHAR(255),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    CHECK (status IN ('pending', 'succeeded', 'failed', 'cancelled'))
);

CREATE INDEX idx_refunds_transaction_id ON refunds(transaction_id);
CREATE INDEX idx_refunds_status ON refunds(status);

-- ============================================
-- FONCTION: Mise à jour automatique updated_at
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- VUE: Transactions avec détails complets
-- ============================================
CREATE OR REPLACE VIEW v_transactions_detailed AS
SELECT 
    t.transaction_id,
    t.amount,
    t.currency,
    t.status,
    t.created_at,
    t.is_fraud,
    t.fraud_score,
    t.ml_fraud_probability,
    u.user_id,
    u.email,
    u.name as user_name,
    u.country as user_country,
    m.merchant_id,
    m.name as merchant_name,
    m.category as merchant_category,
    m.risk_score as merchant_risk_score,
    COUNT(fe.event_id) as fraud_events_count
FROM transactions t
JOIN users u ON t.user_id = u.user_id
JOIN merchants m ON t.merchant_id = m.merchant_id
LEFT JOIN fraud_events fe ON t.transaction_id = fe.transaction_id
GROUP BY t.transaction_id, u.user_id, u.email, u.name, u.country, m.merchant_id, m.name, m.category, m.risk_score;

-- ============================================
-- VUE: Statistiques fraude par marchand
-- ============================================
CREATE OR REPLACE VIEW v_merchant_fraud_stats AS
SELECT 
    m.merchant_id,
    m.name as merchant_name,
    m.category,
    COUNT(t.transaction_id) as total_transactions,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / NULLIF(COUNT(t.transaction_id), 0), 2) as fraud_rate,
    SUM(t.amount) as total_amount,
    AVG(t.amount) as avg_amount
FROM merchants m
LEFT JOIN transactions t ON m.merchant_id = t.merchant_id
GROUP BY m.merchant_id, m.name, m.category;

-- ============================================
-- PERMISSIONS
-- ============================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO stripe_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO stripe_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO stripe_user;

-- ============================================
-- INFO
-- ============================================
SELECT 'PostgreSQL OLTP schema initialized successfully!' as message;
