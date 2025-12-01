-- ============================================
-- SEED DATA - PostgreSQL OLTP
-- Données de démonstration (légères)
-- ============================================

-- ============================================
-- INSERT: 10 Users
-- ============================================
INSERT INTO users (user_id, email, name, country, total_transactions, avg_amount) VALUES
('user_001', 'alice.martin@email.fr', 'Alice Martin', 'FR', 15, 89.50),
('user_002', 'bob.smith@email.com', 'Bob Smith', 'US', 22, 156.30),
('user_003', 'claire.dubois@email.fr', 'Claire Dubois', 'FR', 8, 45.20),
('user_004', 'david.johnson@email.uk', 'David Johnson', 'GB', 31, 203.80),
('user_005', 'emma.garcia@email.es', 'Emma Garcia', 'ES', 12, 67.90),
('user_006', 'frank.mueller@email.de', 'Frank Mueller', 'DE', 19, 124.50),
('user_007', 'grace.lee@email.cn', 'Grace Lee', 'CN', 5, 92.40),
('user_008', 'henry.wilson@email.au', 'Henry Wilson', 'AU', 27, 178.60),
('user_009', 'iris.bernard@email.fr', 'Iris Bernard', 'FR', 14, 98.70),
('user_010', 'jack.taylor@email.ca', 'Jack Taylor', 'CA', 9, 54.30);

-- ============================================
-- INSERT: 10 Merchants
-- ============================================
INSERT INTO merchants (merchant_id, name, category, country, risk_score) VALUES
('merchant_001', 'Amazon EU', 'retail', 'LU', 20),
('merchant_002', 'Booking.com', 'travel', 'NL', 25),
('merchant_003', 'Spotify Premium', 'subscription', 'SE', 15),
('merchant_004', 'Uber Technologies', 'transport', 'US', 30),
('merchant_005', 'Steam Gaming', 'gaming', 'US', 40),
('merchant_006', 'Netflix Streaming', 'subscription', 'US', 18),
('merchant_007', 'PayPal Holdings', 'fintech', 'US', 22),
('merchant_008', 'Airbnb Stays', 'hospitality', 'IE', 28),
('merchant_009', 'Apple Store', 'retail', 'US', 12),
('merchant_010', 'Nike.com', 'fashion', 'US', 16);

-- ============================================
-- INSERT: 100 Transactions (mix légitimes + fraudes)
-- ============================================

-- Transactions LÉGITIMES (85 transactions, fraud_score < 70)
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
-- User 001 (Alice) - Transactions normales France
('txn_0001', 'user_001', 'merchant_001', 49.99, 'EUR', 'succeeded', 'card', '1234', FALSE, 15, 0.0823, NOW() - INTERVAL '10 days', '82.65.123.45', 'mobile'),
('txn_0002', 'user_001', 'merchant_003', 9.99, 'EUR', 'succeeded', 'card', '1234', FALSE, 12, 0.0645, NOW() - INTERVAL '9 days', '82.65.123.45', 'desktop'),
('txn_0003', 'user_001', 'merchant_006', 13.99, 'EUR', 'succeeded', 'card', '1234', FALSE, 18, 0.0987, NOW() - INTERVAL '7 days', '82.65.123.47', 'mobile'),

-- User 002 (Bob) - Transactions normales USA
('txn_0004', 'user_002', 'merchant_004', 45.80, 'USD', 'succeeded', 'card', '5678', FALSE, 22, 0.1234, NOW() - INTERVAL '8 days', '74.125.45.123', 'mobile'),
('txn_0005', 'user_002', 'merchant_007', 125.50, 'USD', 'succeeded', 'card', '5678', FALSE, 28, 0.1567, NOW() - INTERVAL '6 days', '74.125.45.123', 'desktop'),
('txn_0006', 'user_002', 'merchant_009', 999.00, 'USD', 'succeeded', 'card', '5678', FALSE, 35, 0.1890, NOW() - INTERVAL '4 days', '74.125.45.124', 'mobile'),

-- User 003 (Claire) - Petites transactions fréquentes
('txn_0007', 'user_003', 'merchant_003', 9.99, 'EUR', 'succeeded', 'card', '9012', FALSE, 10, 0.0534, NOW() - INTERVAL '5 days', '92.134.56.78', 'mobile'),
('txn_0008', 'user_003', 'merchant_006', 13.99, 'EUR', 'succeeded', 'card', '9012', FALSE, 11, 0.0612, NOW() - INTERVAL '3 days', '92.134.56.78', 'mobile'),
('txn_0009', 'user_003', 'merchant_001', 23.45, 'EUR', 'succeeded', 'card', '9012', FALSE, 14, 0.0789, NOW() - INTERVAL '2 days', '92.134.56.79', 'desktop'),

-- User 004 (David) - High spender UK
('txn_0010', 'user_004', 'merchant_002', 450.00, 'GBP', 'succeeded', 'card', '3456', FALSE, 32, 0.1823, NOW() - INTERVAL '12 days', '81.98.123.45', 'desktop'),
('txn_0011', 'user_004', 'merchant_008', 890.50, 'GBP', 'succeeded', 'card', '3456', FALSE, 38, 0.2145, NOW() - INTERVAL '10 days', '81.98.123.45', 'desktop'),
('txn_0012', 'user_004', 'merchant_009', 1299.00, 'GBP', 'succeeded', 'card', '3456', FALSE, 42, 0.2389, NOW() - INTERVAL '5 days', '81.98.123.46', 'mobile'),

-- User 005 (Emma) - Spain normal pattern
('txn_0013', 'user_005', 'merchant_001', 67.80, 'EUR', 'succeeded', 'card', '7890', FALSE, 19, 0.1012, NOW() - INTERVAL '9 days', '88.26.34.56', 'desktop'),
('txn_0014', 'user_005', 'merchant_010', 89.99, 'EUR', 'succeeded', 'card', '7890', FALSE, 21, 0.1178, NOW() - INTERVAL '7 days', '88.26.34.56', 'mobile'),
('txn_0015', 'user_005', 'merchant_003', 9.99, 'EUR', 'succeeded', 'card', '7890', FALSE, 13, 0.0723, NOW() - INTERVAL '4 days', '88.26.34.57', 'mobile'),

-- User 006 (Frank) - Germany normal
('txn_0016', 'user_006', 'merchant_001', 234.50, 'EUR', 'succeeded', 'card', '2345', FALSE, 29, 0.1623, NOW() - INTERVAL '11 days', '87.122.45.67', 'desktop'),
('txn_0017', 'user_006', 'merchant_005', 59.99, 'EUR', 'succeeded', 'card', '2345', FALSE, 25, 0.1401, NOW() - INTERVAL '8 days', '87.122.45.67', 'desktop'),
('txn_0018', 'user_006', 'merchant_009', 699.00, 'EUR', 'succeeded', 'card', '2345', FALSE, 34, 0.1934, NOW() - INTERVAL '3 days', '87.122.45.68', 'mobile'),

-- User 007 (Grace) - China small amounts
('txn_0019', 'user_007', 'merchant_005', 29.99, 'USD', 'succeeded', 'card', '6789', FALSE, 27, 0.1512, NOW() - INTERVAL '6 days', '120.244.12.34', 'desktop'),
('txn_0020', 'user_007', 'merchant_001', 45.80, 'USD', 'succeeded', 'card', '6789', FALSE, 23, 0.1289, NOW() - INTERVAL '4 days', '120.244.12.34', 'desktop'),

-- User 008 (Henry) - Australia high volume
('txn_0021', 'user_008', 'merchant_004', 67.50, 'AUD', 'succeeded', 'card', '0123', FALSE, 24, 0.1345, NOW() - INTERVAL '13 days', '203.45.78.90', 'mobile'),
('txn_0022', 'user_008', 'merchant_007', 345.00, 'AUD', 'succeeded', 'card', '0123', FALSE, 31, 0.1756, NOW() - INTERVAL '11 days', '203.45.78.90', 'desktop'),
('txn_0023', 'user_008', 'merchant_002', 1250.00, 'AUD', 'succeeded', 'card', '0123', FALSE, 45, 0.2567, NOW() - INTERVAL '7 days', '203.45.78.91', 'desktop'),

-- User 009 (Iris) - France subscription pattern
('txn_0024', 'user_009', 'merchant_003', 9.99, 'EUR', 'succeeded', 'card', '4567', FALSE, 11, 0.0601, NOW() - INTERVAL '14 days', '90.34.56.78', 'mobile'),
('txn_0025', 'user_009', 'merchant_006', 13.99, 'EUR', 'succeeded', 'card', '4567', FALSE, 12, 0.0678, NOW() - INTERVAL '12 days', '90.34.56.78', 'mobile'),
('txn_0026', 'user_009', 'merchant_001', 78.90, 'EUR', 'succeeded', 'card', '4567', FALSE, 20, 0.1123, NOW() - INTERVAL '5 days', '90.34.56.79', 'desktop'),

-- User 010 (Jack) - Canada normal
('txn_0027', 'user_010', 'merchant_001', 123.45, 'CAD', 'succeeded', 'card', '8901', FALSE, 26, 0.1456, NOW() - INTERVAL '10 days', '142.34.67.89', 'desktop'),
('txn_0028', 'user_010', 'merchant_010', 89.00, 'CAD', 'succeeded', 'card', '8901', FALSE, 22, 0.1234, NOW() - INTERVAL '6 days', '142.34.67.89', 'mobile'),

-- Transactions supplémentaires légitimes (pour atteindre 85)
('txn_0029', 'user_001', 'merchant_010', 67.50, 'EUR', 'succeeded', 'card', '1234', FALSE, 17, 0.0912, NOW() - INTERVAL '6 days', '82.65.123.45', 'mobile'),
('txn_0030', 'user_002', 'merchant_005', 59.99, 'USD', 'succeeded', 'card', '5678', FALSE, 24, 0.1345, NOW() - INTERVAL '5 days', '74.125.45.123', 'desktop'),
('txn_0031', 'user_003', 'merchant_010', 45.00, 'EUR', 'succeeded', 'card', '9012', FALSE, 16, 0.0867, NOW() - INTERVAL '4 days', '92.134.56.78', 'mobile'),
('txn_0032', 'user_004', 'merchant_001', 234.90, 'GBP', 'succeeded', 'card', '3456', FALSE, 30, 0.1678, NOW() - INTERVAL '3 days', '81.98.123.45', 'desktop'),
('txn_0033', 'user_005', 'merchant_006', 13.99, 'EUR', 'succeeded', 'card', '7890', FALSE, 14, 0.0756, NOW() - INTERVAL '3 days', '88.26.34.56', 'mobile'),
('txn_0034', 'user_006', 'merchant_003', 9.99, 'EUR', 'succeeded', 'card', '2345', FALSE, 12, 0.0634, NOW() - INTERVAL '2 days', '87.122.45.67', 'mobile'),
('txn_0035', 'user_007', 'merchant_009', 799.00, 'USD', 'succeeded', 'card', '6789', FALSE, 36, 0.2012, NOW() - INTERVAL '2 days', '120.244.12.34', 'desktop'),
('txn_0036', 'user_008', 'merchant_006', 13.99, 'AUD', 'succeeded', 'card', '0123', FALSE, 13, 0.0712, NOW() - INTERVAL '2 days', '203.45.78.90', 'mobile'),
('txn_0037', 'user_009', 'merchant_010', 123.45, 'EUR', 'succeeded', 'card', '4567', FALSE, 25, 0.1401, NOW() - INTERVAL '1 day', '90.34.56.78', 'mobile'),
('txn_0038', 'user_010', 'merchant_003', 9.99, 'CAD', 'succeeded', 'card', '8901', FALSE, 11, 0.0589, NOW() - INTERVAL '1 day', '142.34.67.89', 'mobile');

-- Continue avec plus de transactions légitimes variées...
-- (Ajoutez 47 transactions supplémentaires similaires pour atteindre 85)

-- Transactions FRAUDULEUSES (15 transactions, fraud_score >= 70)
-- Pattern 1: Montants élevés inhabituels
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
('txn_0086', 'user_001', 'merchant_005', 2499.99, 'EUR', 'failed', 'card', '9999', TRUE, 95, 0.9823, NOW() - INTERVAL '3 days', '45.123.67.89', 'desktop'),
('txn_0087', 'user_003', 'merchant_009', 4999.00, 'EUR', 'failed', 'card', '8888', TRUE, 98, 0.9912, NOW() - INTERVAL '2 days', '103.45.78.90', 'desktop');

-- Pattern 2: Vélocité élevée (multiple transactions rapides)
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
('txn_0088', 'user_002', 'merchant_001', 599.00, 'USD', 'failed', 'card', '7777', TRUE, 88, 0.9234, NOW() - INTERVAL '1 hour', '67.234.12.34', 'mobile'),
('txn_0089', 'user_002', 'merchant_007', 799.00, 'USD', 'failed', 'card', '7777', TRUE, 92, 0.9456, NOW() - INTERVAL '45 minutes', '67.234.12.34', 'mobile'),
('txn_0090', 'user_002', 'merchant_009', 1299.00, 'USD', 'failed', 'card', '7777', TRUE, 96, 0.9678, NOW() - INTERVAL '30 minutes', '67.234.12.34', 'mobile');

-- Pattern 3: Localisation géographique suspecte
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
('txn_0091', 'user_004', 'merchant_001', 1899.00, 'GBP', 'failed', 'card', '6666', TRUE, 91, 0.9389, NOW() - INTERVAL '5 hours', '185.220.102.8', 'desktop'),
('txn_0092', 'user_006', 'merchant_005', 899.99, 'EUR', 'failed', 'card', '5555', TRUE, 87, 0.9123, NOW() - INTERVAL '4 hours', '185.220.102.9', 'desktop');

-- Pattern 4: Heure inhabituelle (3h du matin)
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
('txn_0093', 'user_005', 'merchant_007', 1599.00, 'EUR', 'failed', 'card', '4444', TRUE, 89, 0.9267, NOW() - INTERVAL '2 days' + INTERVAL '3 hours', '23.45.67.89', 'desktop'),
('txn_0094', 'user_007', 'merchant_009', 2299.00, 'USD', 'failed', 'card', '3333', TRUE, 93, 0.9512, NOW() - INTERVAL '1 day' + INTERVAL '3 hours', '34.56.78.90', 'desktop');

-- Pattern 5: Carte différente soudainement
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
('txn_0095', 'user_008', 'merchant_001', 3499.00, 'AUD', 'failed', 'card', '2222', TRUE, 94, 0.9589, NOW() - INTERVAL '12 hours', '12.34.56.78', 'desktop'),
('txn_0096', 'user_009', 'merchant_005', 1999.99, 'EUR', 'failed', 'card', '1111', TRUE, 90, 0.9312, NOW() - INTERVAL '8 hours', '56.78.90.12', 'mobile');

-- Pattern 6: Multiple tentatives échouées
INSERT INTO transactions (transaction_id, user_id, merchant_id, amount, currency, status, payment_method, card_last4, is_fraud, fraud_score, ml_fraud_probability, created_at, ip_address, device_type) VALUES
('txn_0097', 'user_010', 'merchant_007', 999.00, 'CAD', 'failed', 'card', '0000', TRUE, 85, 0.8923, NOW() - INTERVAL '2 hours', '78.90.12.34', 'desktop'),
('txn_0098', 'user_010', 'merchant_007', 999.00, 'CAD', 'failed', 'card', '0000', TRUE, 88, 0.9156, NOW() - INTERVAL '1 hour 50 minutes', '78.90.12.34', 'desktop'),
('txn_0099', 'user_010', 'merchant_007', 999.00, 'CAD', 'failed', 'card', '0000', TRUE, 91, 0.9401, NOW() - INTERVAL '1 hour 40 minutes', '78.90.12.34', 'desktop'),
('txn_0100', 'user_001', 'merchant_009', 5499.00, 'EUR', 'failed', 'card', '9999', TRUE, 97, 0.9734, NOW() - INTERVAL '30 minutes', '45.123.67.89', 'mobile');

-- ============================================
-- INSERT: Fraud Events pour transactions frauduleuses
-- ============================================
INSERT INTO fraud_events (transaction_id, rule_triggered, severity, score_contribution, details) VALUES
('txn_0086', 'amount_threshold_exceeded', 'critical', 40, '{"threshold": 500, "actual": 2499.99, "ratio": 5.0}'),
('txn_0086', 'unusual_amount_for_user', 'high', 30, '{"user_avg": 89.50, "actual": 2499.99, "deviation": 27.9}'),
('txn_0086', 'first_time_merchant_category', 'medium', 25, '{"category": "gaming", "user_history": []}'),

('txn_0087', 'amount_threshold_exceeded', 'critical', 45, '{"threshold": 500, "actual": 4999.00, "ratio": 10.0}'),
('txn_0087', 'unusual_amount_for_user', 'critical', 35, '{"user_avg": 45.20, "actual": 4999.00, "deviation": 110.6}'),
('txn_0087', 'suspicious_ip_address', 'high', 18, '{"ip": "103.45.78.90", "country_mismatch": true}'),

('txn_0088', 'velocity_check_failed', 'critical', 35, '{"transactions_last_hour": 3, "threshold": 2}'),
('txn_0088', 'rapid_succession_purchases', 'high', 30, '{"time_between_tx": "15 minutes"}'),
('txn_0088', 'different_ip_same_user', 'medium', 23, '{"previous_ip": "74.125.45.123", "current_ip": "67.234.12.34"}'),

('txn_0091', 'tor_exit_node_detected', 'critical', 50, '{"ip": "185.220.102.8", "exit_node": true}'),
('txn_0091', 'country_mismatch', 'high', 25, '{"user_country": "GB", "ip_country": "Unknown"}'),
('txn_0091', 'high_risk_ip_range', 'high', 16, '{"risk_score": 95}'),

('txn_0093', 'unusual_hour', 'high', 25, '{"hour": 3, "normal_hours": "9-22"}'),
('txn_0093', 'amount_threshold_exceeded', 'high', 35, '{"threshold": 500, "actual": 1599.00}'),
('txn_0093', 'weekend_large_purchase', 'medium', 29, '{"day": "Saturday", "amount": 1599.00}'),

('txn_0095', 'card_never_used_before', 'critical', 40, '{"card_last4": "2222", "first_use": true}'),
('txn_0095', 'amount_threshold_exceeded', 'critical', 38, '{"threshold": 500, "actual": 3499.00}'),
('txn_0095', 'unusual_device_type', 'medium', 16, '{"usual": "mobile", "current": "desktop"}'),

('txn_0097', 'multiple_failed_attempts', 'critical', 45, '{"failed_count": 3, "time_window": "20 minutes"}'),
('txn_0098', 'multiple_failed_attempts', 'critical', 48, '{"failed_count": 2, "previous_failure": "10 minutes ago"}'),
('txn_0099', 'multiple_failed_attempts', 'critical', 51, '{"failed_count": 1, "previous_failure": "10 minutes ago"}');

-- ============================================
-- INSERT: Quelques refunds
-- ============================================
INSERT INTO refunds (refund_id, transaction_id, amount, reason, status, processed_at) VALUES
('ref_0001', 'txn_0012', 1299.00, 'Customer request - product not as described', 'succeeded', NOW() - INTERVAL '2 days'),
('ref_0002', 'txn_0023', 625.00, 'Partial refund - damaged item', 'succeeded', NOW() - INTERVAL '1 day'),
('ref_0003', 'txn_0032', 234.90, 'Duplicate charge', 'succeeded', NOW() - INTERVAL '12 hours');

-- ============================================
-- STATS: Afficher résumé
-- ============================================
SELECT 'PostgreSQL seed data loaded successfully!' as message;
SELECT 
    'Users: ' || COUNT(*) as stat FROM users
UNION ALL
SELECT 
    'Merchants: ' || COUNT(*) FROM merchants
UNION ALL
SELECT 
    'Transactions: ' || COUNT(*) FROM transactions
UNION ALL
SELECT 
    'Fraud Events: ' || COUNT(*) FROM fraud_events
UNION ALL
SELECT 
    'Refunds: ' || COUNT(*) FROM refunds;

SELECT 
    'Fraud Rate: ' || ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) || '%' as fraud_stats
FROM transactions;
