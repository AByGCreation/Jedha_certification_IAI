// ============================================
// NEO4J - Graph Database Initialization
// ============================================

// Create constraints
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE;

// Create indexes
CREATE INDEX user_country IF NOT EXISTS FOR (u:User) ON (u.country);
CREATE INDEX merchant_category IF NOT EXISTS FOR (m:Merchant) ON (m.category);

// Insert sample users
CREATE (u1:User {user_id: 'user_001', name: 'Alice Martin', country: 'FR'});
CREATE (u2:User {user_id: 'user_002', name: 'Bob Smith', country: 'US'});
CREATE (u3:User {user_id: 'user_003', name: 'Claire Dubois', country: 'FR'});

// Insert sample merchants
CREATE (m1:Merchant {merchant_id: 'merchant_001', name: 'Amazon EU', category: 'retail'});
CREATE (m2:Merchant {merchant_id: 'merchant_005', name: 'Steam Gaming', category: 'gaming'});
CREATE (m3:Merchant {merchant_id: 'merchant_009', name: 'Apple Store', category: 'retail'});

// Create sample relationships
MATCH (u:User {user_id: 'user_001'}), (m:Merchant {merchant_id: 'merchant_001'})
CREATE (u)-[:TRANSACTED_WITH {count: 5, total_amount: 450.00}]->(m);

MATCH (u:User {user_id: 'user_001'}), (m:Merchant {merchant_id: 'merchant_005'})
CREATE (u)-[:FRAUD_DETECTED {transaction_id: 'txn_0086', amount: 2499.99, timestamp: datetime()}]->(m);
