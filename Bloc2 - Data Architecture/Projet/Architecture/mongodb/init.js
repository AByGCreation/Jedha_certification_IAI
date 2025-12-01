// ============================================
// MONGODB - NoSQL Database Initialization
// ============================================

db = db.getSiblingDB('stripe_nosql');

// Create disputes collection
db.createCollection('disputes');
db.disputes.createIndex({ "transaction_id": 1 }, { unique: true });
db.disputes.createIndex({ "status": 1 });
db.disputes.createIndex({ "created_at": -1 });

// Create transactions collection (flexible schema)
db.createCollection('transactions');
db.transactions.createIndex({ "transaction_id": 1 }, { unique: true });
db.transactions.createIndex({ "user_id": 1 });
db.transactions.createIndex({ "created_at": -1 });
db.transactions.createIndex({ "fraud_score": -1 });

// Create product_catalog collection
db.createCollection('product_catalog');
db.product_catalog.createIndex({ "merchant_id": 1 });
db.product_catalog.createIndex({ "category": 1 });

print('MongoDB collections created successfully');
