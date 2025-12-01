// ============================================
// MONGODB - Seed Data
// ============================================

db = db.getSiblingDB('stripe_nosql');

// Insert sample disputes
db.disputes.insertMany([
    {
        transaction_id: "txn_0086",
        user_id: "user_001",
        reason: "unauthorized_transaction",
        status: "pending",
        amount: 2499.99,
        evidence: {
            customer_statement: "I did not authorize this purchase",
            ip_mismatch: true,
            device_unknown: true
        },
        created_at: new Date()
    },
    {
        transaction_id: "txn_0087",
        user_id: "user_003",
        reason: "fraudulent",
        status: "resolved",
        amount: 4999.00,
        evidence: {
            customer_statement: "Card stolen",
            police_report: true,
            ip_address: "103.45.78.90"
        },
        created_at: new Date(Date.now() - 86400000)
    }
]);

// Insert sample product catalog
db.product_catalog.insertMany([
    {
        merchant_id: "merchant_001",
        products: [
            { sku: "AMZ-001", name: "Laptop Dell XPS", price: 1299.00, category: "electronics" },
            { sku: "AMZ-002", name: "iPhone 15 Pro", price: 1099.00, category: "electronics" }
        ]
    },
    {
        merchant_id: "merchant_005",
        products: [
            { sku: "STM-001", name: "Cyberpunk 2077", price: 59.99, category: "games" },
            { sku: "STM-002", name: "Baldur's Gate 3", price: 69.99, category: "games" }
        ]
    }
]);

print('MongoDB seed data inserted');
