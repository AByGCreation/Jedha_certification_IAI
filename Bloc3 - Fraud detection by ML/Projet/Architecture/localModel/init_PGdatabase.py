"""
Script to create the transactions table in Neon PostgreSQL database
and optionally populate it with sample data
"""



from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
BACKEND_STORE_URI = os.getenv("BACKEND_STORE_URI")

if not BACKEND_STORE_URI:
    raise ValueError("BACKEND_STORE_URI not found in environment variables")

print("Connecting to Neon PostgreSQL database...")
engine = create_engine(BACKEND_STORE_URI)

# SQL to create transactions table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    trans_date_trans_time TIMESTAMP NOT NULL,
    cc_num BIGINT,
    merchant VARCHAR(255),
    category VARCHAR(100),
    amt FLOAT,
    first VARCHAR(100),
    last VARCHAR(100),
    gender VARCHAR(10),
    street VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    zip INTEGER,
    lat FLOAT,
    long FLOAT,
    city_pop INTEGER,
    job VARCHAR(255),
    dob DATE,
    trans_num VARCHAR(255) UNIQUE,
    unix_time BIGINT,
    merch_lat FLOAT,
    merch_long FLOAT,
    is_fraud INTEGER DEFAULT 0,
    prediction INTEGER,
    distance_km FLOAT,
    age INTEGER,
    trans_hour INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_trans_date ON transactions(trans_date_trans_time);
CREATE INDEX IF NOT EXISTS idx_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_trans_num ON transactions(trans_num);
CREATE INDEX IF NOT EXISTS idx_category ON transactions(category);
"""

# SQL to check if table exists
CHECK_TABLE_SQL = """
SELECT EXISTS (
    SELECT FROM information_schema.tables
    WHERE table_name = 'transactions'
);
"""

def create_table():
    """Create the transactions table"""
    try:
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text(CHECK_TABLE_SQL))
            table_exists = result.scalar()

            if table_exists:
                print("⚠️  Table 'transactions' already exists.")
                response = input("Do you want to drop and recreate it? (yes/no): ")
                if response.lower() == 'yes':
                    print("Dropping existing table...")
                    conn.execute(text("DROP TABLE IF EXISTS transactions CASCADE;"))
                    conn.commit()
                    print("✓ Table dropped.")
                else:
                    print("Keeping existing table.")
                    return

            # Create table
            print("Creating table 'transactions'...")
            for statement in CREATE_TABLE_SQL.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
            print("✅ Table 'transactions' created successfully!")

            # Display table structure
            print("\nTable structure:")
            result = conn.execute(text("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'transactions'
                ORDER BY ordinal_position;
            """))

            for row in result:
                print(f"  - {row[0]}: {row[1]}" + (f"({row[2]})" if row[2] else ""))

    except Exception as e:
        print(f"❌ Error creating table: {str(e)}")
        raise

def load_sample_data(csv_path=None):
    """Load sample data from CSV file"""
    if csv_path is None:
        csv_path = input("Enter path to CSV file (or press Enter to skip): ").strip()

    if not csv_path:
        print("Skipping sample data loading.")
        return

    try:
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Prepare columns to match database schema
        column_mapping = {
            'trans_date_trans_time': 'trans_date_trans_time',
            'cc_num': 'cc_num',
            'merchant': 'merchant',
            'category': 'category',
            'amt': 'amt',
            'first': 'first',
            'last': 'last',
            'gender': 'gender',
            'street': 'street',
            'city': 'city',
            'state': 'state',
            'zip': 'zip',
            'lat': 'lat',
            'long': 'long',
            'city_pop': 'city_pop',
            'job': 'job',
            'dob': 'dob',
            'trans_num': 'trans_num',
            'unix_time': 'unix_time',
            'merch_lat': 'merch_lat',
            'merch_long': 'merch_long',
            'is_fraud': 'is_fraud'
        }

        # Select and rename columns
        df_to_insert = df[[col for col in column_mapping.keys() if col in df.columns]].copy()

        # Convert datetime columns
        if 'trans_date_trans_time' in df_to_insert.columns:
            df_to_insert['trans_date_trans_time'] = pd.to_datetime(df_to_insert['trans_date_trans_time'])
        if 'dob' in df_to_insert.columns:
            df_to_insert['dob'] = pd.to_datetime(df_to_insert['dob'])

        # Insert data
        print(f"Inserting {len(df_to_insert)} rows into database...")
        df_to_insert.to_sql('transactions', engine, if_exists='append', index=False, method='multi', chunksize=1000)

        print(f"✅ Successfully loaded {len(df_to_insert)} rows!")

        # Display summary
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM transactions;"))
            total_count = result.scalar()

            result = conn.execute(text("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1;"))
            fraud_count = result.scalar()

            print(f"\nDatabase summary:")
            print(f"  Total transactions: {total_count}")
            print(f"  Fraudulent transactions: {fraud_count}")
            print(f"  Fraud rate: {(fraud_count/total_count*100):.2f}%")

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def verify_table():
    """Verify the table was created correctly"""
    try:
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text(CHECK_TABLE_SQL))
            if not result.scalar():
                print("❌ Table 'transactions' does not exist!")
                return False

            # Get row count
            result = conn.execute(text("SELECT COUNT(*) FROM transactions;"))
            count = result.scalar()
            print(f"\n✅ Table verification successful!")
            print(f"   Total rows: {count}")

            return True

    except Exception as e:
        print(f"❌ Error verifying table: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("NEON POSTGRESQL DATABASE SETUP")
    print("=" * 80)
    print()

    # Create table
    create_table()

    # Verify table
    verify_table()

    # Ask to load sample data
    print("\n" + "=" * 80)
    response = input("Do you want to load sample data? (yes/no): ")
    if response.lower() == 'yes':
        load_sample_data()

    print("\n" + "=" * 80)
    print("Database setup complete!")
    print("=" * 80)
