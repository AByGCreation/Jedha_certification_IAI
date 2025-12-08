from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import os
from urllib.error import URLError
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

@st.cache_data
def check_table_exists() -> bool:
    """Check if the transactions table exists in the database"""
    try:
        engine = create_engine(os.environ['BACKEND_STORE_URI'])
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'transactions'
                );
            """))
            return result.scalar()
    except Exception as e:
        st.error(f"Error checking table existence: {str(e)}")
        return False

@st.cache_data
def get_data() -> pd.DataFrame:
    """Fetch transactions from the previous day"""
    try:
        conn = st.connection(
            "postgresql",
            type="sql",
            url=os.environ['BACKEND_STORE_URI']
        )

        # Use parameterized query for better safety
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        query = f"""
        SELECT * FROM transactions
        WHERE trans_date_trans_time >= '{yesterday} 00:00:00'
        AND trans_date_trans_time < '{today} 00:00:00'
        ORDER BY trans_date_trans_time DESC
        """

        df = conn.query(query)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()
JOUR_HIER = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
st.title(f"Tableau des transactions bancaires du {JOUR_HIER}")

# Check if table exists
if not check_table_exists():
    st.error("⚠️ La table 'transactions' n'existe pas dans la base de données!")
    st.info("""
    ### Pour créer la table, exécutez:
    ```bash
    python create_database.py
    ```
    Ce script va:
    - Créer la table 'transactions' avec le bon schéma
    - Optionnellement charger des données d'exemple
    """)
    st.stop()

try:
    df = get_data()

    if df.empty:
        st.warning(f"⚠️ Aucune transaction trouvée pour le {JOUR_HIER}")
        st.info("Assurez-vous que des transactions ont été chargées dans la base de données pour cette date.")
    else:
        st.success(f"✅ {len(df)} transactions trouvées")

        st.dataframe(data=df, width="content", hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Nombre de transactions", value=len(df))
            st.metric(label="Montant total des transactions", value=f"${df['amt'].sum():,.2f}")
        with col2:
            # Check if prediction column exists
            if 'prediction' in df.columns:
                fraud_count = len(df[df['prediction'] == 1])
                fraud_amount = df[df['prediction'] == 1]['amt'].sum()
                st.metric(label="Nombre de transactions frauduleuses", value=fraud_count)
                st.metric(label="Montant total des transactions frauduleuses", value=f"${fraud_amount:,.2f}")
            elif 'is_fraud' in df.columns:
                fraud_count = len(df[df['is_fraud'] == 1])
                fraud_amount = df[df['is_fraud'] == 1]['amt'].sum()
                st.metric(label="Nombre de transactions frauduleuses", value=fraud_count)
                st.metric(label="Montant total des transactions frauduleuses", value=f"${fraud_amount:,.2f}")
            else:
                st.info("Colonne 'prediction' ou 'is_fraud' non trouvée")

except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")
except Exception as e:
    st.error(f"Une erreur s'est produite: {str(e)}")
    st.info("Vérifiez que la variable d'environnement BACKEND_STORE_URI est correctement définie.")
