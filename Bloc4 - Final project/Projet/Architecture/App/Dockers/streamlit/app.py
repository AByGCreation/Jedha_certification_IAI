# dashboard/test_quality_dashboard.py
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# st.set_option('wideMode' , True)
st.set_page_config(
    page_title="Dashboard Suivi des indicateurs qualité",
    layout="wide"
)
st.header("📊 Dashboard test Video")
st.markdown("""
Ce dashboard présente les indicateurs clés de performance (KPI) des tests automatisés
exécutés sur le modèle de détection de fraude. Il permet de suivre la qualité du modèle
au fil du temps et d'identifier rapidement les éventuels problèmes.
""")

# Connexion DB avec SQLAlchemy
engine = create_engine(os.getenv("BACKEND_STORE_URI"))

# KPIs
col1, col2, col3, col4 = st.columns(4)

# Dernières 24h
df_recent = pd.read_sql("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed,
        AVG(duration_ms) as avg_duration
    FROM test_logs
    WHERE created_at >= NOW() - INTERVAL '24 hours'
""", engine)

col1.metric("Total Tests (24h)", df_recent['total'].iloc[0])
col2.metric("Taux Réussite", f"{df_recent['passed'].iloc[0] / df_recent['total'].iloc[0] * 100:.1f}%")
col3.metric("Durée Moyenne", f"{df_recent['avg_duration'].iloc[0]:.0f}ms")

# Graphique évolution accuracy
df_accuracy = pd.read_sql("""
    SELECT
        DATE(created_at) as date,
        AVG(model_accuracy) as accuracy
    FROM test_logs
    WHERE test_name = 'test_model_accuracy_threshold'
      AND created_at >= NOW() - INTERVAL '30 days'
    GROUP BY DATE(created_at)
    ORDER BY date
""", engine)

fig = px.line(df_accuracy, x='date', y='accuracy', 
              title="Évolution Accuracy (30 jours)",
              labels={'accuracy': 'Accuracy', 'date': 'Date'})
fig.add_hline(y=0.92, line_dash="dash", line_color="red", 
              annotation_text="Seuil minimum (92%)")
st.plotly_chart(fig)

# Table des derniers échecs
st.subheader("🔴 Derniers Tests Échoués")
df_failures = pd.read_sql("""
    SELECT
        test_name,
        error_message,
        created_at,
        git_commit_hash
    FROM test_logs
    WHERE status = 'failed'
    ORDER BY created_at DESC
    LIMIT 10
""", engine)
st.dataframe(df_failures)