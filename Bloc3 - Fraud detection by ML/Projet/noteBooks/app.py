"""
Fraud Detection Data Collection and Analysis App
Based on collectData.ipynb notebook

This Streamlit application provides:
- Data collection from multiple sources (CSV, SQLite, Neon PostgreSQL, HuggingFace)
- Data optimization and transformation
- Exploratory Data Analysis (EDA)
- Fraud detection visualization
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sqlalchemy import create_engine
import os
import numpy as np
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Fraud Detection - Data Collection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Jedha Colors
JEDHA_VIOLET = '#8409FF'
JEDHA_BLUE = '#3AE5FF'
JEDHA_BLUE_LIGHT = '#89C2FF'
JEDHA_WHITE = '#DFF4F5'
JEDHA_BLACK = '#170035'

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        background-color: {JEDHA_WHITE};
    }}
    .stMetric {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid {JEDHA_VIOLET};
    }}
    h1, h2, h3 {{
        color: {JEDHA_BLACK};
    }}
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on earth (in km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

@st.cache_data
def load_data(source_type, connection_url, table_name=None):
    """Load data from various sources"""
    try:
        if source_type == "CSV":
            df = pd.read_csv(connection_url)
        elif source_type == "SQLite":
            conn = sqlite3.connect(connection_url)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
        elif source_type == "Neon PostgreSQL":
            engine = create_engine(connection_url)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, engine)
        else:
            st.error(f"Unknown source type: {source_type}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    df_optimized = df.copy()

    # Datetime columns
    datetime_cols = ['dob', 'trans_date_trans_time']
    for col in datetime_cols:
        if col in df_optimized.columns:
            df_optimized[col] = pd.to_datetime(df_optimized[col], errors='coerce')

    # Type conversions
    type_conversions = {
        'cc_num': 'int64',
        'merchant': 'category',
        'category': 'category',
        'job': 'category',
        'gender': 'category',
        'city': 'category',
        'state': 'category',
        'amt': 'float32',
        'zip': 'int32',
        'city_pop': 'int32',
        'unix_time': 'int64',
        'lat': 'float32',
        'long': 'float32',
        'merch_lat': 'float32',
        'merch_long': 'float32',
        'is_fraud': 'int8',
    }

    for col, dtype in type_conversions.items():
        if col in df_optimized.columns:
            try:
                if dtype == 'category':
                    df_optimized[col] = df_optimized[col].astype('category')
                elif dtype in ['float32', 'float64']:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], errors='coerce').astype(dtype)
                elif dtype in ['int8', 'int16', 'int32', 'int64']:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], errors='coerce').fillna(0).astype(dtype)
            except Exception as e:
                st.warning(f"Could not convert {col}: {str(e)}")

    return df_optimized

def prepare_data(df):
    """Prepare data for analysis"""
    df = df.copy()

    # Calculate distance between customer and merchant
    if all(col in df.columns for col in ['long', 'lat', 'merch_long', 'merch_lat']):
        df['distance_km'] = df.apply(
            lambda row: haversine(row['long'], row['lat'], row['merch_long'], row['merch_lat']),
            axis=1
        )

    # Calculate age
    if 'dob' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['dob']):
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365

    # Extract hour from transaction
    if 'trans_date_trans_time' in df.columns:
        df['trans_hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour

    # Convert amount to numeric
    if 'amt' in df.columns:
        df['amt'] = pd.to_numeric(df['amt'], errors='coerce')

    return df

# Main App
def main():
    st.title("🔍 Fraud Detection - Data Collection & Analysis")
    st.markdown("---")

    # Sidebar Configuration
    st.sidebar.header("📊 Data Source Configuration")

    source_type = st.sidebar.selectbox(
        "Select Data Source",
        ["CSV", "SQLite", "Neon PostgreSQL"]
    )

    # Source-specific inputs
    if source_type == "CSV":
        connection_url = st.sidebar.text_input(
            "CSV File Path or URL",
            value="https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
        )
        table_name = None
    elif source_type == "SQLite":
        connection_url = st.sidebar.text_input(
            "SQLite Database Path",
            value="../datasSources/inputDataset/fraudTest.db"
        )
        table_name = st.sidebar.text_input("Table Name", value="transactions")
    else:  # Neon PostgreSQL
        connection_url = st.sidebar.text_input(
            "PostgreSQL Connection String",
            value=os.getenv("BACKEND_STORE_URI", ""),
            type="password"
        )
        table_name = st.sidebar.text_input("Table Name", value="neondb")

    # Load Data Button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            df_raw = load_data(source_type, connection_url, table_name)

            if df_raw is not None:
                st.session_state['df_raw'] = df_raw
                st.success(f"✅ Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns")

    # Display data if loaded
    if 'df_raw' in st.session_state:
        df_raw = st.session_state['df_raw']

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🔧 Data Optimization", "📊 EDA", "🚨 Fraud Analysis"])

        with tab1:
            st.header("Data Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df_raw):,}")
            with col2:
                st.metric("Total Columns", len(df_raw.columns))
            with col3:
                if 'is_fraud' in df_raw.columns:
                    fraud_count = df_raw['is_fraud'].sum()
                    st.metric("Fraud Cases", f"{fraud_count:,}")
            with col4:
                if 'is_fraud' in df_raw.columns:
                    fraud_rate = (df_raw['is_fraud'].sum() / len(df_raw)) * 100
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            st.subheader("Sample Data")
            st.dataframe(df_raw.head(100), use_container_width=True)

            st.subheader("Data Types")
            st.write(df_raw.dtypes)

        with tab2:
            st.header("Data Optimization")

            if st.button("🚀 Optimize DataFrame"):
                with st.spinner("Optimizing data types..."):
                    original_memory = df_raw.memory_usage(deep=True).sum() / (1024**2)
                    df_optimized = optimize_dataframe(df_raw)
                    optimized_memory = df_optimized.memory_usage(deep=True).sum() / (1024**2)
                    memory_saved = original_memory - optimized_memory
                    memory_saved_pct = (memory_saved / original_memory) * 100

                    st.session_state['df_optimized'] = df_optimized

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Memory", f"{original_memory:.2f} MB")
                    with col2:
                        st.metric("Optimized Memory", f"{optimized_memory:.2f} MB")
                    with col3:
                        st.metric("Memory Saved", f"{memory_saved:.2f} MB ({memory_saved_pct:.1f}%)")

                    st.success("✅ Optimization complete!")

        with tab3:
            st.header("Exploratory Data Analysis")

            # Use optimized dataframe if available
            df = st.session_state.get('df_optimized', df_raw)

            if st.button("🔍 Run EDA"):
                with st.spinner("Preparing data..."):
                    df = prepare_data(df)
                    st.session_state['df_prepared'] = df

                st.success("✅ Data preparation complete!")

                # Display key metrics
                if 'distance_km' in df.columns:
                    st.subheader("Distance Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min Distance", f"{df['distance_km'].min():.2f} km")
                    with col2:
                        st.metric("Mean Distance", f"{df['distance_km'].mean():.2f} km")
                    with col3:
                        st.metric("Max Distance", f"{df['distance_km'].max():.2f} km")

                # Distributions
                st.subheader("Numeric Feature Distributions")

                numeric_features = []
                if 'age' in df.columns:
                    numeric_features.append('age')
                if 'amt' in df.columns:
                    numeric_features.append('amt')
                if 'trans_hour' in df.columns:
                    numeric_features.append('trans_hour')
                if 'distance_km' in df.columns:
                    numeric_features.append('distance_km')

                for feature in numeric_features:
                    fig = px.histogram(df, x=feature, title=f'Distribution of {feature}')
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header("Fraud Analysis")

            df = st.session_state.get('df_prepared', df_raw)

            if 'is_fraud' not in df.columns:
                st.error("'is_fraud' column not found in dataset")
                return

            # Fraud distribution
            st.subheader("Fraud Distribution")
            fraud_counts = df['is_fraud'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Transactions", f"{fraud_counts.get(0, 0):,}")
            with col2:
                st.metric("Fraudulent Transactions", f"{fraud_counts.get(1, 0):,}")

            # Amount distribution by fraud status
            if 'amt' in df.columns:
                st.subheader("Transaction Amount Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['is_fraud']==1]['amt'],
                    name='Fraudulent',
                    nbinsx=50,
                    opacity=0.7,
                    marker_color=JEDHA_VIOLET
                ))
                fig.update_layout(
                    title='Fraudulent Transaction Amounts',
                    xaxis_title='Amount',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Hour distribution
            if 'trans_hour' in df.columns:
                st.subheader("Fraud Distribution by Hour")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['is_fraud']==1]['trans_hour'],
                    name='Fraudulent',
                    nbinsx=24,
                    opacity=0.7,
                    marker_color=JEDHA_VIOLET
                ))
                fig.update_layout(
                    title='Fraudulent Transactions by Hour',
                    xaxis_title='Hour of Day',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Category distribution
            if 'category' in df.columns:
                st.subheader("Fraud Distribution by Category")
                fraud_by_category = df[df['is_fraud']==1].groupby('category').size().reset_index(name='count')
                fraud_by_category = fraud_by_category.sort_values('count', ascending=False)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=fraud_by_category['category'],
                    y=fraud_by_category['count'],
                    marker_color=JEDHA_VIOLET
                ))
                fig.update_layout(
                    title='Fraudulent Transactions by Category',
                    xaxis_title='Category',
                    yaxis_title='Number of Frauds',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
