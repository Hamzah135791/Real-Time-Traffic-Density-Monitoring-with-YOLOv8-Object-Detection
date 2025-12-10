# dashboard.py
import streamlit as st
import pandas as pd
import psycopg2
import time

# CONFIG DATABASE
DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "trafficdb",
    "user": "admin",
    "password": "admin123"
}

# FUNCTION LOAD DATA
def load_data():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
        SELECT *
        FROM traffic_weather
        ORDER BY ts DESC
        LIMIT 100
    """, conn)
    conn.close()
    return df

# STREAMLIT DASHBOARD
st.set_page_config(page_title="Traffic & Weather Dashboard", layout="wide")
st.title("🚦 Traffic & Weather Dashboard")

# Auto-refresh setiap 30 detik
refresh_interval = 30

while True:
    df = load_data()
    
    # Grafik kendaraan
    st.subheader("Vehicle Count")
    st.line_chart(df[["car", "motor", "bus", "truck"]].rename_axis("Time").iloc[::-1])

    # Grafik cuaca
    st.subheader("Weather")
    st.line_chart(df[["temperature", "humidity", "wind_speed"]].iloc[::-1])

    # Tabel data
    st.subheader("Recent Records")
    st.dataframe(df.iloc[::-1])

    st.info(f"Data refresh every {refresh_interval} seconds...")

    time.sleep(refresh_interval)
    st.experimental_rerun()
