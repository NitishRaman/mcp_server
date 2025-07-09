# 3 Chart Generator.py

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from app.core.config import FILES_ROOT
from app.chart_generator import plot_chart

#---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*

st.set_page_config(page_title="📊 Chart Generator", layout="wide")
st.title("📊 Generate Charts from Uploaded Tables")

#---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*

# === 📦 Helper Functions === #

def list_csv_tables(dataset_name):
    
    """List CSV files (without extension) available in the dataset folder."""
    
    folder = os.path.join(FILES_ROOT, dataset_name)
    return [f.replace('.csv', '') for f in os.listdir(folder) if f.endswith('.csv')]
    
#---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*

def load_table(dataset_name, table_name):
    
    """Load a CSV file into a pandas DataFrame."""
    
    path = os.path.join(FILES_ROOT, dataset_name, f"{table_name}.csv")
    return pd.read_csv(path)

#---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*

# === 🚀 Streamlit UI === #

st.header("1. Select Dataset and Table")
dataset_name = st.text_input("Dataset name (used in upload page)")

if dataset_name:
    dataset_path = os.path.join(FILES_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        st.error("❌ Dataset folder not found. Please upload and ingest first.")
    else:
        tables = list_csv_tables(dataset_name)
        if not tables:
            st.warning("⚠️ No tables found. Make sure ingestion created .csv files.")
        else:
            table_name = st.selectbox("Select table", tables)

            df = load_table(dataset_name, table_name)
            st.dataframe(df.head())
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", list(df.columns))

            st.header("2. Choose Chart Options")
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"])
            x_col = st.selectbox("X-axis column", df.columns)
            y_col = None
            if chart_type != "Pie Chart":
                y_col = st.selectbox("Y-axis column", df.columns)

            st.header("3. Generate Chart")
            if st.button("Generate"):
                try:
                    fig = plot_chart(df, chart_type, x_col, y_col)  # ✅ get the fig
                    st.pyplot(fig)  # ✅ explicitly show it in Streamlit
                except Exception as e:
                    st.error(f"❌ Chart generation failed: {e}")


#---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*---*