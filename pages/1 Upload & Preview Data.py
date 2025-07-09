"""
📄 1 Upload & preview page: allows users to upload a dataset file, preview its contents,
and optionally ingest it into a SQLite database.
"""

import streamlit as st
import os
from app.core.config import FILES_ROOT
from app.data_ingestor import (
    ensure_dataset_folder,
    save_and_ingest_file,
    read_excel,
    read_csv_from_bytes,
    read_zip,
    read_json,
    read_xml,
    read_db
)
from pathlib import Path

# Page configuration
st.set_page_config(page_title="📁 Upload & Preview", layout="wide")

#Title
st.title("📁 Upload Data and Preview")

# Section: Dataset name input and file upload
st.header("1. Select Dataset and Upload File")
dataset_name = st.text_input("Enter dataset name", help="Name will be used to save files and the SQLite database.")
file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "zip", "json", "xml", "db"])

if dataset_name and file:
    # Ensure dataset folder exists and save uploaded file
    folder = ensure_dataset_folder(dataset_name)
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    st.success(f"✅ Saved to {file_path}")
    
    # Detect file extension and parse accordingly
    ext = Path(file_path).suffix.lower()
    df_store = {}

    try:
        if ext == ".csv":
            df = read_csv_from_bytes(file.getvalue())
            df_store[file.name] = df

        elif ext in [".xlsx", ".xls"]:
            sheets = read_excel(file_path)
            for sheet_name, df in sheets.items():
                df_store[f"{file.name}::{sheet_name}"] = df

        elif ext == ".zip":
            extracted = read_zip(file_path, dataset_name)
            df_store.update(extracted)

        elif ext == ".json":
            df = read_json(file_path)
            df_store[file.name] = df

        elif ext == ".xml":
            df = read_xml(file_path)
            df_store[file.name] = df

        elif ext == ".db":
            db_tables = read_db(file_path, dataset_name)
            for tbl, df in db_tables.items():
                df_store[tbl] = df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

    # Section: Preview the data
    if df_store:
        keys = list(df_store.keys())
        selected_key = st.selectbox("Choose table/sheet to preview", options=keys)
        df = df_store[selected_key]
        st.dataframe(df)
        st.write(f"Shape: {df.shape}")
        st.write("Columns:", list(df.columns))

        # Section: Option to ingest into SQLite
        st.header("3. Ingest to SQLite")
        if st.button("Ingest and Create SQLite DB"):
            db_path = save_and_ingest_file(file_path, dataset_name)
            st.success(f"Database created at: {db_path}")
