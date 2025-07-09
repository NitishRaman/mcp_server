# 📄 5 Supabase.py — Cleaned, Commented & Modularized
# Streamlit page to trigger the Supabase data ingestion UI flow

import streamlit as st
from app.supabase_utils import handle_supabase_connection, render_supabase_streamlit_ui
from app.data_ingestor_server import fetch_table_schema

# === 🚀 Entry Point === #

def render_supabase_ingestion_page():
    """
    Entry point called by the Supabase Streamlit page.

    This function handles:
    - Prompting user for Supabase credentials
    - Connecting to the database
    - Fetching table list and schema
    - Delegating full UI rendering to `render_supabase_streamlit_ui`
    """

    conn, dataset_name, tables = handle_supabase_connection()

    if conn and dataset_name and tables:
        # Build schema dict per table
        schema_dict = {t: fetch_table_schema(conn, t) for t in tables}

        # Launch full UI flow
        render_supabase_streamlit_ui(dataset_name, conn, schema_dict, tables)

# === 🚀 Streamlit Page Setup === #
st.set_page_config(
    page_title="📡 Supabase Ingest",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 🧭 Page Content === #
st.title("📡 Ingest Data from Supabase")

# 🔄 This function calls the full ingestion UI pipeline
# Handles:
# 1. Credential input
# 2. Table preview, CSV download
# 3. LLM-based schema description
# 4. PK/FK detection
# 5. Vectorization to Chroma
# 6. NL → SQL querying
render_supabase_ingestion_page()
