# ✅ Supabase Data Ingestor (Streamlit-based UI with table preview + optional save + download)
import os
import json
import zipfile
import glob
import chromadb
import psycopg2
import sqlite3
import pandas as pd
import streamlit as st
from io import BytesIO
from psycopg2.extras import RealDictCursor
from app.core.config import FILES_ROOT
from app.llm_client import LLMClient
from app.data_ingestor import ensure_dataset_folder
from app.schema_diagram import generate_supabase_diagram, generate_schema_diagram_with_eralchemy
from app.schema_vectorizer import vectorize_database





# === 🔌 Supabase Connection & Fetch === #

# -----------------------------
# ✅ Dataset Path Safety Guard 
# -----------------------------

# --- 🔐 Dataset Path Handling ---
# 🟩 Use dataset_name from session state if available
if "dataset_name" in st.session_state:
    dataset_name = st.session_state["dataset_name"]
    folder = os.path.join(FILES_ROOT, dataset_name)
else:
    dataset_name = ""
    folder = ""


# --- Function: connect_supabase ---
def connect_supabase(host, port, dbname, user, password, sslmode="require"):
    """
    Establishes a connection to a Supabase (PostgreSQL) database.
    Returns a psycopg2 connection using RealDictCursor.
    """
    DSN = f"host={host} port={port} dbname={dbname} user={user} password={password} sslmode={sslmode}"
    return psycopg2.connect(DSN, cursor_factory=RealDictCursor)


# --- Function: fetch_supabase_tables ---
def fetch_supabase_tables(conn):
    """
    Fetch list of all public table names from a Supabase (PostgreSQL) connection.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public';
    """)
    return [r['table_name'] for r in cur.fetchall()]


# --- Function: fetch_table_sample ---
def fetch_table_sample(conn, table_name, limit=100):
    """
    Fetch sample rows from a table in Supabase.
    Returns a pandas DataFrame.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
    return pd.DataFrame(cur.fetchall())


# --- Function: fetch_table_schema ---
def fetch_table_schema(conn, table_name):
    """
    Retrieve column names and data types for a given table in Supabase.
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table_name}';
    """)
    return cur.fetchall()


# --- Function: save_dataset_to_sqlite_with_schema ---
def save_dataset_to_sqlite_with_schema(dataset_name: str) -> str:
    """
    Builds SQLite DB using combined schema + metadata.json that includes PK/FK info.
    Requires:
        - {dataset_name}_combined_schema.json
        - {dataset_name}_metadata.json
        - CSVs for each table
    Returns the path to the built SQLite DB.
    """
    dataset_path = os.path.join(FILES_ROOT, dataset_name)
    db_path = os.path.join(dataset_path, f"{dataset_name}.db")

    # Load schema and metadata
    with open(os.path.join(dataset_path, f"{dataset_name}_combined_schema.json"), "r") as f:
        schema = json.load(f)
    with open(os.path.join(dataset_path, f"{dataset_name}_metadata.json"), "r") as f:
        metadata = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    # Step 1: Create all tables
    for table, columns in schema.items():
        col_defs = []
        pks = metadata.get(table, {}).get("primary_keys", [])
        fk_defs = []

        for col in columns:
            colname = col["column_name"]
            coltype = col.get("data_type", "TEXT").upper()
            col_defs.append(f"{colname} {coltype}")

        # Foreign keys from metadata
        for fk in metadata.get(table, {}).get("foreign_keys", []):
            try:
                src_col, ref = fk.split("→")
                src_col = src_col.strip()
                ref_table, ref_col = [x.strip() for x in ref.split(".")]
                fk_defs.append(f"FOREIGN KEY({src_col}) REFERENCES {ref_table}({ref_col})")
            except Exception as e:
                print(f"⚠️ Skipping malformed FK: {fk} in table: {table}")

        if pks:
            col_defs.append(f"PRIMARY KEY ({', '.join(pks)})")
        col_defs.extend(fk_defs)

        ddl = f"CREATE TABLE IF NOT EXISTS {table} (\n  " + ",\n  ".join(col_defs) + "\n);"
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(ddl)

    # Step 2: Insert data (manually ordered for FK safety)
    insert_order = ["teacher", "class", "student", "enrollment", "marks"]
    for table in insert_order:
        csv_path = os.path.join(dataset_path, f"{table}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.to_sql(table, conn, if_exists="append", index=False)

    # Step 3: Foreign key check
    fk_violations = list(cursor.execute("PRAGMA foreign_key_check"))
    if fk_violations:
        print("❌ Foreign key constraint violations:")
        for row in fk_violations:
            print(row)
        raise RuntimeError("Foreign key check failed — see logs for details.")

    conn.commit()
    conn.close()
    return db_path

    
# --- Function: save_dataset_to_sqlite ---
def save_dataset_to_sqlite(dataset_name: str) -> str:
    """
    Builds a basic SQLite database by importing all CSVs in the dataset folder.
    No foreign key constraints are created.
    """
    dataset_path = os.path.join(FILES_ROOT, dataset_name)
    db_path = os.path.join(dataset_path, f"{dataset_name}.db")

    conn = sqlite3.connect(db_path)
    for fname in os.listdir(dataset_path):
        if fname.endswith(".csv"):
            table_name = fname.replace(".csv", "")
            csv_path = os.path.join(dataset_path, fname)
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return db_path



