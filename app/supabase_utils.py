# supabase_utils.py
# 🔌 Tools for Supabase connection, schema generation, vectorization, and Streamlit UI
import json
import os
import pandas as pd
import streamlit as st
import base64   
from app.schema_describer import describe_column_naturally
from app.llm_client import LLMClient
from app.core.config import FILES_ROOT
from app.data_ingestor_server import fetch_table_sample, save_dataset_to_sqlite_with_schema, connect_supabase, fetch_supabase_tables
from app.schema_describer import llm_infer_keys_from_description
from app.schema_diagram import generate_supabase_diagram, generate_schema_diagram_with_eralchemy
from app.schema_vectorizer import vectorize_database
from app.natural_sql_query import generate_sql_from_prompt
    


# --- Function: generate_description_from_supabase_schema ---	
def generate_description_from_supabase_schema(dataset_name: str, schema_dict: dict):
    """
    Uses LLM to describe schema from Supabase JSON, including table summaries,
    column types, and natural language explanations. Saves multiple output files.
    """
	
    folder = os.path.join(FILES_ROOT, dataset_name)
    os.makedirs(folder, exist_ok=True)

    description_path = os.path.join(folder, f"{dataset_name}_description.txt")
    stats_path = os.path.join(folder, f"{dataset_name}_stats.json")
    summary_path = os.path.join(folder, f"{dataset_name}_tablesummary.txt")

    docs = []
    table_summaries = []
    stats_json = {}

    tables = list(schema_dict.keys())
    print(f"📋 Describing {len(tables)} tables from Supabase...")

    for i, table in enumerate(tables, 1):
        columns = schema_dict[table]
        col_names = [col["column_name"] for col in columns]
        col_types = [col["data_type"] for col in columns]

        # === Table-level summary
        prompt = f"""
You are a database analyst. A table named `{table}` has the following columns:

""" + "\n".join([f"- {name} ({dtype})" for name, dtype in zip(col_names, col_types)]) + """
        
In 1-2 lines, summarize what this table likely contains. Use plain English for business users.
"""
        try:
            llm = LLMClient()
            response = llm.chat_completion([{'role': 'user', 'content': prompt}])
            table_summary = response.strip().replace("\n", " ")
        except Exception as e:
            table_summary = f"⚠️ Failed to summarize: {e}"

        summary_block = f"### Table {i}/{len(tables)}: {table}\n**Summary:** {table_summary}\n"
        table_summaries.append(summary_block)
        table_doc = summary_block + "\n## Columns:\n"

        col_stats = {}
        for col in columns:
            name, dtype = col["column_name"], col["data_type"]
            col_doc = f"- **{name}** *(type: {dtype})*"

            # 🔍 Add LLM-based natural explanation
            sample_vals = col.get("sample_values", [])  # optional
            try:
                explanation = describe_column_naturally(table, name, dtype, sample_vals)
                col_doc += f"\n    ↪ {explanation}"
            except:
                col_doc += "\n    ↪ ⚠️ Failed to generate explanation"

            table_doc += col_doc + "\n"
            col_stats[name] = {"type": dtype}

        stats_json[table] = col_stats
        docs.append(table_doc)

    # === Save outputs
    with open(description_path, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    with open(stats_path, "w", encoding="utf-8") as jf:
        json.dump(stats_json, jf, indent=2)
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("\n\n".join(table_summaries))

    print(f"✅ Description saved to: {description_path}")
    print(f"✅ Stats saved to: {stats_path}")
    print(f"✅ Table summaries saved to: {summary_path}")

    return description_path  # <- for next phase


# --- Function: get_supabase_or_inferred_keys ---
def get_supabase_or_inferred_keys(conn, dataset_name: str, tables: list[str]) -> dict:
    """
    Decide whether to use Supabase-extracted keys or fall back to LLM-inferred keys.

    Args:
        description_path (str): Path to schema description text file
        supabase_keys (dict): Optional override for PK/FK info

    Returns:
        dict: PK/FK structure for each table
    """
    pkfk = {}

    # Supabase: actual primary keys
    cur = conn.cursor()
    for table in tables:
        cur.execute(f"""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{table}'::regclass AND i.indisprimary;
        """)
        pks = [r['attname'] for r in cur.fetchall()]
        pkfk[table] = {
            "primary_keys": pks,
            "foreign_keys": [],
            "source": "supabase" if pks else "unknown"
        }

    # Supabase: foreign keys
    cur.execute("""
        SELECT
            tc.table_name, kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_col
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
        WHERE constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
    """)
    for row in cur.fetchall():
        src_table = row['table_name']
        ref_line = f"{row['column_name']} → {row['ref_table']}.{row['ref_col']}"
        if src_table in pkfk:
            pkfk[src_table]['foreign_keys'].append(ref_line)

    # fallback
    inferred_path = os.path.join(FILES_ROOT, dataset_name, f"{dataset_name}_inferred_keys.json")
    if os.path.exists(inferred_path):
        with open(inferred_path, "r") as f:
            inferred = json.load(f)
        for table in tables:
            if not pkfk.get(table) or not pkfk[table]["primary_keys"]:
                pkfk[table] = inferred.get(table, {
                    "primary_keys": [],
                    "foreign_keys": [],
                    "source": "inferred"
                })

    return pkfk


# --- Function: handle_supabase_connection ---
def handle_supabase_connection():
    """
    Complete pipeline for Supabase ingestion: fetch schema, generate description,
    infer PK/FK, save DB, generate diagrams and vector DB.

    Args:
        params (dict): Connection parameters for Supabase (host, db, user, etc.)
        dataset_name (str): Dataset identifier
    """
    st.header("📥 Supabase Data Ingestion")

    host = st.text_input("Host", placeholder="xyz.supabase.co")
    port = st.text_input("Port", value="5432")
    dbname = st.text_input("Database", value="postgres")
    user = st.text_input("Username", value="postgres")
    password = st.text_input("Password", type="password")
    sslmode = st.selectbox("SSL Mode", ["require", "disable"], index=0)
    dataset = st.text_input("Dataset Name")

    if st.button("Ingest Data"):
        st.session_state['supabase_config'] = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password,
            'sslmode': sslmode,
        }
        st.session_state['dataset_name'] = dataset
        st.session_state['ingest_triggered'] = True

    if st.session_state.get('ingest_triggered'):
        config = st.session_state['supabase_config']
        dataset_name = st.session_state['dataset_name']
        conn = connect_supabase(**config)
        tables = fetch_supabase_tables(conn)
        st.success(f"✅Connected to Supabase. Tables: {tables}")
        return conn, dataset_name, tables

    return None, None, None


# --- Function: render_supabase_streamlit_ui ---
def render_supabase_streamlit_ui(dataset_name, conn, schema_dict, tables):
    """
    Full Streamlit-based UI for working with an ingested Supabase dataset.

    Features:
    - Preview any selected table (schema + rows)
    - Save individual or all tables as CSV
    - Download combined schema JSON
    - Generate and preview LLM-based schema description
    - Infer PK/FK keys using LLM
    - Get PK/FK keys from Supabase (fallbacks to inferred)
    - Vectorize schema and relationships into ChromaDB
    - Accept natural language prompt and run SQL via LLM on Supabase

    Args:
        dataset_name (str): Name of the dataset (folder name)
        conn (psycopg2.connection): Active DB connection
        schema_dict (dict[str, list[dict]]): Combined schema per table
        tables (list[str]): List of available tables in Supabase
    """

    folder = os.path.join(FILES_ROOT, dataset_name)
    os.makedirs(folder, exist_ok=True)  # 🔧 Ensure the folder exists before saving

    selected_table = st.selectbox("📋Select table to preview", tables)
    if selected_table:
        df = fetch_table_sample(conn, selected_table)
        st.dataframe(df.head(20))
        st.json(schema_dict[selected_table])

        if st.button("💾Save CSV", key=f"csv_{selected_table}"):
            df.to_csv(os.path.join(folder, f"{selected_table}.csv"), index=False)
            st.success("CSV saved.")

    if st.button("💾Save All Tables to CSV"):
        for t in tables:
            df = fetch_table_sample(conn, t)
            df.to_csv(os.path.join(folder, f"{t}.csv"), index=False)
        st.success("All tables saved.")

    if st.button("🗓️ Download Combined Schema"):
        schema_path = os.path.join(folder, f"{dataset_name}_combined_schema.json")
        with open(schema_path, "w") as f:
            json.dump(schema_dict, f, indent=2)
        with open(schema_path, "rb") as f:
            st.download_button("📄Download JSON", f, file_name=os.path.basename(schema_path))

    if st.button("📒 Generate LLM Description"):
        path = generate_description_from_supabase_schema(dataset_name, schema_dict)
        st.success("LLM Description generated.")

    # 📄 Display the schema description text (if available)
    schema_txt = os.path.join(folder, f"{dataset_name}_description.txt")
    if os.path.exists(schema_txt):
        with open(schema_txt, "r", encoding="utf-8") as f:
            content = f.read()
            st.markdown(f"<div style='max-height: 400px; overflow-y: auto; white-space: pre-wrap'>{content}</div>", unsafe_allow_html=True)

    if st.button("🔑 Infer PK/FK via LLM"):
        path = os.path.join(folder, f"{dataset_name}_description.txt")
        keys = llm_infer_keys_from_description(path)
        st.json(keys)

    if st.button("🔐 Get PK/FK from Supabase + Fallback"):
        keys = get_supabase_or_inferred_keys(conn, dataset_name, tables)
        st.json(keys)

    if st.button("🚀 Vectorize to Chroma"):
        desc_path = os.path.join(folder, f"{dataset_name}_description.txt")
        keys = get_supabase_or_inferred_keys(conn, dataset_name, tables)
        vectorize_database(desc_path, folder, override_keys=keys)
        st.success("Chroma vectorization complete.")
    
    
    if st.button("🧭 Generate ER Diagram"):
        keys = get_supabase_or_inferred_keys(conn, dataset_name, tables)
        relationships = []
        for table, meta in keys.items():
            for fk in meta.get("foreign_keys", []):
                if "→" in fk:
                    src, tgt = fk.split("→")
                    source_column = src.strip()
                    target_table, target_column = tgt.strip().split(".")
                    relationships.append({
                        "source_table": table,
                        "source_column": source_column,
                        "target_table": target_table,
                        "target_column": target_column
                    })

        png_path = os.path.join(folder, "schema_diagram")
        generate_supabase_diagram(schema_dict, relationships, png_path, pk_fk_map=keys)
        st.image(png_path + ".png", caption="📌 Supabase ER Diagram")

    
    # 📦 Build SQLite database from CSVs + schema
    if st.button("🗃️ Build SQLite DB from CSVs"):
        try:
            db_path = save_dataset_to_sqlite_with_schema(dataset_name)
            st.session_state["db_path"] = db_path  # ✅ save for later
            st.success(f"✅ SQLite database saved at: {db_path}")
        except Exception as e:
            st.error(f"❌ Failed to create SQLite DB: {e}")
    
    # Button to generate and show ER diagram
    pdf_path = os.path.join(folder, f"{dataset_name}_schema.pdf")

    if st.button("📐 Generate ER Diagram"):
        db_path = os.path.join(folder, f"{dataset_name}.db")  # ✅ read from session
        if not db_path or not os.path.exists(db_path):
            st.warning("⚠️ Please build the SQLite DB first.")
        else:
            try:
                generate_schema_diagram_with_eralchemy(db_path, pdf_path)
                st.success("✅ Diagram generated successfully!")
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Failed to generate diagram: {e}")


    user_prompt = st.text_input("💬 Ask a question (NL to SQL):")
    if st.button("🌟 Run Query"):
        with st.spinner("🤖 Thinking like a database analyst..."):
            sql, result = generate_sql_from_prompt(dataset_name, user_prompt)
            result = run_sql_on_supabase(sql, conn)

        st.code(sql, language="sql")
        if isinstance(result, str) and result.startswith("❌"):
            st.error(result)
        else:
            st.dataframe(result)


# --- Function: render_sql_on_supabase ---
def run_sql_on_supabase(sql: str, conn):
    """
    Executes a SQL SELECT query on the connected Supabase database.

    Args:
        sql (str): SQL query string to execute
        conn (psycopg2.connection): Active Supabase database connection

    Returns:
        pd.DataFrame or str: Query result as DataFrame, or error message
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return pd.DataFrame(result)
    except Exception as e:
        return f"❌ SQL execution failed: {e}"

