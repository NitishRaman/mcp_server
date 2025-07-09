# 📄 2 Description & Schema Generator.py
# This Streamlit page allows users to generate a schema description and ER diagram from an uploaded SQLite database

import streamlit as st
import os
import json
import sqlite3
import chromadb
import base64
import streamlit.components.v1 as components
from urllib.parse import quote
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 📦 Import core app modules
from app.core.config import FILES_ROOT
from app.schema_describer import describe_database, llm_infer_keys_from_description
from app.schema_diagram import generate_schema_diagram_with_eralchemy
from app.schema_vectorizer import vectorize_database

os.environ["ANONYMIZED_TELEMETRY"] = "0"

# 🚀 Set up the Streamlit page
st.set_page_config(page_title="🧬 Schema", layout="wide")
st.title("🧬 Describe Schema and Generate Diagram")

# ✅ 1. Choose dataset (based on available folders)
dataset_options = [
    d for d in os.listdir(FILES_ROOT)
    if os.path.isdir(os.path.join(FILES_ROOT, d)) and not d.lower().startswith("chroma")
]
dataset_name = st.selectbox("Choose dataset:", dataset_options if dataset_options else ["No datasets found"])


# ✅ Early exit if no name is provided
if not dataset_name:
    st.warning("⚠️ No datasets found in the files directory.")
    st.stop()


# 📁 Resolve path to dataset folder
dataset_path = os.path.join(FILES_ROOT, dataset_name) if dataset_name else None
db_path = None

# derive collection name once, before any vector/query logic
if dataset_path and os.path.exists(dataset_path):
    collection_name = f"schema_{dataset_name}"
else:
    collection_name = None

# 🔍 Search for .db file in the dataset folder
if dataset_path and os.path.exists(dataset_path):
    db_files = [f for f in os.listdir(dataset_path) if f.endswith(".db")]
    if db_files:
        db_path = os.path.join(dataset_path, db_files[0])

# ✅ Main UI logic if DB is found
if db_path:
    st.success(f"✅ Found DB: {os.path.basename(db_path)}")
    if st.button("Generate Schema Description and Diagram"):
        describe_database(db_path, dataset_path)
        generate_schema_diagram_with_eralchemy(db_path, os.path.join(dataset_path, f"{dataset_name}_schema.pdf"))
        st.success("✅ Description and diagram generated")
        desc_file = os.path.join(dataset_path, f"{dataset_name}_description.txt")
        llm_infer_keys_from_description(desc_file)
        vectorize_database(desc_file, dataset_path)

    # 📄 Display the schema description text (if available)
    schema_txt = os.path.join(dataset_path, f"{dataset_name}_description.txt")
    if os.path.exists(schema_txt):
        with open(schema_txt, "r", encoding="utf-8") as f:
            content = f.read()
            st.markdown(f"<div style='max-height: 400px; overflow-y: auto'>{content}</div>", unsafe_allow_html=True)

    # 📌 Download button for ER diagram (PDF)
    pdf_path = os.path.join(dataset_path, f"{dataset_name}_schema.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button("Download ER Diagram", f, file_name=f"{dataset_name}_schema.pdf")
else:
    if dataset_name:
        st.warning("❌ No DB found. Please make sure to upload and ingest data first.")

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# 🔍 Semantic Schema Search
st.subheader("🔍 Semantic Schema Search")
query = st.text_input("Enter a question like 'Which table has email?', 'columns about pricing', or 'foreign key between orders and customers'")

if query and st.button("Search"):
    try:
        embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name=f"schema_{dataset_name}")

        results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        for idx, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
            with st.expander(f"🔎 Result {idx + 1} - {meta.get('type', '')} ({meta.get('table', '')})"):
                st.markdown(f"**📜 Type:** `{meta.get('type', '')}`")
                st.markdown(f"**💃️ Table:** `{meta.get('table', '')}`")
                if meta.get('column'):
                    st.markdown(f"**🔹 Column:** `{meta.get('column', '')}`")
                if meta.get('relation'):
                    st.markdown(f"**🔗 Relation:** `{meta.get('relation', '')}`")
                if meta.get("primary_key"):
                    src = meta.get("source", "unknown")
                    label = "🔑 **Primary Key**"
                    label += f" _(source: {src})_" if src else ""
                    st.markdown(label)
                if meta.get("foreign_key"):
                    src = meta.get("source", "unknown")
                    label = "🔗 **Foreign Key**"
                    label += f" _(source: {src})_" if src else ""
                    st.markdown(label)
                st.markdown(f"**🎯 Score:** `{dist:.3f}`")
                st.markdown("---")
                st.markdown(f"<div style='max-height: 300px; overflow-y: auto'>{doc}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ No vector DB collection found for `{dataset_name}`. Please run vectorization first.")
        st.stop()

# 📊 Show Parsed Tags/Relationships if Available
if dataset_name and dataset_path:
    tags_path = os.path.join(dataset_path, f"{dataset_name}_tags.json")
    metadata_path = os.path.join(dataset_path, f"{dataset_name}_metadata.json")
    if os.path.exists(tags_path):
        st.subheader("🕷️ Parsed Tags & Relationships")
        with open(tags_path, "r", encoding="utf-8") as f:
            tags = json.load(f)

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)

        for table, result in tags.items():
            with st.expander(f"📜 {table}"):
                st.markdown("### 🔑 PK/FK & its Source")
                table_meta = metadata.get(table, {})
                source = table_meta.get("source", "unknown")
                if "primary_keys" in table_meta:
                    st.markdown(f"<span style='color:green;font-weight:bold'>Primary Keys: {', '.join(table_meta['primary_keys'])}</span>", unsafe_allow_html=True)
                if "foreign_keys" in table_meta:
                    st.markdown("<span style='color:red;font-weight:bold'>Foreign Keys:</span>", unsafe_allow_html=True)
                    for fk in table_meta["foreign_keys"]:
                        st.markdown(f"🔗 {fk}")
                st.markdown(f"<span style='font-size: 90%; color: gray;'>🧭 Source: <code>{source}</code></span>", unsafe_allow_html=True)

                st.markdown("### 🧩 Tags and Business Purpose")
                lines = result.splitlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("Tags") or line.startswith("Keywords"):
                        st.markdown(f"🧩 **{line}**")
                    elif "→" in line or "->" in line:
                        st.markdown(f"🔗 *{line}*")
                    elif line.lower().startswith("business purpose"):
                        st.markdown(f"💼 **{line}**")
                    elif line and not line.startswith("Primary") and not line.startswith("Foreign"):
                        st.markdown(line)
    else:
        st.info("ℹ️ Tags not generated yet. Click 'Generate Schema Description and Diagram' above.")

    stats_path = os.path.join(dataset_path, f"{dataset_name}_stats.json")
    metadata_path = os.path.join(dataset_path, f"{dataset_name}_metadata.json")

    if os.path.exists(stats_path):
        st.subheader("📊 Column Stats Viewer")

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)

        table = st.selectbox("Choose table to view stats:", sorted(stats.keys()))
        if table:
            table_meta = metadata.get(table, {})
            pk_set = set(table_meta.get("primary_keys", []))
            fk_set = set(fk.split(" → ")[0] for fk in table_meta.get("foreign_keys", []))

            for col, detail in stats[table].items():
                st.markdown(f"#### 🔹 {col}")
                if col in pk_set:
                    st.markdown("🔑 This column is a primary key")
                if col in fk_set:
                    st.markdown("🔗 This column is a foreign key")
                st.json(detail, expanded=False)
    else:
        st.info("ℹ️ Column stats not available yet. Click 'Generate Schema Description and Diagram' above.")