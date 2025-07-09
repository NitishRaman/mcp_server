# 🚀 MCP Server

A modular Python-based platform to **ingest**, **describe**, **query**, and **visualize** structured datasets using **LLMs**, **vector databases**, and an intuitive **Streamlit UI**.

---

## 📋 Overview & Use Case

MCP Server is ideal for:

* 📊 Data analysts exploring unknown datasets
* 🧠 LLM-based schema summarization
* 🧾 Prompt-to-SQL generation with conversational context
* 🌐 Teams working with Supabase or local data pipelines

---

## 🔧 Key Features

* 📥 Upload & parse CSV, Excel, ZIP, JSON, XML, and SQLite
* 🧠 Generate column descriptions, PK/FK inference, and tags via LLM
* 🔍 Vectorize schemas with `sentence-transformers` and ChromaDB
* 🧾 Translate prompts to SQL (with logs)
* 💬 Multi-turn chat with LLM to explore database
* 📈 Build visual charts from structured data
* 🛢️ Push SQLite tables to Supabase
* 🌐 Streamlit UI + FastAPI API backend

---

## 🗂️ Project Structure

```
📁 mcp_server_project/
├── 📁 app/                         # Core modules
│   ├── core/                      # Constants and config
│   ├── chromadb_prompt_utils.py
│   ├── chart_generator.py
│   ├── data_ingestor.py
│   ├── data_ingestor_server.py
│   ├── llm_client.py
│   ├── natural_sql_query.py
│   ├── schema_describer.py
│   ├── schema_diagram.py
│   ├── schema_vectorizer.py
│   └── supabase_utils.py
│
├── 📁 mcp_server/
│   └── files/                     # Generated files per dataset
│       └── chroma/                # Vector DB store
│
├── 📁 sample_datasets/            # Sample datasets to test flow
├── 📁 pages/                      # Streamlit UI pages (modular)
│   ├── 1 Upload & Preview.py
│   ├── 2 Description & Schema Generator.py
│   ├── 3 Prompt to SQL.py
│   ├── 4 Chart Generator.py
│   ├── 5 Supabase.py
│   └── 6 Interactive Prompt Chat.py
│
├── Home.py                       # Landing page
├── app_server.py                 # FastAPI backend
├── requirements.txt
├── run.bat                       # One-click launcher
├── mcp-documentation.pdf         # Full documentation
├── mcp-presentation.pdf          # Project presentation deck
└── README.md                     # This file
```

---

## ⚙️ Setup

### ✅ One-click (Windows)

```bash
run.bat
```

This will:

* Create `.venv`
* Install dependencies
* Launch ChromaDB, pull Ollama, start backend + UI

### 🐧 Manual (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app_server:app --reload
```

---

## 📊 Streamlit Modules

### 1. Upload & Preview

* `save_and_ingest_file()`
* Helpers: `read_csv_from_bytes()`, `read_zip()`, etc.

### 2. Description & Schema

* `describe_database()`
* Helpers: `llm_infer_keys_from_description()`, `vectorize_database()`

### 3. Prompt to SQL

* `generate_sql_from_prompt()`
* Helpers: `run_query_on_db()`, `log_successful_query()`

### 4. Chart Generator

* `plot_chart(df, chart_type, x_col, y_col)`

### 5. Supabase

* `handle_supabase_connection()`
* Helpers: `fetch_supabase_tables()`, `fetch_table_sample()`

### 6. Interactive Prompt Chat

* `generate_sql_from_prompt()`
* Helpers: `generate_sql_strict()`, `log_query()`

---

## 💾 Output Directory (per dataset)

```
📁 files/my_dataset/
├── my_dataset.db                  # SQLite version
├── *.csv / *.json / *.xml         # Original input
├── my_dataset_description.txt     # LLM-generated schema summary
├── my_dataset_metadata.json       # PK, FK, tags, etc.
├── query_log.txt                  # Prompt-to-SQL history
├── interactive_query_log.txt      # Chat history
├── my_dataset_supabase.png        # ER Diagram (optional)
```

---

## 🔎 Vector Embedding (ChromaDB)

Uses `sentence-transformers` to embed column names, types, and summaries. Stored per dataset under collection `schema_<dataset>`.

✅ Enables context-aware SQL, semantic search, and chat memory.

---

## 🧪 Run & Test

```bash
streamlit run Home.py            # Run full UI
uvicorn app_server:app --reload  # Backend API only
```

---

## 📦 Key Dependencies (`requirements.txt`)

### 🔹 Core

* `pandas`, `numpy`, `openpyxl`, `sqlparse`, `pyyaml`, `xmlschema`, `jsonschema`, `tqdm`, `jinja2`, `chardet`

### 🔹 Visualization

* `matplotlib`, `graphviz`, `eralchemy`

### 🔹 UI & Backend

* `streamlit>=1.30`, `fastapi`, `uvicorn`, `ipywidgets`, `ipython`, `notebook`

### 🔹 LLM + Vector DB

* `openai`, `ollama`, `gpt4all`, `chromadb>=0.5.0`, `sentence-transformers`, `torch`, `scikit-learn`

### 🔹 SQL / DB

* `sqlite-utils`, `psycopg2-binary`

### 🔹 Utilities

* `typing-extensions`

---

## ❗ Troubleshooting

| Issue                | Solution                              |
| -------------------- | ------------------------------------- |
| Port 8000 in use     | Kill existing app or change port      |
| Ollama pull fails    | Make sure Ollama CLI is installed     |
| ChromaDB won't start | Activate venv + ensure port open      |
| No module named xyz  | Run `pip install -r requirements.txt` |

---

## 🛠️ Extending the Project

* Add new file readers in `data_ingestor.py`
* Swap models in `llm_client.py`
* Customize outputs (descriptions, diagrams, SQL prompts)

---

## 📚 Documentation

* 📘 Full Guide → `mcp-documentation.pdf`
* 📊 Pitch Deck → `mcp-presentation.pdf`

---
