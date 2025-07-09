# config.py
"""
Global configuration & default constants for the MCP Server project.
"""

import os
from sentence_transformers import SentenceTransformer

# === 📁 Filesystem Paths ===

# Root directory where all dataset folders are stored
FILES_ROOT = os.path.join(os.getcwd(), "mcp_server", "files")

# === 🤖 LLM Model Selection ===

# Backend can be 'ollama' or 'gpt4all'
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# Model name used for the selected LLM backend
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")

# === 🔍 Embedding Model ===

# Model used for generating vector embeddings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
model = SentenceTransformer(EMBEDDING_MODEL)

# === 🔧 Global Constants and Configuration ========= #
# === 📊 UI Chart Types ===

# Supported chart types for visualization UI
CHART_TYPES = ['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot']

# === 🛠️ Data Parsing & Cleaning Defaults ===

# Prefixes of tables to ignore during ingestion
EXCLUDE_PREFIXES = ("sqlite_", "sys", "pg_", "flyway", "metadata")

# Max number of rows to sample from each table
MAX_SAMPLE_ROWS = 3

# Maximum length of column names to allow
MAX_COL_NAME_LEN = 60

# Limit for previewing string content from a column
MAX_STRING_PREVIEW = 150

# Average character length limit used to skip text-heavy fields
MAX_AVG_CHARLEN = 200

# Whether to skip BLOB or binary columns entirely
SKIP_BLOB_COLS = True

# Path to write log of skipped databases
SKIPPED_LOG = "skipped_dbs.txt"

# Optional: for Hugging Face API
HF_API_KEY = os.getenv("HF_API_KEY")
HF_ENDPOINT = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"