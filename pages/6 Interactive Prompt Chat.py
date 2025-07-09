# interactive_prompt_chat.py — memory-aware, schema-grounded, multi-turn SQL assistant

import streamlit as st
import os
import pandas as pd
import datetime

from app.core.config import FILES_ROOT
from app.natural_sql_query import run_sql
from app.supabase_utils import run_sql_on_supabase
from app.chromadb_prompt_utils import generate_sql_strict  # now memory-aware

# --------------------------------------------------------------------------------------
# 🔧 Log interaction
# --------------------------------------------------------------------------------------
def log_query(dataset_name: str, user_prompt: str, llm_reply: str, sql_query: str = ""):
    log_path = os.path.join(FILES_ROOT, dataset_name, "interactive_query_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}]\n")
        f.write(f"User Prompt: {user_prompt}\n")
        f.write(f"LLM Response:\n{llm_reply}\n")
        if sql_query:
            f.write(f"Final SQL:\n{sql_query}\n")
        f.write("=" * 80 + "\n")

# --------------------------------------------------------------------------------------
# 🧱 UI setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="💬 Talk to Your Data", layout="wide")
st.title("💬 Chat with Your Data (Interactive, Schema-Grounded SQL)")

# --------------------------------------------------------------------------------------
# 📁 Dataset selection
# --------------------------------------------------------------------------------------
st.subheader("1. Choose a Dataset")

available_datasets = sorted([
    d for d in os.listdir(FILES_ROOT)
    if os.path.isdir(os.path.join(FILES_ROOT, d)) and any(f.endswith(".db") for f in os.listdir(os.path.join(FILES_ROOT, d)))
])

dataset_name = st.selectbox("Select a dataset to work with:", available_datasets)
dataset_path = os.path.join(FILES_ROOT, dataset_name)
db_path = os.path.join(dataset_path, f"{dataset_name}.db")

# --------------------------------------------------------------------------------------
# 🧠 Multi-turn LLM Chat (Schema-Grounded)
# --------------------------------------------------------------------------------------
st.divider()
st.subheader("2. Ask a Question (LLM will use schema and memory)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate assistant reply
    with st.chat_message("assistant"):
        sql, full_reply = generate_sql_strict(dataset_name, st.session_state.chat_history[:-1], user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": full_reply})
        log_query(dataset_name, user_input, full_reply, sql)

        if sql:
            with st.expander("🧾 View Generated SQL"):
                st.code(sql, language="sql")

# --------------------------------------------------------------------------------------
# 🧪 Manual SQL Execution
# --------------------------------------------------------------------------------------
st.divider()
st.subheader("3. Run Final SQL Query")

sql_input = st.text_area("Paste or edit SQL before execution:", height=150)
use_supabase = st.toggle("🔗 Use Supabase instead of local DB", value=False)

if st.button("Run SQL Query"):
    if sql_input.strip().lower().startswith("select"):
        if use_supabase and "supabase_conn" in st.session_state:
            result_df = run_sql_on_supabase(sql_input.strip(), st.session_state["supabase_conn"])
        else:
            result_df = run_sql(sql_input.strip(), db_path)

        log_query(dataset_name, "(manual execution)", "", sql_input)

        if isinstance(result_df, pd.DataFrame):
            st.dataframe(result_df)
        else:
            st.error(result_df)
    else:
        st.warning("⚠️ Only SELECT queries are allowed.")
        st.warning("⚠️ Only SELECT queries are allowed.")