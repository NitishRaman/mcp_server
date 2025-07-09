import streamlit as st
import pandas as pd
import base64
from app.natural_sql_query import generate_sql_from_prompt
from app.core.config import FILES_ROOT
import os

st.title("🧠 Prompt to SQL Generator")

# ✅ 1. Choose dataset (based on available folders)
dataset_options = [d for d in os.listdir(FILES_ROOT) if os.path.isdir(os.path.join(FILES_ROOT, d))]
dataset_name = st.selectbox("Choose dataset:", dataset_options if dataset_options else ["No datasets found"])
dataset_path = os.path.join(FILES_ROOT, dataset_name)
pdf_path = os.path.join(dataset_path, f"{dataset_name}_schema.pdf")
if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# ✅ 2. Enter natural language question
question = st.text_area("Ask your question about the data", placeholder="e.g. Show me all customers from Canada")

# ✅ 3. Debug toggle
debug = st.checkbox("🔍 Debug: Show generated prompt and LLM response")

if st.button("Generate SQL") and dataset_name and question:
    with st.spinner("Thinking like a database analyst..."):
        result = generate_sql_from_prompt(dataset_name, question)

        if isinstance(result, str):
            st.error(result)
        elif result is None or not isinstance(result, tuple):
            st.error("❌ SQL generation failed: no response received.")
        else:
            sql_code, result_df = result

            if isinstance(sql_code, str) and sql_code.strip():
                st.code(sql_code, language="sql")
                if isinstance(result_df, pd.DataFrame):
                    st.dataframe(result_df)
                else:
                    st.error(result_df)  # this is likely an error string from run_sql
            else:
                st.error("❌ SQL code was empty or invalid.")

        # ✅ Show raw prompt+response if debug is on
        if debug:
            log_path = os.path.join(FILES_ROOT, dataset_name, "query_log.txt")
            st.markdown(f"📝 Expected log path: `{log_path}`")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    logs = f.read().strip().split("=" * 80)
                    if logs:
                        last_log = logs[-2] if len(logs) >= 2 else logs[-1]
                        st.markdown("### 📝 Last Prompt & Response")
                        st.code(last_log.strip())
            else:
                st.warning("No log file found yet.")

