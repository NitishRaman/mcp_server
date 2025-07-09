import os
import re
import datetime
import sqlite3
import pandas as pd
import datetime
from app.core.config import FILES_ROOT
from app.llm_client import LLMClient
from app.chromadb_prompt_utils import get_all_documents_from_collection


# --- Function: generate_sql_from_prompt ---
def generate_sql_from_prompt(dataset_name: str, question: str):
    """
    Uses LLM + ChromaDB context to generate a SQL query based on user question.
    Then executes the SQL on the dataset's SQLite DB.

    Returns:
        tuple[str, pd.DataFrame] or str (error message)
    """
    try:
        dataset_path = os.path.join(FILES_ROOT, dataset_name)

        # ✅ Use full context from ChromaDB
        context_chunks = get_all_documents_from_collection(dataset_name)
        if not context_chunks:
            log_query_failure(dataset_path, question, reason="No context found in ChromaDB")
            return "❌ No relevant schema information found in vector DB."

        context_block = "\n---\n".join(context_chunks)
        prompt = f"""
You are an expert database analyst. Given the schema below and the user's question, generate a correct SQL query.

Use only table and column names that appear *exactly* in the schema context.

Schema Context:
{context_block}

User Question:
{question}

Return only the SQL code, nothing else.
"""


        # Query the LLM
        response = LLMClient().ask(prompt)
        if not response:
            log_query_failure(dataset_path, question, reason="LLM returned no output")
            return "❌ LLM returned no output."

        raw_output = response['text'] if isinstance(response, dict) and 'text' in response else response

        # ✅Clean up LLM output-> Remove markdown, comments, explanations
        # Step 1: Strip markdown code block if present
        cleaned = re.sub(r"```(?:sql)?(.*?)```", r"\1", raw_output, flags=re.DOTALL)

        # Step 2: Strip any lines before SELECT
        match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)\s", cleaned, flags=re.IGNORECASE)
        sql_code = cleaned[match.start():].strip() if match else cleaned.strip()

        # Final cleanup
        sql_code = sql_code.rstrip(';') + ';'

        # Log and run the query
        log_successful_query(dataset_path, question, prompt, sql_code)

        # ✅ Run the query
        db_files = [f for f in os.listdir(dataset_path) if f.endswith(".db")]
        if not db_files:
            return sql_code, pd.DataFrame()

        db_path = os.path.join(dataset_path, db_files[0])
        result_df = run_sql(sql_code, db_path)
        return sql_code, result_df

    except Exception as e:
        log_query_failure(dataset_path, question, reason=str(e))
        return f"❌ Error during SQL generation: {str(e)}"


# --- Function: run_sql ---
def run_sql(sql_query: str, db_path: str):
    """
    Execute a SQL query on a SQLite database.

    Returns:
        pd.DataFrame or str (error message)
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        return f"❌ SQL Execution Error: {str(e)}"


# --- Function: log_successful_query ---
def log_successful_query(dataset_path, user_question, full_prompt, llm_response):
    """
    Logs successful LLM responses and the generated SQL to query_log.txt.
    """
    log_path = os.path.join(dataset_path, "query_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] SUCCESS\n")
        f.write(f"User Question: {user_question}\n")
        f.write(f"\n--- Prompt Sent to LLM ---\n{full_prompt}\n")
        f.write(f"\n--- LLM Response ---\n{llm_response}\n")
        f.write("=" * 80 + "\n")


# --- Function: log_query_failure ---
def log_query_failure(dataset_path, user_question, reason):
    """
    Logs failed attempts to generate SQL or query errors.
    """
    log_path = os.path.join(dataset_path, "query_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] ERROR\n")
        f.write(f"User Question: {user_question}\n")
        f.write(f"Error: {reason}\n")
        f.write("=" * 80 + "\n")

