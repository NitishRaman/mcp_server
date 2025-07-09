# chromadb_prompt_utils.py
# 🔍 Utility for retrieving documents from a ChromaDB collection and generating SQL prompts/results

import chromadb
import os
import re
import pandas as pd
from app.core.config import FILES_ROOT
from app.llm_client import LLMClient



# --- Function: get_all_documents_from_collection ---
def get_all_documents_from_collection(dataset_name):
    collection_name = f"schema_{dataset_name}"
    client = chromadb.HttpClient(host="localhost", port=8000)
    try:
        collection = client.get_collection(name=collection_name)
        results = collection.get(include=["documents"])
        return results.get("documents", [])
    except Exception as e:
        print(f"❌ Could not fetch Chroma collection: {e}")
        return []


# --- Function: build_conversational_sql_prompt ---
def build_conversational_sql_prompt(context_block: str, history: list, new_question: str) -> str:
    """
    Build prompt using persistent schema context and prior conversation.
    """
    system_prompt = f"""
You are a SQL assistant.
You have access to the following database schema:

{context_block}

⚠️ Use table and column names exactly as shown.
🎯 Ask clarifying questions if unclear.
✅ Return valid SQL only when confident.
Do NOT include explanations or markdown formatting.

Conversation:
"""
    chat_turns = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    chat_turns += f"\nUser: {new_question}"
    return system_prompt + chat_turns


# --- Function: generate_sql_strict ---
def generate_sql_strict(dataset_name: str, history: list, user_input: str) -> tuple[str, str]:
    context_chunks = get_all_documents_from_collection(dataset_name)
    if not context_chunks:
        return "❌ No vectorized schema available.", ""

    context_block = "\n---\n".join(context_chunks)
    full_prompt = build_conversational_sql_prompt(context_block, history, user_input)

    try:
        raw = LLMClient().ask(full_prompt)
        response = raw['text'] if isinstance(raw, dict) and 'text' in raw else raw
    except Exception as e:
        return f"❌ LLM error: {e}", ""

    # Extract clean SQL only (if present)
    match = re.search(r"```(?:sql)?(.*?)```", response, flags=re.DOTALL)
    if match:
        sql = match.group(1).strip()
    else:
        match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+", response, flags=re.IGNORECASE)
        sql = match.group(0).strip() if match else ""

    return sql.rstrip(';') + ';' if sql else "", response.strip()
