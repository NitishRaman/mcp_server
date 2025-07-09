# schema_describer.py
# 🧠 Analyze SQLite schema and generate LLM-powered summaries, stats, and key inference
import sqlite3
import pandas as pd
import os
import re
import sys
import subprocess
from tqdm import tqdm
from difflib import get_close_matches
from app.llm_client import LLMClient
from app.core.config import (
    FILES_ROOT, EXCLUDE_PREFIXES, MAX_SAMPLE_ROWS, MAX_COL_NAME_LEN,
    MAX_STRING_PREVIEW, MAX_AVG_CHARLEN, SKIP_BLOB_COLS, SKIPPED_LOG
)
from functools import lru_cache
import json
import chardet


# Load LLM model
llm = LLMClient()

# === Utility Functions === #


# --- Function: safe_read_sql_query ---
def safe_read_sql_query(query, conn, expected_columns):
    """
    Read a SQL query into a DataFrame, with safe decoding for mixed encodings.
    """
    try:
        # Force raw bytes instead of default utf-8 decode
        conn.text_factory = bytes
        df = pd.read_sql_query(query, conn)

        # Now decode each object/bytes column safely
        def safe_decode(x):
            if isinstance(x, (bytes, bytearray)):
                for enc in ('utf-8', 'iso-8859-1', 'latin1', 'utf-16'):
                    try:
                        return x.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return x.decode('utf-8', errors='replace')  # fallback
            return x

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(safe_decode)

        return df

    except Exception as e:
        print("\u26a0\ufe0f Safe SQL read failed:", e)
        return pd.DataFrame(columns=expected_columns)


# --- Function: should_skip_db ---
def should_skip_db(db_path):
    """
    Check if a database should be skipped based on size or table names.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            return True, "No tables found"

        system_like = all(any(t.lower().startswith(p) for p in EXCLUDE_PREFIXES) for t in tables)

        very_small = True
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                if count > 2:
                    very_small = False
                    break
            except:
                continue

        if system_like or very_small:
            reason = "System-like table names" if system_like else "All tables are very small"
            return True, reason
        return False, ""
    except Exception as e:
        return True, f"DB load error: {e}"


# --- Function: log_skipped_db ---
def log_skipped_db(db_path, reason):
    with open(SKIPPED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{os.path.basename(db_path)} — {reason}\n")


# --- Function: filter_tables ---
def filter_tables(conn):
    """
    Return list of user tables (exclude system-prefixed ones).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = [row[0] for row in cursor.fetchall()]
    return [t for t in all_tables if not any(t.lower().startswith(p) for p in EXCLUDE_PREFIXES)]


# --- Function: get_table_columns ---
@lru_cache(maxsize=128)
def get_table_columns(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return cursor.fetchall()
    

# --- Function: truncate_row ---
def truncate_row(row):
    """
    Truncate long strings or binary blobs for preview rows.
    """
    truncated = {}
    for k, v in row.items():
        if isinstance(v, str) and len(v) > MAX_STRING_PREVIEW:
            truncated[k] = v[:MAX_STRING_PREVIEW] + "...[truncated]"
        elif isinstance(v, bytes):
            truncated[k] = "[binary blob]"
        else:
            truncated[k] = v
    return truncated


# === Description Functions === #


# --- Function: compute_column_stats ---
def compute_column_stats(col_name, col_series):
    """
    Infer column type (numeric, date, boolean, etc.) and extract basic statistics.
    """
    stats = {}
    try:
        if pd.api.types.is_numeric_dtype(col_series):
            desc = col_series.describe()
            stats = {
                "type": "numeric",
                "count": int(desc.get("count", 0)),
                "mean": float(desc.get("mean", 0)),
                "median": float(col_series.median()),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
            }

            # 🎯 Detect "identifier" based on name and uniqueness
            name_lower = col_name.lower()
            if (
                "id" in name_lower
                and name_lower not in ("standardcost", "listprice")
                and col_series.nunique() >= 0.9 * len(col_series.dropna())
            ):
                stats["type"] = "identifier"

            # 🎯 Detect boolean (only 0/1 or True/False)
            unique_vals = sorted(col_series.dropna().unique().tolist())
            if unique_vals in ([0, 1], [1, 0], [True, False], [False, True]):
                stats["type"] = "boolean"

        elif pd.api.types.is_string_dtype(col_series):
            parsed_dates = pd.to_datetime(col_series, errors="coerce", infer_datetime_format=True)
            parse_ratio = parsed_dates.notna().sum() / len(col_series.dropna()) if len(col_series.dropna()) else 0

            if parse_ratio >= 0.95:
                stats = {
                    "type": "datetime",
                    "min_date": str(parsed_dates.min()),
                    "max_date": str(parsed_dates.max()),
                }
            else:
                top = col_series.value_counts(normalize=True).head(3)
                stats = {
                    "type": "categorical",
                    "unique": int(col_series.nunique()),
                    "top_values": [(val, round(pct * 100, 2)) for val, pct in top.items()],
                    "avg_len": col_series.dropna().map(len).mean()
                }

        elif pd.api.types.is_datetime64_any_dtype(col_series):
            stats = {
                "type": "datetime",
                "min_date": str(col_series.min()),
                "max_date": str(col_series.max()),
            }

        elif col_series.dtype == object and col_series.dropna().apply(lambda x: isinstance(x, bytes)).any():
            stats = {
                "type": "blob",
                "avg_bytes": int(col_series.dropna().apply(len).mean())
            }

        else:
            stats = {"type": "unknown"}

    except Exception as e:
        stats = {"type": "error", "reason": str(e)}

    return stats
    
    
# --- Function: format_column_description ---
def format_column_description(col_name, col_type, stats):
    """
    Convert column stats into a human-readable Markdown-style bullet.
    """
    type_label = stats.get("type", "unknown")
    if type_label == "numeric":
        return f"- **{col_name}** *(numeric)*: Ranges from {stats['min']} to {stats['max']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}, std={stats['std']:.2f}."
    elif type_label == "identifier":
        return f"- **{col_name}** *(identifier)*: Unique ID ranging from {stats['min']} to {stats['max']}."

    elif type_label == "boolean":
        return f"- **{col_name}** *(boolean)*: Binary flag with values 0/1."

    elif type_label == "categorical":
        tops = ", ".join([f"**{val}** ({pct}%)" for val, pct in stats["top_values"]])
        return f"- **{col_name}** *(categorical/text)*: {stats['unique']} unique values. Top values: {tops}."
    elif type_label == "datetime":
        return f"- **{col_name}** *(datetime)*: From {stats['min_date']} to {stats['max_date']}."
    elif type_label == "blob":
        return f"- **{col_name}** *(binary/blob)*: Avg size ≈ {stats['avg_bytes']} bytes."
    else:
        return f"- **{col_name}** *(unknown type)*"


# --- Function: describe_table ---
def describe_table(table, col_names, col_types, sample_data):
    """
    Generate a structured description for a single table, including:
    - Row count, column names/types
    - Sample preview rows
    - Column-level statistics
    """
    try:
        all_cols = ', '.join(f"{n} ({t})" for n, t in zip(col_names, col_types))
        trimmed_data = sample_data[:2]
        
        prompt = f"""
You are a database analyst. A table named `{table}` contains the following columns with types:
{all_cols}.

Here are 2 example rows:
{json.dumps(trimmed_data, indent=2)}

Avoid repeating column names. In 1-2 lines, summarize the purpose of this table in plain English for a business analyst.
"""

        print(f"\n🔍 Prompt for {table}:\n{prompt[:500]}...\n")  # 👈 log for debug

        response = llm.chat_completion([{'role': 'user', 'content': prompt}])

        if not response or not isinstance(response, str) or not response.strip():
            print(f"⚠️ Empty or invalid response for table: {table}")
            return "**Summary:** Failed to generate summary due to empty LLM response."

        cleaned = response.strip().replace("\n", " ").replace("  ", " ")
        return f"**Summary:** {cleaned}"

    except Exception as e:
        print(f"❌ Exception for table {table}: {e}")
        return "**Summary:** This table contains structured data relevant to business operations but exceeded the token context size for auto-summary."


# === Main Description Entry Point === #


# --- Function: describe_database ---
def describe_database(db_path, dataset_folder, tables_to_include=None):
    """
    Generate a natural language summary of a single table using an LLM.

    Args:
        table (str): Table name.
        columns (list): List of column dicts with 'column_name' and 'data_type'.
        sample_data (list): List of sample row dictionaries.

    Returns:
        str: Summary string for the table.
    """
    should_skip, reason = should_skip_db(db_path)
    if should_skip:
        print(f"⚠️ Skipping {os.path.basename(db_path)} — {reason}")
        log_skipped_db(db_path, reason)
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    all_tables = filter_tables(conn)
    tables = [t for t in all_tables if (tables_to_include is None or t in tables_to_include)]

    docs = []
    table_summaries = []   # <-- NEW LINE
    stats_json = {}
    print(f"📋 Found {len(tables)} user tables to describe...\n")

    for i, table in enumerate(tqdm(tables, desc="Describing Tables"), start=1):
        print(f"\n📄 [{i}/{len(tables)}] Table: {table}")
        cols_info = get_table_columns(cursor, table)
        col_names = [col[1].decode("utf-8", errors="replace") if isinstance(col[1], bytes) else col[1] for col in cols_info]
        col_types = [col[2].decode("utf-8", errors="replace") if isinstance(col[2], bytes) else col[2] for col in cols_info]

        df = safe_read_sql_query(f"SELECT * FROM {table} LIMIT 1000;", conn, expected_columns=col_names)
        df.columns = [col.decode("utf-8", errors="replace") if isinstance(col, bytes) else col for col in df.columns]



        # Drop columns with high avg char len or blob types before summary
        filtered_cols = []
        for name in col_names:
            try:
                series = df[name]
                if SKIP_BLOB_COLS and series.dropna().apply(lambda x: isinstance(x, bytes)).any():
                    continue
                if series.dropna().map(lambda x: len(str(x))).mean() > MAX_AVG_CHARLEN:
                    continue
                filtered_cols.append(name)
            except:
                filtered_cols.append(name)

        filtered_types = [col_types[col_names.index(n)] for n in filtered_cols]
        sample_data = [truncate_row(row) for row in df[filtered_cols].head(MAX_SAMPLE_ROWS).to_dict(orient="records")]

        table_summary = describe_table(table, filtered_cols, filtered_types, sample_data)
        summary_only = f"### Table {i}/{len(tables)}: {table}\n{table_summary}\n"
        table_summaries.append(summary_only)
        table_doc = f"\n\n### Table {i}/{len(tables)}: {table}\n{table_summary}\n\n## Columns:\n"


        col_stats = {}
        for idx, col in enumerate(col_names, start=1):
            print(f"      [{idx}/{len(col_names)}] Column: {col}")
            series = df[col] if col in df.columns else pd.Series([], dtype='object')
            col_type = col_types[idx - 1] if idx - 1 < len(col_types) else "UNKNOWN"
            stats = compute_column_stats(col, series)
            col_doc = format_column_description(col, col_type, stats)
            # 🔍 Natural column description using LLM
            sample_vals = df[col].dropna().astype(str).unique().tolist()[:5]
            natural_desc = describe_column_naturally(table, col, col_type, sample_vals)

            # Combine statistical + natural description
            col_doc += f"\n    ↪ {natural_desc}"
            #col_stats[col]["natural_description"] = natural_desc
            table_doc += col_doc + "\n"
            col_stats[col] = stats

        stats_json[table] = col_stats
        docs.append(table_doc)
        print(f"✅ Finished: {table}")

    # Save output
    os.makedirs(dataset_folder, exist_ok=True)
    out_base = os.path.join(dataset_folder, os.path.basename(dataset_folder))
    with open(out_base + "_description.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    with open(out_base + "_stats.json", "w", encoding="utf-8") as jf:
        json.dump(stats_json, jf, indent=2)
    with open(out_base + "_tablesummary.txt", "w", encoding="utf-8") as sf:
        sf.write("\n\n".join(table_summaries))


    print(f"\n✅ Schema description saved to: {out_base + '_description.txt'}")
    print(f"✅ Column stats saved to: {out_base + '_stats.json'}")


# --- Function: describe_column_naturally ---
def describe_column_naturally(table_name, column_name, col_type, sample_values):
    """
    Use LLM to generate a natural language description for a column.
    """
    prompt = f"""
You are a database assistant. The following is metadata for a column in a table:

Table: {table_name}
Column: {column_name}
Type: {col_type}
Sample values: {sample_values}

Describe what this column likely represents in 1-2 plain English sentences.
Avoid repeating the column name directly. Be specific.
"""

    try:
        response = llm.chat_completion([{"role": "user", "content": prompt}])
        return response.strip()
    except Exception as e:
        return f"⚠️ Description failed: {e}"


# --- Function: llm_infer_keys_from_description ---
def llm_infer_keys_from_description(description_path):
    with open(description_path, "r", encoding="utf-8") as f:
        description_txt = f.read()

    table_blocks = description_txt.strip().split("### Table ")[1:]
    pk_fk_info = {}

    for block in table_blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        header = lines[0]
        table_name = header.split(":", 1)[1].strip()
        body = "\n".join(lines[1:])

        prompt = f"""
Given the following table description, identify likely primary key(s) and foreign key(s).

### Table: {table_name}
{body}

Return a JSON with this format:
{{
  "primary_keys": ["..."],
  "foreign_keys": ["ColumnName => ReferencedTable"]
}}
""".strip()


        try:
            llm = LLMClient()
            response = llm.chat_completion([{"role": "user", "content": prompt}])
            if not response.strip():
                raise ValueError("LLM returned empty response.")
           

            # 🧹 Clean response: extract JSON block only
            json_match = re.search(r"\{[\s\S]+?\}", response)
            json_str = json_match.group(0) if json_match else response

            parsed = json.loads(json_str)
            pk_fk_info[table_name] = parsed

        except Exception as e:
            print(f"⚠️ Failed LLM key inference for {table_name}: {e}")
            pk_fk_info[table_name] = {"primary_keys": [], "foreign_keys": []}

    # ✅ Save inferred keys to JSON & Append inferred PK/FK info to description text end
    inferred_path = os.path.join(
        os.path.dirname(description_path),
        os.path.basename(description_path).replace("_description.txt", "_inferred_keys.json")
    )
    with open(inferred_path, "w", encoding="utf-8") as f:
        json.dump(pk_fk_info, f, indent=2)
    print(f" Inferred keys saved to {inferred_path}")
    return pk_fk_info
