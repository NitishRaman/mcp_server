# schema_vectorizer.py
# 🔍 Vectorizes schema description using embeddings + annotates with PK/FK/tag metadata
import os
import json
import sqlite3
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.llm_client import LLMClient
from app.core.config import EMBEDDING_MODEL, model


# --- Function: embed_text ---
def embed_text(text):
    """
    Generate normalized embedding for a given input string using the configured embedding model.
    """
    return model.encode(text, normalize_embeddings=True).tolist()


# --- Function: vectorize_database ---
def vectorize_database(description_path, dataset_path, override_keys=None):
    """
    Main function to process schema description, embed table/column descriptions, and index them in ChromaDB.

    Args:
        description_path (str): Path to the schema description file (.txt)
        dataset_path (str): Path to the dataset folder
        override_keys (dict): Optional override of PK/FK structure (e.g., Supabase parsed output)
    """
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))  # ✅ Fix: extract dataset name
    with open(description_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Load fallback PK/FK structure if available
    inferred_path = description_path.replace("_description.txt", "_inferred_keys.json")
    if os.path.exists(inferred_path):
        with open(inferred_path, "r", encoding="utf-8") as f:
            inferred_keys = json.load(f)
    else:
        inferred_keys = {}


    embedding_func = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collection_name = f"schema_{dataset_name}"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    print(f"✅ Using ChromaDB collection name: {collection_name}")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)

    tag_outputs = {}  # Save final tags to JSON file

    table_blocks = content.strip().split("### Table ")

    # 🔍 Extract real FK/PK info using SQLite PRAGMA
    db_files = [f for f in os.listdir(dataset_path) if f.endswith(".db")]
    pk_fk_info = {}
    
    if override_keys:
        print("✅ Using override PK/FK keys (e.g. from Supabase)")
        pk_fk_info = {
            table: {
                "primary_keys": set(info.get("primary_keys", [])),
                "foreign_keys": info.get("foreign_keys", []),
                "source": info.get("source", "override")
            }
            for table, info in override_keys.items()
        }
        db_files = []  # skip PRAGMA if keys provided

    if db_files:
        db_path = os.path.join(dataset_path, db_files[0])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for block in table_blocks[1:]:
            lines = block.strip().splitlines()
            if not lines:
                continue
            header = lines[0]
            table_name = header.split(":", 1)[1].strip()

            pk_fk_info[table_name] = {
                "primary_keys": set(),
                "foreign_keys": [],
                "source": "pragma"  # default to PRAGMA
            }


            try:
                cursor.execute(f"PRAGMA table_info({table_name});")
                for col in cursor.fetchall():
                    if col[5] == 1:
                        pk_fk_info[table_name]["primary_keys"].add(col[1])

                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                for fk in cursor.fetchall():
                    pk_fk_info[table_name]["foreign_keys"].append(f"{fk[3]} → {fk[2]}.{fk[4]}")
            except:
                pass

            # 🔁 Fallback: try to infer if PRAGMA fails
            if not pk_fk_info[table_name]["primary_keys"]:
                inferred = inferred_keys.get(table_name, {})
                pk_fk_info[table_name]["primary_keys"] = set(inferred.get("primary_keys", []))
                pk_fk_info[table_name]["source"] = "inferred"

            if not pk_fk_info[table_name]["foreign_keys"]:
                inferred = inferred_keys.get(table_name, {})
                pk_fk_info[table_name]["foreign_keys"] = inferred.get("foreign_keys", [])
                pk_fk_info[table_name]["source"] = "inferred"

        conn.close()

    # --- Vectorize table and column descriptions ---
    for block in table_blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        header = lines[0]
        table_name = header.split(":", 1)[1].strip()

        summary = ""
        columns, natural_texts = [], []

        for line in lines[1:]:
            if line.startswith("**Summary:**"):
                summary = line.strip()
            elif line.strip().startswith("- **"):
                columns.append(line.strip())
            elif "↪" in line:
                natural_texts.append(line.split("↪", 1)[1].strip())

        # ✅ Table-level summary
        if summary:
            collection.add(
                documents=[summary],
                embeddings=[embed_text(summary)],
                metadatas=[{"table": table_name, "type": "table_summary"}],
                ids=[f"{table_name}_summary"]
            )

        # ✅ Column-level entries
        for i, col_line in enumerate(columns):
            col_line_lower = col_line.lower()
            is_pk = any(pk.lower() in col_line_lower for pk in pk_fk_info.get(table_name, {}).get("primary_keys", []))
            is_fk = any(fk.split(" → ")[0].lower() in col_line_lower for fk in pk_fk_info.get(table_name, {}).get("foreign_keys", []))

            source = pk_fk_info.get(table_name, {}).get("source", "unknown")
            metadata = {
                "table": table_name,
                "type": "column_description",
                "primary_key": is_pk,
                "foreign_key": is_fk,
                "source": source
            }

            col_doc = col_line  # No hallucination text
            collection.add(
                documents=[col_doc],
                embeddings=[embed_text(col_doc)],
                metadatas=[metadata],
                ids=[f"{table_name}.col_{i}"]
            )

        # ✅ LLM prompt for tags/relationships
        if natural_texts:
            pk_lines = "\n".join(f"- {pk}" for pk in pk_fk_info.get(table_name, {}).get("primary_keys", []))
            fk_lines = "\n".join(f"- {fk}" for fk in pk_fk_info.get(table_name, {}).get("foreign_keys", []))

            prompt = f"""
You are an expert database analyst.

Table: {table_name}

Column descriptions:
{chr(10).join(natural_texts)}

Primary Keys:
{pk_lines if pk_lines else '- None'}

Foreign Keys:
{fk_lines if fk_lines else '- None'}

Instructions:
1. List 3-5 keywords (tags) that describe this table.
2. Mention relevant relationships like 'CustomerID → Customer'.
3. Briefly summarize the business purpose of the table.
4. Avoid guessing or hallucinating table or column names not shown above.

ONLY include keywords, relationships, and business purpose. Do not add explanations or assistant-like phrases.
""".strip()

            try:
                llm = LLMClient()
                tag_result = llm.chat_completion([{"role": "user", "content": prompt}]).strip()
                tag_outputs[table_name] = tag_result

                collection.add(
                    documents=[tag_result],
                    embeddings=[embed_text(tag_result)],
                    metadatas=[{"table": table_name, "type": "table_tags"}],
                    ids=[f"{table_name}_tags"]
                )
            except Exception as e:
                print(f"⚠️ Tagging failed for {table_name}: {e}")

    # Save LLM-generated tags
    json_out = os.path.join(os.path.dirname(description_path), f"{os.path.basename(dataset_path)}_tags.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(tag_outputs, f, indent=2)

    print(f"✅ Vectorization complete. Tags saved to: {json_out}")
    
    # ✅ Save final metadata JSON with PK, FK, and source
    metadata_out = os.path.join(os.path.dirname(description_path), f"{dataset_name}_metadata.json")
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump({
            k: {
                "primary_keys": list(v["primary_keys"]) if isinstance(v["primary_keys"], set) else v["primary_keys"],
                "foreign_keys": v["foreign_keys"],
                "source": v["source"]
            } for k, v in pk_fk_info.items()
        }, f, indent=2)
        
    # ✅ Optionally append a summary block to description.txt for UI readability
    with open(description_path, "a", encoding="utf-8") as f:
        f.write("\n\n# 🔑 Final Inferred Keys (PRAGMA + Fallback)\n")
        for table, info in pk_fk_info.items():
            f.write(f"\n## {table}\n")
            f.write("Primary Keys: " + ", ".join(info.get("primary_keys", [])) + "\n\n")
            f.write("Foreign Keys:\n")
            for fk in info.get("foreign_keys", []):
                f.write(f"- {fk}\n")


    print(f"✅ Final metadata saved to: {metadata_out}")

