# schema_diagram.py
# 📊 ER Diagram Generation for SQLite or Supabase using ERAlchemy or Graphviz
import os
import subprocess
from graphviz import Digraph


# --- Function: generate_schema_diagram_with_eralchemy ---
def generate_schema_diagram_with_eralchemy(db_path: str, output_pdf: str):
    """
    Generate a schema diagram PDF using ERAlchemy from a SQLite database file.
    Args:
        db_path (str): Path to the SQLite .db file.
        output_pdf (str): Full output PDF path (e.g., /path/to/files/dataset/schema.pdf)
    """
    output_dir = os.path.dirname(output_pdf)
    os.makedirs(output_dir, exist_ok=True)

    er_file = os.path.join(output_dir, "schema.er")

    # Step 1: Generate intermediate .er file in dataset folder
    subprocess.run([
        "eralchemy",
        "-i", f"sqlite:///{db_path}",
        "-o", er_file
    ], check=True)

    # Step 2: Convert the .er file to a PDF diagram in same folder
    subprocess.run([
        "eralchemy",
        "-i", er_file,
        "-o", output_pdf
    ], check=True)

    print(f"📊 Schema diagram saved to {output_pdf}")


# --- Function: generate_supabase_diagram ---
def generate_supabase_diagram(tables_schema: dict, relationships: list, output_path: str, pk_fk_map: dict = None):
    """
    Generate an ER diagram (PNG) from Supabase-style schema and relationships.

    Args:
        tables_schema (dict): table → list of {column_name, data_type}
        relationships (list): source/target relationships as dicts
        output_path (str): File path to save PNG diagram
        pk_fk_map (dict): Optional PK/FK info for styling columns
    """

    dot = Digraph(comment="Supabase ER Diagram", format="png")
    dot.attr(rankdir='LR')  # Layout direction: Left to Right

    # Create nodes with PK/FK highlights
    for table, columns in tables_schema.items():
        pk_set = set()
        fk_set = set()

        if pk_fk_map and table in pk_fk_map:
            pk_set = set(pk_fk_map[table].get("primary_keys", []))
            fk_set = set(
                fk.split("→")[0].strip()
                for fk in pk_fk_map[table].get("foreign_keys", []) if "→" in fk
            )

        label = f"<<TABLE BORDER='1' CELLBORDER='0' CELLSPACING='0'>"
        label += f"<TR><TD BGCOLOR='lightblue'><B>{table}</B></TD></TR>"

        for col in columns:
            col_name = col["column_name"]
            col_type = col["data_type"]

            # Style PK/FK columns
            if col_name in pk_set:
                color = "#d4edda"  # light green
                prefix = "PK ➤ "
            elif col_name in fk_set:
                color = "#fff3cd"  # light yellow
                prefix = "FK → "
            else:
                color = "white"
                prefix = ""

            label += f"<TR><TD ALIGN='LEFT' BGCOLOR='{color}'>{prefix}{col_name} ({col_type})</TD></TR>"

        label += "</TABLE>>"
        dot.node(table, label=label, shape='plain')

    # Draw edges for foreign key relationships
    for rel in relationships:
        dot.edge(rel['source_table'], rel['target_table'], label=f"{rel['source_column']} → {rel['target_column']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, cleanup=True)
