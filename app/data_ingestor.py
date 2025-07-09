import os
import io
import zipfile
import json
import pandas as pd
import xml.etree.ElementTree as ET
import csv
from app.core.config import FILES_ROOT
from pathlib import Path
import sqlite3  # For reading .db SQLite files



# --- Function: ensure_dataset_folder ---
def ensure_dataset_folder(dataset_name: str) -> str:
    """
    Ensure a folder exists for the given dataset name under FILES_ROOT.
    Returns the full path of the folder.
    """
    folder = os.path.join(FILES_ROOT, dataset_name)
    os.makedirs(folder, exist_ok=True)
    return folder


# --- Function: save_csv ---
def save_csv(df: pd.DataFrame, dataset_name: str, filename: str):
    """
    Save a DataFrame as CSV inside the dataset folder.
    """
    folder = ensure_dataset_folder(dataset_name)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)


# --- Function: detect_delimiter_from_bytes ---
def detect_delimiter_from_bytes(content: bytes) -> str:
    """
    Use csv.Sniffer to guess the delimiter of CSV content.
    Defaults to comma if detection fails.
    """
    try:
        sample = content.decode('utf-8', errors='replace')
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ','


# --- Function: read_csv_from_bytes ---
def read_csv_from_bytes(content: bytes) -> pd.DataFrame:
    """
    Read CSV data from bytes with multiple encodings and delimiters trials,
    fallback to ensure best parsing.
    """
    sample = content[:2048]
    encodings = ['utf-8-sig', 'utf-8', 'latin1']
    delimiters = [',', ';', '|', '\t']
    try:
        # Try to detect delimiter from sample
        try:
            guessed = detect_delimiter_from_bytes(sample)
            print(f"🔍 Detected delimiter: '{guessed}'")
        except Exception:
            guessed = ','

        # Try parsing with guessed delimiter and different encodings
        for enc in encodings:
            try:
                df = pd.read_csv(
                    io.BytesIO(content), 
                    delimiter=guessed, 
                    encoding=enc, 
                    engine='python', 
                    on_bad_lines='skip'
                )
                if df.shape[1] > 2:
                    print(f"✅ Parsed CSV with shape: {df.shape}")
                    return df
            except Exception:
                continue

        # Fallback: try different delimiters and encodings
        for delim in delimiters:
            for enc in encodings:
                try:
                    df = pd.read_csv(
                        io.BytesIO(content), 
                        delimiter=delim, 
                        encoding=enc, 
                        engine='python', 
                        on_bad_lines='skip'
                    )
                    if df.shape[1] > 2:
                        print(f"✅ Fallback worked with delimiter '{delim}' and encoding '{enc}' → shape: {df.shape}")
                        return df
                except Exception:
                    continue

        # If parsing was unsuccessful, attempt a basic parse with the guessed delimiter
        print("⚠️ Could not fully parse CSV; returning basic result")
        df = pd.read_csv(
            io.BytesIO(content), 
            delimiter=guessed, 
            encoding='utf-8', 
            engine='python', 
            on_bad_lines='skip'
        )
        print(f"⚠️ Parsed fallback CSV with shape: {df.shape}")
        return df

    except Exception as e:
        raise ValueError(f"❌ Error reading CSV: {e}")


# --- Function: read_excel ---
def read_excel(path_or_bytes) -> dict:
    """
    Read an Excel file from a path or bytes and return a dict of {sheet_name: DataFrame}.
    """
    return pd.read_excel(path_or_bytes, sheet_name=None)


# --- Function: read_zip ---
def read_zip(path: str, dataset: str) -> dict:
    """
    Extract a ZIP file, read contained CSV and Excel files, save extracted CSVs,
    and return a dict of {filename or filename::sheetname: DataFrame}.
    """
    extracted = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            try:
                with z.open(name) as f:
                    content = f.read()
                    if name.endswith('.csv'):
                        try:
                            df = read_csv_from_bytes(content)
                            if df.shape[1] == 1:
                                print(f"⚠️ Warning: {name} parsed into 1 column — possible delimiter issue.")
                            extracted[name] = df
                            save_csv(df, dataset, Path(name).stem + '.csv')
                        except Exception as e:
                            print(f"❌ Failed to read CSV inside ZIP: {name} → {e}")
                            extracted[name] = f"❌ Error reading CSV: {e}"
                    elif name.endswith('.xlsx'):
                        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
                        extracted[name] = sheets
                        for sheet_name, df in sheets.items():
                            clean_name = f"{Path(name).stem}_{sheet_name}.csv"
                            save_csv(df, dataset, clean_name)
            except Exception as e:
                extracted[name] = f"❌ Failed to read {name}: {e}"
    return extracted


# --- Function: read_json ---
def read_json(path: str) -> pd.DataFrame:
    """
    Read a JSON file and normalize into a DataFrame.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return pd.json_normalize(data)


# --- Function: read_xml ---
def read_xml(path: str) -> pd.DataFrame:
    """
    Read an XML file and parse it into a DataFrame assuming a flat structure.
    """
    tree = ET.parse(path)
    rows = [{child.tag: child.text for child in elem} for elem in tree.getroot()]
    return pd.DataFrame(rows)
    

# --- Function: clean_value ---
def clean_value(value):
    """
    Attempt to decode a cell value safely into UTF-8.
    Handles both str and bytes, falling back to Latin1 where needed.
    """
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, str):
        return value.encode('latin1', errors='replace').decode('utf-8', errors='replace')
    return value


# --- Function: read_db ---
def read_db(path: str, dataset_name: str) -> dict:
    """
    Read SQLite .db and return dict of {table_name: DataFrame} with all text cleaned.
    Each table is saved as UTF-8 CSV in the dataset folder.
    """
    conn = sqlite3.connect(path)
    conn.text_factory = lambda x: x.decode("utf-8", "replace")
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    data = {}

    # Read and clean each table
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM `{table}`")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            # Clean each cell in every row
            cleaned_rows = [[clean_value(cell) for cell in row] for row in rows]
            df = pd.DataFrame(cleaned_rows, columns=columns)

            save_csv(df, dataset_name, f"{table}.csv")
            data[table] = df
            print(f"✅ Loaded and saved table '{table}' with shape {df.shape}")

        except Exception as e:
            print(f"❌ Failed to load table {table}: {e}")

    conn.close()
    return data


# --- Function: describe_table ---
def describe_table(df: pd.DataFrame) -> str:
    """
    Generate a textual summary of a DataFrame: shape, columns, datatypes, and null counts.
    """
    summary = f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
    nulls = df.isnull().sum()
    summary += "🧾 Columns:\n"
    for col in df.columns:
        dtype = df[col].dtype
        summary += f"   - {col}: {dtype}, nulls={nulls[col]}\n"
    return summary


# --- Function: save_and_ingest_file ---
def save_and_ingest_file(file_path: str, dataset_name: str) -> str:
    """
    Given a file path and dataset name, auto-detect file type, parse it,
    save CSVs, and convert into a SQLite database.
    Returns path to generated DB file.
    """
    ext = Path(file_path).suffix.lower()
    dataset_folder = ensure_dataset_folder(dataset_name)
    db_path = os.path.join(dataset_folder, f"{dataset_name}.db")
    conn = sqlite3.connect(db_path)

    if ext == ".csv":
        df = pd.read_csv(file_path)
        save_csv(df, dataset_name, f"{dataset_name}_data.csv")
        df.to_sql("data", conn, if_exists="replace", index=False)

    elif ext in [".xlsx", ".xls"]:
        sheets = read_excel(file_path)
        for sheet_name, df in sheets.items():
            sheet_clean = sheet_name[:30]
            save_csv(df, dataset_name, f"{sheet_clean}.csv")
            df.to_sql(sheet_clean, conn, if_exists="replace", index=False)

    elif ext == ".zip":
        extracted = read_zip(file_path, dataset_name)
        for name, df in extracted.items():
            if isinstance(df, pd.DataFrame):
                clean_name = Path(name).stem[:30]
                save_csv(df, dataset_name, f"{clean_name}.csv")
                df.to_sql(clean_name, conn, if_exists="replace", index=False)
            elif isinstance(df, dict):
                for sheet, subdf in df.items():
                    tbl = f"{Path(name).stem}_{sheet}"[:30]
                    save_csv(subdf, dataset_name, f"{tbl}.csv")
                    subdf.to_sql(tbl, conn, if_exists="replace", index=False)

    elif ext == ".json":
        df = read_json(file_path)
        save_csv(df, dataset_name, f"{dataset_name}_data.csv")
        df.to_sql("data", conn, if_exists="replace", index=False)

    elif ext == ".xml":
        df = read_xml(file_path)
        save_csv(df, dataset_name, f"{dataset_name}_data.csv")
        df.to_sql("data", conn, if_exists="replace", index=False)

    elif ext == ".db":
        if os.path.abspath(file_path) != os.path.abspath(db_path):
            with open(file_path, "rb") as src, open(db_path, "wb") as dst:
                dst.write(src.read())
        read_db(db_path, dataset_name)  # extract tables to CSV

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    conn.close()
    return db_path



