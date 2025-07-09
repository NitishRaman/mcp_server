# ✅ app_server.py — Modular, Clean FastAPI MCP Server

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os

from app.data_ingestor import save_csv
from app.schema_describer import describe_database
from app.natural_sql_query import generate_sql_from_prompt

# Initialize FastAPI app
app = FastAPI()

# ✅ Upload endpoint to accept dataset files and save them to dataset folder
@app.post("/upload")
async def upload(dataset_name: str = Form(...), file: UploadFile = Form(...)):
    content = await file.read()
    folder = f"mcp_server/files/{dataset_name}"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file.filename)
    with open(path, "wb") as f:
        f.write(content)
    return {"status": "uploaded", "file": path}

# ✅ Schema endpoint: generate and return schema description from .db file
@app.get("/schema/{dataset_name}")
def schema(dataset_name: str):
    db_path = f"mcp_server/files/{dataset_name}/{dataset_name}.db"
    describe_database(db_path, f"mcp_server/files/{dataset_name}")
    description_path = f"mcp_server/files/{dataset_name}/{dataset_name}_description.txt"
    if not os.path.exists(description_path):
        return JSONResponse(content={"error": "Schema description not found."}, status_code=404)

    with open(description_path) as f:
        return f.read()

# ✅ Natural language query endpoint
@app.post("/query")
def query(dataset_name: str = Form(...), question: str = Form(...)):
    folder = f"mcp_server/files/{dataset_name}"
    response = generate_sql_from_prompt(dataset_name, question)
    if isinstance(response, tuple) and len(response) == 2:
        sql, result = response
    else:
        return {"sql": response, "results": None}

    if hasattr(result, 'to_dict'):
        return {"sql": sql, "results": result.to_dict(orient="records")}
    else:
        return {"sql": sql, "results": str(result)}
