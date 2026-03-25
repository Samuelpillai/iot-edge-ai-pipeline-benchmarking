from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
import os

app = FastAPI(
    title="Vector Upload API",
    description="Upload vector-label pairs from IoT pipelines to the server",
    version="1.0.0"
)

VECTOR_LOG_PATH = "/Users/sam/CSC8199/OpenCV/RPiPipeline/vector_inference.jsonl"

class VectorEntry(BaseModel):
    vector: List[float]
    label: str

@app.post("/upload", tags=["Upload Vector"])
def upload_vector(entry: VectorEntry):
    """Upload a single vector and label"""
    if len(entry.vector) != 1280:
        raise HTTPException(status_code=400, detail="Vector must have 1280 features")

    try:
        with open(VECTOR_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "vector": entry.vector,
                "label": entry.label
            }) + "\n")
        return JSONResponse(content={"message": " Vector saved successfully"}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview", tags=["Utility"])
def preview_latest(n: int = 5):
    """Preview last N uploaded vectors"""
    if not os.path.exists(VECTOR_LOG_PATH):
        raise HTTPException(status_code=404, detail="vector_inference.jsonl not found")

    try:
        with open(VECTOR_LOG_PATH, "r") as f:
            lines = f.readlines()[-n:]
        entries = [json.loads(line) for line in lines]
        return {"preview": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this at the bottom to prevent 404 on /
@app.get("/", tags=["Root"])
def read_root():
    return {"message": " Vector Upload API is running. Visit /docs for Swagger UI."}