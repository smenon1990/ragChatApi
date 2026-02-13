import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.models import QueryRequest, QueryResponse
from app.rag import ingest_pdf, query_rag

app = FastAPI(title="RAG API")

# Pick ONE upload directory (cross-platform safe)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CORS
origins = [
    "http://localhost:5173",
    "https://rag-ui-production.up.railway.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RAG API running", "health": "/health", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest-pdf")
async def ingest_pdf_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(file_path)
    return {"message": "PDF ingested successfully"}

@app.post("/query", response_model=QueryResponse)
def query_api(request: QueryRequest):
    answer = query_rag(request.question)
    return QueryResponse(answer=answer)
