import io
import os
import subprocess
import time
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

try:  # pragma: no cover - optional heavy dependencies
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised via tests
    SentenceTransformer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependencies
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover - exercised via tests
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependencies
    from pdfminer.high_level import extract_text_to_fp as _extract_text_to_fp
except ModuleNotFoundError:  # pragma: no cover - exercised via tests
    def _extract_text_to_fp(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("pdfminer.six is required for PDF processing")

try:  # pragma: no cover - optional heavy dependencies
    from docx import Document as _Document
except ModuleNotFoundError:  # pragma: no cover - exercised via tests
    _Document = None  # type: ignore[assignment]

# config via env
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/chroma_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # change to BGE when available
OCR_LANG = os.environ.get("OCR_LANG", "eng")

# ensure persistence directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# init embedding model and chroma client
if SentenceTransformer is not None:  # pragma: no branch - simple guard
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    except Exception:  # pragma: no cover - exercised in tests when dependency unavailable
        embedder = None
else:  # pragma: no cover - exercised when dependency missing
    embedder = None

if chromadb is not None and Settings is not None:  # pragma: no branch - simple guard
    try:
        chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR)
        )
    except Exception:  # pragma: no cover - exercised in tests when dependency unavailable
        chroma_client = None
else:  # pragma: no cover - exercised when dependency missing
    chroma_client = None

router = APIRouter()

# helpers -------------------------------------------------
def save_upload_to_disk(session_id: str, upload: UploadFile) -> str:
    session_dir = os.path.join("data", session_id)
    os.makedirs(session_dir, exist_ok=True)
    filename = upload.filename or f"{uuid.uuid4().hex}"
    out_path = os.path.join(session_dir, filename)
    with open(out_path, "wb") as f:
        f.write(upload.file.read())
    return out_path

def extract_text_from_pdf_bytes(pdf_path: str) -> str:
    # try direct extract first
    out = io.StringIO()
    try:
        with open(pdf_path, "rb") as f:
            _extract_text_to_fp(f, out)
        text = out.getvalue()
        if text and len(text.strip()) > 20:
            return text
    except Exception:
        pass
    # fallback: run ocrmypdf -> extract again
    ocred = pdf_path.replace(".pdf", ".ocr.pdf")
    try:
        subprocess.run(["ocrmypdf", "-l", OCR_LANG, "--skip-text", pdf_path, ocred], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = io.StringIO()
        with open(ocred, "rb") as f:
            _extract_text_to_fp(f, out)
        text = out.getvalue()
        # optionally remove ocred file to save space
        try:
            os.remove(ocred)
        except Exception:
            pass
        return text
    except subprocess.CalledProcessError:
        return ""

def extract_text_from_docx(path: str) -> str:
    if _Document is None:
        return ""
    try:
        doc = _Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception:
        return ""

def chunk_text(text: str, chunk_size:int=2000, overlap:int=400):
    if not text:
        return []
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = i + chunk_size
        chunks.append(text[i:end])
        i = end - overlap
    return chunks

def get_or_create_collection(session_id: str):
    name = f"session_{session_id}"
    if chroma_client is None:  # pragma: no cover - validated via tests
        raise RuntimeError("Chroma client is not configured")
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return chroma_client.create_collection(name=name)

# API ------------------------------------------------------
class IngestResponse(BaseModel):
    status: str
    added_chunks: int
    saved_files: List[str]
    duration_seconds: float

@router.post("/sessions/{session_id}/ingest", response_model=IngestResponse)
async def ingest(session_id: str, files: List[UploadFile] = File(...)):
    start = time.time()
    saved = []
    total_chunks = 0
    if embedder is None or chroma_client is None:  # pragma: no cover - validated via tests
        raise HTTPException(status_code=503, detail="Vector store dependencies are unavailable")

    collection = get_or_create_collection(session_id)

    for upload in files:
        # ensure rewind file
        upload.file.seek(0)
        saved_path = save_upload_to_disk(session_id, upload)
        saved.append(saved_path)

        text = ""
        low_name = saved_path.lower()
        if low_name.endswith(".pdf"):
            text = extract_text_from_pdf_bytes(saved_path)
        elif low_name.endswith(".docx"):
            text = extract_text_from_docx(saved_path)
        else:
            # try decode as text
            try:
                with open(saved_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                try:
                    with open(saved_path, "r", encoding="latin-1") as f:
                        text = f.read()
                except Exception:
                    text = ""

        # chunk and embed
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": os.path.basename(saved_path), "session_id": session_id, "chunk_index": idx} for idx in range(len(chunks))]

        # add to chroma collection
        collection.add(documents=chunks, embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings, metadatas=metadatas, ids=ids)
        total_chunks += len(chunks)

    chroma_client.persist()
    duration = time.time() - start
    return {"status": "ok", "added_chunks": total_chunks, "saved_files": saved, "duration_seconds": duration}
