"""
AI for Legal Assistance & Access
FastAPI Backend — main application entry point with RAG pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from services.document_service import DocumentService
from services.ai_service import AIService

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
HOST           = os.getenv("HOST", "0.0.0.0")
PORT           = int(os.getenv("PORT", 8000))
MAX_UPLOAD_MB  = int(os.getenv("MAX_UPLOAD_SIZE_MB", 20))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# ── Base Directory paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path("/tmp/uploads") if os.getenv("VERCEL") else (BASE_DIR / "uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    logger.warning("GEMINI_API_KEY is not set — AI features will be unavailable")

# ── Service layer ─────────────────────────────────────────────────────────────
doc_service = DocumentService(upload_dir=UPLOAD_DIR, max_size_mb=MAX_UPLOAD_MB)

try:
    ai_service = AIService(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
except RuntimeError as _ai_init_err:
    logger.warning("AI service not initialized: %s", _ai_init_err)
    ai_service = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="LexAI — Legal Assistance & Access",
    description="RAG-powered conversational legal document assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & UI ─────────────────────────────────────────────────────────
STATIC_DIR = BASE_DIR / "frontend" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

INDEX_HTML = BASE_DIR / "frontend" / "index.html"


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return JSONResponse({"status": "LexAI API active", "version": "2.0.0"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "ai_ready": ai_service is not None,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document (PDF, DOCX, TXT), extract text,
    chunk passages, compute embeddings, and build RAG index.
    """
    try:
        session_id, text, meta = await doc_service.upload_and_extract(file)

        chunks = []
        starter_questions = [
            "What are the main obligations in this contract?",
            "What are the termination conditions?",
            "Are there any liability or indemnification clauses?",
            "What happens in the event of a breach?",
        ]

        if ai_service is not None:
            # Semantic chunking
            chunks = ai_service.rag.chunk_document(text)
            # Embed chunks with Gemini Embeddings API
            chunks = await ai_service.rag.index_chunks(chunks)
            # Generate tailored starter questions
            starter_questions = await ai_service.generate_initial_starter_questions(
                chunks, file.filename
            )
        else:
            # Fallback simple chunking if AI service key is missing
            from services.rag_service import RAGService
            fallback_rag = RAGService(None)
            chunks = fallback_rag.chunk_document(text)

        doc_service.set_chunks(session_id, chunks, starter_questions)
        logger.info(
            "Uploaded %s → session %s (%d chars, %d chunks)",
            file.filename,
            session_id,
            len(text),
            len(chunks),
        )

        return {
            "session_id": session_id,
            "filename": meta["filename"],
            "pages": meta.get("pages"),
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
            "starter_questions": starter_questions,
            "preview": text[:400] + ("…" if len(text) > 400 else ""),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    RAG-grounded legal assistant chat.
    Retrieves most relevant document clauses and generates structured, visually appealing analysis.
    """
    if ai_service is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Set GEMINI_API_KEY in .env and restart.",
        )

    session = doc_service.get_session(req.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please upload your document again.",
        )

    chunks = session.get("chunks", [])
    if not chunks:
        # Re-chunk if missing
        chunks = ai_service.rag.chunk_document(session["text"])
        chunks = await ai_service.rag.index_chunks(chunks)
        doc_service.set_chunks(req.session_id, chunks)

    try:
        # 1. Retrieve top-4 most relevant clauses
        retrieved_chunks = await ai_service.rag.retrieve(req.question, chunks, top_k=4)

        # 2. Generate grounded, visually structured answer
        result = await ai_service.answer_rag(
            question=req.question,
            retrieved_chunks=retrieved_chunks,
            document_filename=session["meta"].get("filename", "Document"),
        )

        return {
            "session_id": req.session_id,
            "answer": result["answer"],
            "citations": result["citations"],
            "follow_ups": result["follow_ups"],
        }
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up uploaded file and session data."""
    doc_service.delete_session(session_id)
    return {"detail": "Session deleted"}


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
