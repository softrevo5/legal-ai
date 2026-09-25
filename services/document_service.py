"""
Document Service — handles file upload, text extraction, session indexing and storage.
Supports PDF, DOCX, and TXT files.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from fastapi import UploadFile

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentService:
    def __init__(self, upload_dir: Optional[Path] = None, max_size_mb: int = 20):
        # In serverless environments like Vercel, use /tmp
        if upload_dir is None:
            if os.getenv("VERCEL"):
                self.upload_dir = Path("/tmp/uploads")
            else:
                self.upload_dir = Path("uploads")
        else:
            self.upload_dir = upload_dir

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        # In-memory session store: session_id → {text, meta, chunks, path}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def upload_and_extract(
        self, file: UploadFile
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Validate, save, extract text and return (session_id, text, meta)."""
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        raw = await file.read()
        if len(raw) > self.max_size_bytes:
            raise ValueError(
                f"File exceeds maximum size of {self.max_size_bytes // (1024 * 1024)} MB"
            )

        session_id = str(uuid.uuid4())
        save_path = self.upload_dir / f"{session_id}{suffix}"
        save_path.write_bytes(raw)

        text, meta = self._extract_text(save_path, suffix)
        meta["filename"] = file.filename

        self._sessions[session_id] = {
            "text": text,
            "meta": meta,
            "path": str(save_path),
            "chunks": [],
            "starter_questions": [],
        }
        logger.info("Session %s created for %s", session_id, file.filename)
        return session_id, text, meta

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def set_chunks(self, session_id: str, chunks: List[Dict[str, Any]], starter_questions: Optional[List[str]] = None):
        if session_id in self._sessions:
            self._sessions[session_id]["chunks"] = chunks
            if starter_questions:
                self._sessions[session_id]["starter_questions"] = starter_questions

    def get_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        return session.get("chunks", []) if session else []

    def get_text(self, session_id: str) -> Optional[str]:
        session = self._sessions.get(session_id)
        return session["text"] if session else None

    def delete_session(self, session_id: str):
        session = self._sessions.pop(session_id, None)
        if session:
            try:
                Path(session["path"]).unlink(missing_ok=True)
            except Exception:
                pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, path: Path, suffix: str) -> Tuple[str, Dict[str, Any]]:
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".docx":
            return self._extract_docx(path)
        else:
            return self._extract_txt(path)

    def _extract_pdf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        full_text = "\n\n".join(pages)
        if not full_text.strip():
            raise ValueError(
                "Could not extract text from this PDF. It may be scanned or image-only."
            )
        return full_text, {"pages": len(reader.pages)}

    def _extract_docx(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        from docx import Document
        doc = Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paras)
        if not full_text.strip():
            raise ValueError("Could not extract text from this DOCX file.")
        return full_text, {"paragraphs": len(paras)}

    def _extract_txt(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        raw = path.read_bytes()
        # Try UTF-8 first, fallback to latin-1
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                if not text.strip():
                    raise ValueError("The uploaded text file is empty.")
                return text, {"encoding": enc}
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file with supported encodings.")
