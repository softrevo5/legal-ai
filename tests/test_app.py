"""
Tests for document service, RAG service, and AI services.
Run with: pytest tests/ -v
"""

import sys
import os
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.document_service import DocumentService
from services.rag_service import RAGService, cosine_similarity, keyword_overlap_score


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_upload_dir(tmp_path):
    return tmp_path / "uploads"


@pytest.fixture
def doc_service(tmp_upload_dir):
    tmp_upload_dir.mkdir()
    return DocumentService(upload_dir=tmp_upload_dir, max_size_mb=5)


def make_upload_file(content: bytes, filename: str):
    """Create a minimal FastAPI UploadFile-like mock."""
    mock = MagicMock()
    mock.filename = filename
    mock.read = AsyncMock(return_value=content)
    return mock


# ── DocumentService tests ──────────────────────────────────────────────────────

class TestDocumentService:

    @pytest.mark.asyncio
    async def test_upload_txt_success(self, doc_service):
        txt = b"This is a legal agreement between Party A and Party B."
        file = make_upload_file(txt, "contract.txt")
        session_id, text, meta = await doc_service.upload_and_extract(file)
        assert session_id
        assert "legal agreement" in text
        assert meta["filename"] == "contract.txt"
        assert doc_service.get_text(session_id) == text

    @pytest.mark.asyncio
    async def test_upload_unsupported_type_raises(self, doc_service):
        file = make_upload_file(b"data", "document.xls")
        with pytest.raises(ValueError, match="Unsupported file type"):
            await doc_service.upload_and_extract(file)

    @pytest.mark.asyncio
    async def test_upload_exceeds_size_raises(self, doc_service):
        big = b"x" * (6 * 1024 * 1024)  # 6 MB > 5 MB limit
        file = make_upload_file(big, "big.txt")
        with pytest.raises(ValueError, match="exceeds maximum size"):
            await doc_service.upload_and_extract(file)

    @pytest.mark.asyncio
    async def test_get_text_unknown_session_returns_none(self, doc_service):
        assert doc_service.get_text("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_delete_session_removes_text(self, doc_service):
        txt = b"Some contract text."
        file = make_upload_file(txt, "contract.txt")
        session_id, _, _ = await doc_service.upload_and_extract(file)
        doc_service.delete_session(session_id)
        assert doc_service.get_text(session_id) is None

    @pytest.mark.asyncio
    async def test_empty_txt_raises(self, doc_service):
        file = make_upload_file(b"   \n  ", "empty.txt")
        with pytest.raises(ValueError, match="empty"):
            await doc_service.upload_and_extract(file)


# ── RAG Service tests ──────────────────────────────────────────────────────────

class TestRAGService:

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0

        v3 = [0.0, 1.0]
        assert pytest.approx(cosine_similarity(v1, v3), 0.001) == 0.0

    def test_keyword_overlap_score(self):
        query = "termination notice period"
        text = "This contract specifies a 30-day termination notice period in Section 4."
        score = keyword_overlap_score(query, text)
        assert score == 1.0

    def test_chunk_document(self):
        rag = RAGService(client=None)
        doc = """Section 1. Definitions
The terms used in this Agreement shall have the following meanings.

Section 2. Scope of Services
The Contractor agrees to perform software development services.

Section 3. Termination
Either party may terminate this agreement with 30 days written notice."""

        chunks = rag.chunk_document(doc, target_size=120)
        assert len(chunks) >= 2
        assert all("heading" in c and "snippet" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_retrieve_lexical_fallback(self):
        rag = RAGService(client=None)
        chunks = [
            {"id": 1, "heading": "Clause 1", "text": "Payment shall be made in USD within 15 days.", "snippet": "Payment...", "vector": None},
            {"id": 2, "heading": "Clause 2", "text": "Confidentiality terms remain active for 2 years.", "snippet": "Confidentiality...", "vector": None},
        ]
        top = await rag.retrieve("When is payment due?", chunks, top_k=1)
        assert len(top) == 1
        assert top[0]["id"] == 1


# ── AI Service tests ───────────────────────────────────────────────────────────

class TestAIService:

    def _make_service(self):
        """Build AIService with a mocked Gemini client."""
        from services.ai_service import AIService
        service = AIService.__new__(AIService)
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(
            return_value=MagicMock(
                text="### 📌 Key Takeaway\nPayment is due in 30 days.\n\n```json\n{\"follow_ups\": [\"What about penalties?\"]}\n```"
            )
        )
        mock_aio = MagicMock()
        mock_aio.models = mock_models
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        service._client = mock_client
        service._model_name = "gemini-3.5-flash-lite"
        service.rag = RAGService(mock_client)
        return service

    @pytest.mark.asyncio
    async def test_answer_rag_parses_json_and_followups(self):
        service = self._make_service()
        chunks = [
            {"id": 1, "heading": "Payment", "text": "Due in 30 days.", "snippet": "Due...", "score": 0.9}
        ]
        res = await service.answer_rag("When to pay?", chunks, "sample.pdf")
        assert "Key Takeaway" in res["answer"]
        assert len(res["follow_ups"]) >= 1
        assert len(res["citations"]) == 1

    @pytest.mark.asyncio
    async def test_answer_rag_empty_question_raises(self):
        service = self._make_service()
        with pytest.raises(ValueError, match="empty"):
            await service.answer_rag("   ", [], "sample.pdf")


# ── API integration tests ─────────────────────────────────────────────────────

class TestAPIRoutes:

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3.5-flash-lite"}):
            from main import app
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "model" in data

    def test_upload_no_file_returns_422(self):
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            from main import app
            client = TestClient(app)
            response = client.post("/api/upload")
            assert response.status_code == 422

    def test_upload_unsupported_type_returns_400(self):
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            from main import app
            client = TestClient(app)
            response = client.post(
                "/api/upload",
                files={"file": ("doc.xls", b"fake xls content", "application/vnd.ms-excel")},
            )
            assert response.status_code == 400
            assert "Unsupported" in response.json()["detail"]

    def test_chat_unknown_session_returns_404(self):
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            from main import app
            client = TestClient(app)
            response = client.post(
                "/api/chat",
                json={"session_id": "ghost", "question": "What is this?"},
            )
            assert response.status_code == 404
