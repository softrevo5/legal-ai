"""
AI Service — wraps Google GenAI for RAG-grounded legal document analysis and conversational Q&A.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types as genai_types

from services.rag_service import RAGService

logger = logging.getLogger(__name__)


_RAG_ANSWER_SYSTEM_PROMPT = """You are LexAI, an elite AI legal assistant designed to help people understand complex legal documents and contracts in plain, transparent English.

YOUR MISSION:
Answer the user's question accurately using ONLY the provided retrieved clauses from their legal document.
Make your response visually appealing, beautifully structured, and immediately actionable.

FORMATTING REQUIREMENTS:
Structure your response using these exact visual sections:

### 📌 Key Takeaway
A clear, 1-2 sentence executive summary answering the question directly in simple English.

### ⚖️ Legal Analysis & Clauses
Detailed explanation of what the contract provisions say. Use bullet points, bold key terms, and explain legal jargon simply. Cite the relevant clause numbers (e.g. `[Clause 2]`).

### ⚠️ Risks & Considerations
Any potential traps, obligations, liabilities, deadlines, or penalties the reader should be aware of regarding this topic.

### 💡 Practical Advice & Next Steps
Concrete advice on what to verify, ask, or negotiate. (Always maintain informational posture: note that you are an AI assistant and not their formal legal counsel).

SUGGESTED FOLLOW-UPS:
At the very end of your response, output exactly 3 smart follow-up questions that the user might want to ask next, enclosed in a ```json codeblock:
```json
{
  "follow_ups": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}
```
"""


class AIService:
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        if not api_key or api_key == "your_gemini_api_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Please set it in .env and restart."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self.rag = RAGService(self._client)
        logger.info("AI Service initialised with model: %s", model)

    async def answer_rag(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        document_filename: str = "document",
    ) -> Dict[str, Any]:
        """
        Answer a user question grounded in retrieved document chunks.
        Returns:
          {
            "answer": str (clean markdown without raw json block),
            "follow_ups": List[str],
            "citations": List[Dict]
          }
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Build context block from retrieved chunks
        context_parts = []
        citations = []
        for c in retrieved_chunks:
            chunk_ref = f"[Clause #{c['id']}: {c['heading']}]"
            context_parts.append(f"--- {chunk_ref} ---\n{c['text']}")
            citations.append({
                "id": c["id"],
                "heading": c["heading"],
                "snippet": c["snippet"],
                "score": c.get("score", 0.0),
            })

        context_str = "\n\n".join(context_parts) if context_parts else "No specific clauses found."

        user_content = f"""DOCUMENT: {document_filename}

RETRIEVED CONTRACT CLAUSES:
{context_str}

USER QUESTION:
{question.strip()}

Please analyze the question against the retrieved clauses and provide a beautifully structured, plain-English response. Follow the required sections (📌 Key Takeaway, ⚖️ Legal Analysis & Clauses, ⚠️ Risks & Considerations, 💡 Practical Advice & Next Steps) and conclude with the 3 JSON follow-ups."""

        raw_response = await self._generate(user_content, system_prompt=_RAG_ANSWER_SYSTEM_PROMPT)

        # Parse follow_ups from JSON block if present
        follow_ups = []
        cleaned_answer = raw_response

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                follow_ups = parsed.get("follow_ups", [])
                # Strip the json block from the displayed answer
                cleaned_answer = raw_response[:json_match.start()].strip()
            except Exception:
                pass

        # Fallback default follow-ups if model omitted or malformed
        if not follow_ups or len(follow_ups) < 2:
            follow_ups = [
                "What are the termination conditions?",
                "Are there any financial penalties or fees mentioned?",
                "What obligations do I have upon breach of contract?",
            ]

        return {
            "answer": cleaned_answer,
            "follow_ups": follow_ups[:3],
            "citations": citations,
        }

    async def generate_initial_starter_questions(
        self, chunks: List[Dict[str, Any]], filename: str
    ) -> List[str]:
        """Generate 4 tailored starter questions based on the document's introductory chunks."""
        sample_text = "\n\n".join(c["text"] for c in chunks[:3])
        prompt = f"""Based on this legal document preview from '{filename}':
{sample_text[:2000]}

Suggest 4 concise, high-value questions that someone reviewing this document would want answered first.
Return ONLY a valid JSON array of 4 strings, e.g.:
["What are the key obligations?", "How can either party terminate?", "What are the liability limits?", "What are the payment deadlines?"]"""

        try:
            raw = await self._generate(prompt)
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                if isinstance(questions, list) and len(questions) >= 3:
                    return [str(q).strip('" ') for q in questions[:4]]
        except Exception as e:
            logger.warning("Failed to generate starter questions: %s", e)

        return [
            "What are the main obligations in this contract?",
            "What are the conditions for termination?",
            "Are there any indemnification or liability clauses?",
            "What are the payment terms and deadlines?",
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to Gemini and return text response."""
        try:
            config = genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
            )
            if system_prompt:
                config.system_instruction = system_prompt

            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            text = response.text
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text
        except Exception as exc:
            logger.exception("Gemini generation failed")
            raise RuntimeError(f"AI generation error: {exc}") from exc
