# LexAI — RAG-Powered Legal Document Assistant & Chatbot

> **AI-powered legal contract analysis & conversational intelligence** — upload any contract, agreement, or policy and ask questions in plain English with clause-level citations, risk alerts, and actionable legal insights.

---

## 🏗 System Architecture & RAG Pipeline

```
┌────────────────────────────────────────────────────────┐
│             Browser UI (Vanilla HTML/CSS/JS)            │
│  • File Upload & Dropzone  • Dynamic RAG Chat Interface │
│  • Visual Section Formatting  • Citations Drawer       │
│  • Speech Synthesis (Read Aloud)  • Follow-up Chips    │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI Backend                       │
│  /api/upload  ──> DocumentService + RAGService (Index) │
│  /api/chat    ──> RAGService (Retrieve) + AIService    │
│  /health      ──> Model & Status Check                 │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       Text Chunks + Vectors    Gemini 2.0 Flash
                │                        │
┌───────────────▼──────────┐   ┌─────────▼───────────────┐
│  Vector & Hybrid Index   │   │  Google GenAI SDK       │
│  • gemini-embedding-001  │   │  • Grounded Generation  │
│  • Cosine + Lexical BM25 │   │  • Clause Citations     │
└──────────────────────────┘   └─────────────────────────┘
```

---

## ✨ Core Features

| Feature | Description |
|---|---|
| 📄 **Document Indexing** | Upload PDF, DOCX, or TXT (up to 20 MB). Extracts and breaks into overlapping semantic clauses. |
| 🔍 **Clause-Level RAG** | Vector embeddings via `gemini-embedding-001` paired with hybrid lexical retrieval for pinpoint precision. |
| 💬 **Conversational Legal Chatbot** | Ask any question about obligations, termination, penalties, breach, or governing law. |
| 📑 **Inspectable Clause Citations** | Every answer provides an expandable citations drawer showing the exact text chunks retrieved from your contract. |
| 📊 **Visually Structured Responses** | Answers are partitioned into: **📌 Key Takeaway**, **⚖️ Legal Analysis & Clauses**, **⚠️ Risks & Considerations**, and **💡 Practical Advice & Next Steps**. |
| 💡 **Dynamic Follow-Up Suggestions** | Generates context-aware follow-up question chips after every response. |
| 🔊 **Voice Accessibility** | In-browser Text-to-Speech (`🔊 Read Aloud`) and one-click markdown export (`⬇ Save`). |

---

## ⚙️ Local Development

### 1. Environment Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
HOST=0.0.0.0
PORT=8000
MAX_UPLOAD_SIZE_MB=20
```

### 2. Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI dev server
python main.py
```

The app will be live at `http://localhost:8000`.

### 3. Run Test Suite

```bash
python -m pytest tests/ -v
```

---

## 🐳 Docker Deployment

The application includes a production-ready, multi-platform `Dockerfile` and `docker-compose.yml`.

### Option A: Using Docker CLI

```bash
# 1. Build the Docker image
docker build -t lexai-legal-assistant .

# 2. Run the container passing your API key
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY=your_gemini_api_key_here \
  --name lexai \
  lexai-legal-assistant
```

Access the application at `http://localhost:8000`.

### Option B: Using Docker Compose

```bash
# Start container with .env values
docker compose up -d --build
```

---

## 🚀 Deploying to Vercel

LexAI is configured for Vercel deployment with serverless routing via `vercel.json` and `api/index.py`.

### Deploy via Vercel CLI

1. Install the Vercel CLI (if not already installed):
   ```bash
   npm i -g vercel
   ```

2. Link and deploy your project:
   ```bash
   vercel
   ```

3. Add your environment variables in the Vercel Dashboard (or via CLI):
   ```bash
   vercel env add GEMINI_API_KEY
   ```

4. Deploy to production:
   ```bash
   vercel --prod
   ```

### Deploy via GitHub (Vercel Web Dashboard)

1. Push your repository to GitHub.
2. In [Vercel Dashboard](https://vercel.com), click **Add New...** -> **Project**.
3. Import your GitHub repository.
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: Your Google Gemini API Key.
5. Click **Deploy**. Vercel will build and serve the application using the configuration in `vercel.json`.

---

## 🌐 Deploying Docker to Cloud Platforms (Render, Railway, Fly.io, Cloud Run)

Since Vercel's native engine is serverless, you can also deploy the included `Dockerfile` directly to container hosting platforms in one click:

- **Railway / Render**: Connect your repo, select Dockerfile, and add `GEMINI_API_KEY`.
- **Google Cloud Run**:
  ```bash
  gcloud run deploy lexai --source . --set-env-vars GEMINI_API_KEY=your_key
  ```
- **Fly.io**:
  ```bash
  fly launch
  fly secrets set GEMINI_API_KEY=your_key
  ```

---

## ⚖️ Legal Disclaimer

LexAI provides automated informational assistance and document analysis for educational and comprehension purposes. It does not constitute formal legal counsel or create an attorney-client relationship. Always consult a licensed attorney for binding legal matters.
