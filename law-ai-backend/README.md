# Sophie AI — Australian Family Law Assistant

## Demo Deployment Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** (for React frontend)
- **API keys** set in `.env`:
  - `GOOGLE_API_KEY` (for Gemini embeddings + LLM)
  - `OPENAI_API_KEY` (if using OpenAI for synthesis)
  - `ANTHROPIC_API_KEY` (if using Claude for synthesis)

---

### Option A: Local Development (Recommended for Demo)

#### Step 1: Set up backend environment

```bash
cd law-ai-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Configure environment

```bash
copy .env.example .env
# Edit .env → add your API keys
```

#### Step 3: Start the backend

```bash
python main.py
```

Backend runs at http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

#### Step 4: Start the React frontend (new terminal)

```bash
cd law-ai-frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173

---

### Option B: Docker Deployment

```bash
cd law-ai-backend
docker compose up --build -d
```

| Service   | URL                        |
|-----------|----------------------------|
| Backend   | http://localhost:8000       |
| Frontend  | http://localhost:5173       |

---

### Architecture

```
User → React Frontend (5173)
         ↓
       FastAPI Backend (8000)
         ├── ChromaDB (embedded vector store)
         ├── Gemini Embeddings (semantic search)
         ├── BM25 (keyword search)
         ├── Hybrid Retrieval (RRF fusion)
         └── LLM Synthesis (OpenAI / Anthropic / Gemini)
              ↓
         Response with citations
```

### Project Structure

```
law-ai-backend/
├── main.py                    # FastAPI entry point
├── app/
│   ├── core/
│   │   ├── config.py          # All configuration
│   │   ├── models.py          # Model manager (embeddings, LLM, indices)
│   │   └── logger.py          # Logging
│   ├── api/
│   │   └── routes.py          # API endpoints (/query, etc.)
│   └── db/                    # Database layer
├── chroma_db/                 # Vector store (auto-created)
├── data/                      # SQLite DB + data files
├── docker-compose.yml         # Docker orchestration
├── Dockerfile                 # Backend container
└── requirements.txt           # Python dependencies

law-ai-frontend/
├── src/                       # React source code
├── package.json
└── vite.config.ts
```

### Example Demo Questions

1. "What factors does the court consider for the best interests of the child?"
2. "What is the definition of family violence under the Family Law Act?"
3. "Explain parental responsibility under the Act"
4. "What are the grounds for property settlement?"
5. "How does Section 60CC work?"

### Key Config (in `app/core/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `SYNTHESIS_LLM` | `openai` | LLM provider: openai / anthropic / gemini |
| `TOP_K` | `5` | Documents to retrieve |
| `HYBRID_VECTOR_WEIGHT` | `0.6` | Weight for semantic search |
| `HYBRID_BM25_WEIGHT` | `0.4` | Weight for keyword search |
| `HYBRID_USE_RERANK` | `false` | Enable cross-encoder reranking |
