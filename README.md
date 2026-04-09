# Sophie AI — Australian Family Law Assistant

An AI-powered legal research tool that analyses uploaded family law cases against a database of 74,440+ indexed AustLII precedent vectors, providing evidence-based legal analysis with case citations and statutory references.

> ⚠️ **Disclaimer:** This tool is for informational purposes only and does not provide legal advice.

---

## Features

- **Upload a case** — Upload a family law scenario (`.txt`) and get a structured summary
- **Ask questions** — Get detailed legal analysis grounded in real AustLII precedents
- **Clarification flow** — The system asks follow-up questions when key facts are missing
- **Topic detection** — Automatically routes questions to the right legal area (property, parenting, maintenance, etc.)
- **Evidence-based answers** — Every response cites specific case law and statutory provisions
- **Predicted outcome ranges** — Narrow percentage predictions (≤10 points) based on comparable decided cases
- **Chat history** — Persistent Q&A history per case, per user

---

## Architecture

```
User → React Frontend → FastAPI Backend
                            ├── ChromaDB (74,440 precedent vectors)
                            │   ├── cases_summary_gemini (topic summaries)
                            │   ├── cases_full_gemini (full judgment chunks)
                            │   └── rules_statutes_gemini (Family Law Act)
                            ├── In-memory index (uploaded case embeddings)
                            ├── Gemini Embeddings (gemini-embedding-2-preview)
                            ├── SQLite (users, cases, Q&A history)
                            └── Synthesis LLM (OpenAI / Anthropic / Gemini)
```

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 18+
- API keys: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

### 1. Backend

```bash
cd law-ai-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env → add your API keys

# Start the backend
python main.py
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd law-ai-frontend
npm install
npm run dev
```

- App: http://localhost:5173

---

## Azure Deployment (Production)

### Infrastructure

| Component | Service | Spec |
|---|---|---|
| Backend | Azure Container App | 4 CPU, 8Gi RAM, min-replicas 0 |
| Frontend | Azure Container App | 0.5 CPU, 1Gi RAM, min-replicas 0 |
| Images | Azure Container Registry (`sophieaicr`) | Built from GitHub |
| ChromaDB | Azure Files → local disk on startup | ~4.8GB, 74,440 vectors |
| Database | SQLite on local disk (ephemeral) | Users, cases, Q&A |

### URLs

| Service | URL |
|---|---|
| Frontend | `https://sophieai-frontend.delightfulcoast-830e07c4.australiaeast.azurecontainerapps.io` |
| Backend API | `https://sophieai-backend.delightfulcoast-830e07c4.australiaeast.azurecontainerapps.io` |
| API Docs | `https://sophieai-backend.delightfulcoast-830e07c4.australiaeast.azurecontainerapps.io/docs` |

### Deploy Commands

```bash
# Rebuild & deploy backend
az acr build --registry sophieaicr --image law-ai-backend:latest \
  --file Dockerfile https://github.com/SophieeeeeChen/Law_AI.git#main:law-ai-backend

az containerapp update --name sophieai-backend --resource-group sophieai-rg \
  --image sophieaicr.azurecr.io/law-ai-backend:latest --revision-suffix vN \
  --set-env-vars ENV=prd GOOGLE_API_KEY='...' OPENAI_API_KEY='...' ANTHROPIC_API_KEY='...' \
  CORS_ORIGINS=https://sophieai-frontend.delightfulcoast-830e07c4.australiaeast.azurecontainerapps.io

# Rebuild & deploy frontend
az acr build --registry sophieaicr --image law-ai-frontend:latest \
  --file Dockerfile https://github.com/SophieeeeeChen/Law_AI.git#main:law-ai-frontend

az containerapp update --name sophieai-frontend --resource-group sophieai-rg \
  --image sophieaicr.azurecr.io/law-ai-frontend:latest --revision-suffix vN
```

### Check Logs

```bash
az containerapp logs show --name sophieai-backend --resource-group sophieai-rg --follow
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload_case` | Upload a case file for analysis |
| `POST` | `/ask` | Ask a question about an uploaded case |
| `POST` | `/clarify` | Submit answers to clarification questions |
| `POST` | `/reset` | Clear cached data for a case |
| `GET` | `/history/{session_id}` | Get full Q&A history for a user |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger API documentation |

---

## Project Structure

```
law-ai-backend/
├── main.py                          # FastAPI entry point & startup
├── start.sh                         # Azure container startup script
├── Dockerfile                       # Backend container image
├── requirements.txt                 # Python dependencies
├── build_embeddings.py              # Batch indexing for AustLII cases
├── PROCESSING.md                    # Detailed pipeline documentation
├── app/
│   ├── core/
│   │   ├── config.py                # Configuration & environment variables
│   │   ├── models.py                # LLM, embeddings, vector index manager
│   │   ├── auth.py                  # Authentication (dev/prod modes)
│   │   ├── logger.py                # Logging setup
│   │   └── dev_logger.py            # Dev-only structured trace logging
│   ├── api/
│   │   └── routes.py                # All API endpoints & in-memory caches
│   ├── services/
│   │   ├── rag_service.py           # RAG retrieval & synthesis prompt
│   │   ├── summary_service.py       # Summary generation & section parsing
│   │   ├── summary_prompt.py        # LLM prompt for case summarization
│   │   └── clarify_service.py       # Topic detection & clarification logic
│   └── db/
│       ├── __init__.py              # Database session management
│       └── models.py                # SQLAlchemy models (User, Case, QA)
├── chroma_db/                       # Local ChromaDB (dev)
└── logs/                            # Dev trace logs

law-ai-frontend/
├── src/                             # React + TypeScript source
├── nginx.conf                       # Production nginx config (SPA routing)
├── Dockerfile                       # Frontend container image
├── .env.production                  # Production API URL
├── package.json
└── vite.config.ts
```

---

## Configuration

Key settings in `app/core/config.py` (override via environment variables):

| Setting | Default | Description |
|---|---|---|
| `ENV` | `dev` | Environment: `dev` or `prd` |
| `SYNTHESIS_LLM` | `openai` | Synthesis provider: `openai` / `anthropic` / `gemini` |
| `SYNTHESIS_OPENAI_LLM_MODEL` | `gpt-5.2` | OpenAI model for answer generation |
| `SYNTHESIS_ANTHROPIC_LLM_MODEL` | `claude-sonnet-4-20250514` | Anthropic model for answer generation |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini LLM for topic detection & summarization |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2-preview` | Embedding model for all vectors |
| `TOP_K` | `5` | Number of precedent cases to retrieve |
| `HISTORY_MAX_TURNS` | `6` | Chat history turns included in context |
| `HYBRID_VECTOR_WEIGHT` | `0.6` | Semantic search weight in hybrid retrieval |
| `HYBRID_BM25_WEIGHT` | `0.4` | Keyword search weight in hybrid retrieval |

---

## Example Usage

1. **Upload** a family law case scenario (plain text file)
2. **Select a topic** (property division, parenting, maintenance, etc.) or let the system detect it
3. **Ask a question**, e.g.:
   - *"How would a court likely assess the parties' contributions?"*
   - *"What is the likely property division outcome?"*
   - *"Can the wife get spousal maintenance?"*
4. **Answer clarification questions** if the system needs more details
5. **Review the analysis** — includes precedent citations, statutory references, and predicted outcome ranges

---

## Documentation

- **[PROCESSING.md](PROCESSING.md)** — Detailed pipeline documentation (data flow, retrieval strategy, prompt structure, deployment details)