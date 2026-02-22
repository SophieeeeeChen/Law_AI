# Law AI Backend — Project Structure

## Structure

```
law-ai-backend/
├── app/
│   ├── api/                    # FastAPI routes
│   ├── services/               # RAG, summary, clarification logic
│   ├── core/                   # Config, model manager, auth, logging
│   └── db/                     # SQLAlchemy database layer
├── scripts/
│   └── build_embeddings.py     # HTML→MD conversion + embeddings
├── chroma_db/                  # Persistent ChromaDB (gitignored)
├── logs/                       # Logs (gitignored)
├── main.py                     # App entry
├── requirements.txt
└── README.md
```

## What’s in Each Module

- **app/api**: request/response handling  
- **app/services**: RAG, summarization, clarifications  
- **app/core**: config, auth, model manager, logging  
- **app/db**: models + database setup  
- **scripts**: data prep and embedding builds  

## Running the App

**Dev**
```bash
python main.py
```

**Prod**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Build Embeddings

**Convert HTML tree → markdown + build case/summaries**
```bash
python scripts/build_embeddings.py --convert-html-md-tree --cases --summaries
```

**Statutes only**
```bash
python scripts/build_embeddings.py --statutes
```

## Environment

Create `.env`:
```
ENV=dev
DATABASE_URL=sqlite:///./app.db
DEV_DEFAULT_USER_ID=dev_user_123
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
OPENAI_EMBED_MODEL=text-embedding-3-large
```

## Notes

- ChromaDB persists under `chroma_db/`.
- Logs are written under `logs/`.
- HTML conversion uses Trafilatura (markdown output).
