# Law AI Backend — Project Structure
🚀 High-Lighted Feature: "Legal-Context Segmented RAG"
Unlike standard RAG that chunks text blindly, this project uses a Structure-Aware Retrieval system specifically designed for Australian Family Law (AustLII cases).

1. Semantic Pre-Processing
Each AustLII case and uploaded document is parsed and categorized into four critical legal domains:

Property Division: Financial settlements and asset splits.

Child Parenting: Custody, living arrangements, and welfare.

spousal_maintenance

family_violence_safety

prenup_postnup: Pre/post-nuptial agreement

Outcome Orders: The final binding orders made by the court.

Reasons & Rationales: The judicial logic and precedents applied.

2. Segmented Vector Search
When a user asks a question, they select a Focus Section.

Targeted Retrieval: The system restricts the vector search to the selected metadata category (e.g., searching only within property_division chunks).

High Precision: This eliminates "noise" from other parts of the case that might contain similar keywords but different legal contexts.

3. Full-Text Traceability
Every retrieved chunk acts as a "Smart Node."

Relational Mapping: Each node maintains a pointer to the Full Case Text.

Context Expansion: Users can instantly jump from a specific "Rationale" chunk to the complete AustLII judgment to verify the legal context.

Python
# Example of your targeted retrieval logic
results = vector_db.query(
    query_text="Section 79 property settlement",
    where={"section": "property_division"}  # This is the "secret sauce"
)

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
