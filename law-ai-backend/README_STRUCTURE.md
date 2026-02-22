# Law AI Backend - Organized Structure 📁

## Project Structure

```
law-ai-backend/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── api/                      # API endpoints
│   │   ├── __init__.py
│   │   └── routes.py            # FastAPI route handlers
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── rag_service.py       # RAG retrieval logic (formerly llama_law.py)
│   │   ├── clarify_service.py   # Clarification logic (formerly clarify.py)
│   │   ├── summary_service.py   # Summary generation (formerly summary_pipeline.py)
│   │   └── summary_prompt.py    # Summary prompts
│   ├── core/                     # Core configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── models.py            # ModelManager (LlamaIndex + ChromaDB)
│   │   ├── logger.py            # Logging setup
│   │   └── auth.py              # Authentication logic
│   └── db/                       # Database layer
│       ├── __init__.py
│       ├── database.py          # SQLAlchemy setup (formerly db.py)
│       └── models.py            # Database models (formerly db_models.py)
├── scripts/                      # Utility scripts
│   ├── build_embeddings.py      # Build vector indices
│   └── test_single_case_json.py # Testing utilities
├── data/                         # Data storage (gitignored)
│   ├── chroma_db_cases/         # Vector DB for cases
│   ├── chroma_db_statutes/      # Vector DB for statutes
│   └── chroma_db_case_summaries/# Vector DB for summaries
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Changes Made

### ✅ Reorganization
- **Separated concerns** into clear modules (API, services, core, DB)
- **Moved files** to appropriate locations
- **Created package structure** with proper `__init__.py` files
- **Updated all imports** to use new package paths

### 📦 Module Responsibilities

**app/api/** - HTTP endpoints and request/response handling
**app/services/** - Business logic (RAG, clarification, summarization)
**app/core/** - Configuration, authentication, model management
**app/db/** - Database models and connection management
**scripts/** - One-off utilities and build scripts

### ✅ Recent Schema/Cache Updates
- `question_answers` includes `user_id` (in addition to `case_id`)
- `case_summary_sections` cache is scoped by `user_id -> case_id -> section`

## Running the Application

### Development Mode
```bash
# From law-ai-backend directory
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Building Embeddings

```bash
# Build vector indices
python scripts/build_embeddings.py
```

## Environment Variables

Create a `.env` file with:
```
ENV=dev
DATABASE_URL=sqlite:///./app.db
DEV_DEFAULT_USER_ID=dev_user_123
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
OPENAI_EMBED_MODEL=text-embedding-3-large
```

## Next Steps

1. ⚠️ **Update frontend** to point to new API structure (if needed)
2. ⚠️ **Test all endpoints** after reorganization
3. ⚠️ **Move old main.py** to `main_old.py` as backup
4. ⚠️ **Rename main_new.py** to `main.py`
5. ✅ **Update imports** in any remaining files

## Benefits

- **Better organization**: Clear separation of concerns
- **Easier maintenance**: Find code faster
- **Scalability**: Add new features without clutter
- **Testing**: Easier to unit test individual modules
- **Collaboration**: Team members can work on separate modules
