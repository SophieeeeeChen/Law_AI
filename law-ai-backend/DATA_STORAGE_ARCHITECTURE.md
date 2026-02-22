# Law AI Backend — Data Storage Architecture

## Overview

The backend uses three storage layers:

1. **SQL database** (SQLite/PostgreSQL) for persistent relational data  
2. **In‑memory caches** (Python dicts) for fast session state  
3. **Vector database** (ChromaDB) for semantic search (RAG)

---

## 1) SQL Database

**Dev DB file**: `app.db` (SQLite)

### Table: users
Stores user accounts.

| Column | Type | Notes |
|---|---|---|
| id | Integer | PK |
| external_id | String(255) | Unique |
| created_at | DateTime | Auto |

### Table: cases
Stores uploaded cases and summaries.

| Column | Type | Notes |
|---|---|---|
| id | Integer | PK |
| user_id | Integer | FK → users.id |
| filename | String(512) | Original name |
| text | Text | Full case text |
| case_summary | Text | JSON summary |
| created_at | DateTime | Auto |

### Table: question_answers
Conversation history per case.

| Column | Type | Notes |
|---|---|---|
| id | Integer | PK |
| case_id | Integer | FK → cases.id |
| user_id | Integer | FK → users.id |
| question | Text | User question |
| answer | Text | AI answer |
| topic | String(50) | Optional category |
| sources | JSON | RAG citations |
| context_snapshot | Text | Summary used |
| created_at | DateTime | Auto |

**Why SQL?**  
- Durable storage for uploads, summaries, and Q&A history  
- Supports auditability and reload on restart

---

## 2) In‑Memory Caches(to be updated: use redis)

**Location**: `app/api/routes.py` (module globals)

### session_case_ids
Maps user/session to active case.

```python
session_case_ids = {"user_123": 45}
```

### case_summary_sections
Parsed summary sections per user → case.

```python
case_summary_sections = {
  10: {45: {"property_division": "...", "children_parenting": "..."}}
}
```

### pending_clarifications
Tracks clarification flows.

```python
pending_clarifications = {
  10: {45: {"questions": ["..."], "missing_fields": ["..."]}}
}
```

### session_history
Recent conversational turns for context.

```python
session_history = {
  10: {45: [{"role": "user", "content": "..."}]}
}
```

**Why in‑memory?**  
- Fast access for active sessions  
- Avoids repeated JSON parsing

---

## 3) Vector Database (ChromaDB)

**Storage root**: `law-ai-backend/chroma_db`  
**Embedding model**: OpenAI `text-embedding-3-large`

### Collection: cases_full
Full AustLII case text, chunked.

**Use**: precedent retrieval  
**Built by**: `scripts/build_embeddings.py`

### Collection: rules_statutes
Legislation and regulations.

**Use**: statute retrieval  
**Built by**: `scripts/build_embeddings.py --statutes`

### Collection: cases_summary
Topic‑section summaries of cases.

**Use**: faster semantic retrieval  
**Metadata** includes `impact_analysis`

---

## Data Flow (High Level)

1. Upload case → store in SQL  
2. Generate summary → store in SQL  
3. Cache summary sections in memory  
4. (Optional) Embed summaries into Chroma  
5. Query → retrieve from Chroma + cached summary → answer → store Q&A in SQL

---

## Cache Lifecycle

| Cache | Cleared When | Restored From |
|---|---|---|
| session_case_ids | /reset | Not restored |
| case_summary_sections | server restart | SQL summary |
| pending_clarifications | after clarification | N/A |
| session_history | /reset | SQL Q&A |

---

## Storage Notes

- **SQL**: durable, single source of truth  
- **Caches**: fast, session‑scoped  
- **ChromaDB**: persistent semantic indices  

---

## Debugging

### SQL quick checks
```bash
sqlite3 app.db
.tables
SELECT * FROM users;
SELECT id, user_id, filename FROM cases;
```

### Chroma collection counts
```python
from app.core.models import model_manager
model_manager.init_models()
model_manager.create_or_load_cases_index()
print(model_manager.cases_collection.count())
```

---

## Summary

- SQL persists users, cases, and Q&A  
- In‑memory caches keep live session state  
- ChromaDB enables semantic search over cases and statutes
