# Law AI Backend - Data Storage Architecture 💾

## Overview

The Law AI Backend uses **three layers** of data storage:
1. **SQL Database (SQLite/PostgreSQL)** - Persistent relational data
2. **In-Memory Caches (Python Dictionaries)** - Session state and fast lookups
3. **Vector Databases (ChromaDB)** - Semantic search for RAG

---

## 1. SQL Database Tables 🗄️

**Database File**: `app.db` (SQLite in dev mode)

### Table: `users`
Stores user account information.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | Integer | Primary Key, Auto-increment | Unique user ID |
| `external_id` | String(255) | Unique, Not Null, Indexed | External auth ID (e.g., Azure AD) |
| `created_at` | DateTime | Auto-set | Account creation timestamp |

**Relationships**: 
- One-to-Many with `cases` (a user can have multiple cases)

---

### Table: `cases`
Stores uploaded case documents and their summaries.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | Integer | Primary Key, Auto-increment | Unique case ID |
| `user_id` | Integer | Foreign Key → `users.id`, Not Null, Indexed | Owner of the case |
| `filename` | String(512) | Nullable | Original uploaded filename |
| `text` | Text | Not Null | **Full case text** (original document) |
| `case_summary` | Text | Nullable | **JSON-formatted summary** (structured) |
| `created_at` | DateTime | Auto-set | Upload timestamp |

**Relationships**:
- Many-to-One with `users`
- One-to-Many with `question_answers`

**Notes**:
- `text` contains the complete uploaded case document
- `case_summary` is a JSON string with structured sections (property_division, children_parenting, etc.)

---

### Table: `question_answers`
Stores conversation history for each case.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | Integer | Primary Key, Auto-increment | Unique Q&A pair ID |
| `case_id` | Integer | Foreign Key → `cases.id`, Not Null, Indexed | Related case |
| `user_id` | Integer | Foreign Key → `users.id`, Not Null, Indexed | Owner of the case/Q&A |
| `question` | Text | Not Null | User's question |
| `answer` | Text | Not Null | AI's response |
| `topic` | String(50) | Nullable | Category (e.g., 'property_division') |
| `sources` | JSON | Nullable | RAG citation metadata |
| `context_snapshot` | Text | Nullable | Case summary section used for this answer |
| `created_at` | DateTime | Auto-set | Q&A timestamp |

**Relationships**:
- Many-to-One with `cases`

**Notes**:
- Persistent backup of all conversations
- `sources` stores retrieved precedent cases with scores
- `context_snapshot` preserves the exact summary used (in case it changes later)

---

## 2. In-Memory Caches 🧠

**Location**: `app/api/routes.py` (module-level variables)

### Cache: `session_case_ids`
**Type**: `Dict[str, int]`

Maps user/session IDs to their active case ID.

```python
session_case_ids = {
    "user_123": 45,       # User 123 is working on case ID 45
    "session_abc": 67     # Anonymous session working on case ID 67
}
```

**Purpose**: Quickly determine which case a user is currently working with.

---

### Cache: `case_summary_sections`
**Type**: `Dict[int, Dict[int, Dict[str, str]]]` (user_id → case_id → sections)

Parsed case summaries organized by topic for fast retrieval.

```python
case_summary_sections = {
    10: {  # User ID 10
        45: {  # Case ID 45
            "property_division": "The parties own a house worth $800k...",
            "children_parenting": "Two children aged 5 and 7...",
            "spousal_maintenance": "Wife seeking $2000/month..."
        },
        67: {  # Case ID 67
            "property_division": "Assets include business worth $2M..."
        }
    }
}
```

**Purpose**: Avoid re-parsing JSON summaries on every query. Enables topic-specific context.

---

### Cache: `pending_clarifications`
**Type**: `Dict[int, Dict[int, Dict[str, object]]]` (user_id → case_id → pending data)

Stores clarification questions awaiting user responses.

```python
pending_clarifications = {
    10: {
        45: {
            "question": "How should property be divided?",
            "topic": "property_division",
            "questions": [
                "What is the total value of the asset pool?",
                "What were each party's financial contributions?"
            ],
            "missing_fields": ["asset_pool", "contributions"]
        }
    }
}
```

**Purpose**: Track multi-turn clarification workflows. Cleared after user responds.

---

### Cache: `session_history`
**Type**: `Dict[int, Dict[int, list[dict]]]` (user_id → case_id → history)

Recent conversation history for context in follow-up questions.

```python
session_history = {
    10: {
        45: [
            {
                "role": "user",
                "content": "How should the house be divided?",
                "topic": "property_division"
            },
            {
                "role": "assistant",
                "content": "Based on the Family Law Act...",
                "topic": "property_division"
            },
            {
                "role": "user",
                "content": "What about the mortgage?",
                "topic": "property_division"
            }
        ]
    }
}
```

**Purpose**: 
- Provide conversational context to the LLM
- Limited to last 20 messages to prevent memory bloat
- Complements SQL history (faster access for recent messages)

---

## 3. Vector Database Collections 🔍

**Technology**: ChromaDB (persistent vector store)  
**Embedding Model**: OpenAI `text-embedding-3-large` (3072 dimensions)

### Collection: `cases_full`
**Location**: `./chroma_db_cases/`  
**Purpose**: Full-text precedent cases from AustLII

**Documents**:
- Australian Family Court cases (FamCA, FamCAFC)
- Chunked into ~1000 token segments
- Used for finding similar precedent cases

**Metadata**:
```python
{
    "source": "[2023] FamCA 123",
    "citation": "Smith v Smith",
    "court": "Family Court of Australia",
    "year": "2023"
}
```

**Built by**: `scripts/build_embeddings.py`

---

### Collection: `rules_statutes`
**Location**: `./chroma_db_statutes/`  
**Purpose**: Legislation and statutory provisions

**Documents**:
- Family Law Act 1975
- Family Law Regulations
- Key sections relevant to family law disputes

**Metadata**:
```python
{
    "source": "Family Law Act 1975",
    "section": "79",
    "title": "Property settlement orders"
}
```

**Built by**: `scripts/build_embeddings.py --statutes`

---

### Collection: `cases_summary`
**Location**: `./chroma_db_case_summaries/`  
**Purpose**: Condensed summaries of precedent cases

**Documents**:
- Summaries instead of full case text
- Faster retrieval, lower token usage
- Contains outcome and reasoning

**Metadata**:
```python
{
    "source_type": "case_summary",           # Type identifier
    "case_id": "[2023] FamCA 123",          # Unique case citation
    "summary_section": "property_division",  # Topic section
    "impact_analysis": "The court applied..."# Strategic significance (NEW!)
}
```

**Built by**: `scripts/build_embeddings.py --summaries`

---

### Collection: `uploaded_cases_index` (In-Memory Only!)
**Location**: RAM (not persisted)  
**Purpose**: User-uploaded case summaries

**Documents**:
- Generated from user's uploaded case document
- Parsed into topic sections (property_division, children_parenting, etc.)
- Created on-the-fly when user uploads a case

**Metadata**:
```python
{
    "source_type": "uploaded_case",
    "case_id": "45",
    "source": "smith_family_case.txt",
    "summary_section": "property_division"
}
```

**Lifecycle**: 
- Created: When user uploads a case
- Persists: For the server session only
- Cleared: When server restarts (reconstructed from SQL on next access)

**Why in-memory?**: 
- Each user's case is unique (not shared across users)
- Faster than disk I/O for temporary data
- Summary already stored in SQL (`cases.case_summary`)

---

## Data Flow Diagram 📊

```
┌─────────────────┐
│  User Uploads   │
│  Case Document  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  1. Save to SQL (cases.text)    │ ← Permanent storage
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2. Generate JSON Summary       │
│     (LLM call)                  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  3. Save summary to SQL         │ ← Permanent storage
│     (cases.case_summary)        │
└────────┬────────────────────────┘
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
┌──────────────────────┐   ┌──────────────────────────┐
│ 4a. Parse & Cache    │   │ 4b. Embed Summary        │
│     Sections         │   │     (OpenAI API)         │
│ (case_summary_       │   │                          │
│  sections cache)     │   │                          │
└──────────────────────┘   └────────┬─────────────────┘
                                    │
                                    ▼
                           ┌────────────────────────┐
                           │ 5. Store in Memory     │
                           │    (uploaded_cases_    │
                           │     index)             │
                           └────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  User Asks Question                                  │
└────────┬─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  RAG Pipeline (3 threads)                            │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────┐  │
│  │ Precedent    │  │ Legislation   │  │ User's   │  │
│  │ Cases        │  │ (Statutes)    │  │ Case     │  │
│  │ (cases_full) │  │ (rules_       │  │ (uploaded│  │
│  │              │  │  statutes)    │  │  _cases_ │  │
│  │              │  │               │  │  index)  │  │
│  └──────────────┘  └───────────────┘  └──────────┘  │
│         │                  │                  │      │
│         └──────────────────┴──────────────────┘      │
│                            │                         │
│                            ▼                         │
│                    ┌───────────────┐                 │
│                    │ LLM Synthesis │                 │
│                    └───────┬───────┘                 │
│                            │                         │
└────────────────────────────┼─────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Save Answer   │
                    │  to SQL        │
                    │  (question_    │
                    │   answers)     │
                    └────────────────┘
```

---

## Storage Usage Estimates 📈

### SQL Database (`app.db`)
- **Users**: ~100 bytes per user
- **Cases**: 
  - Text: 10-500 KB (original document)
  - Summary: 5-20 KB (JSON)
- **Q&A**: ~2-5 KB per exchange

**Total for 100 users with 5 cases each**: ~250 MB

---

### In-Memory Caches
- **session_case_ids**: ~100 bytes per session
- **case_summary_sections**: ~20 KB per case
- **pending_clarifications**: ~1-5 KB per pending clarification
- **session_history**: ~10-50 KB per user

**Total for 50 active sessions**: ~1-2 MB

---

### Vector Databases (ChromaDB)
- **cases_full**: ~500 MB - 2 GB (depends on corpus size)
- **rules_statutes**: ~50-200 MB
- **cases_summary**: ~100-500 MB
- **uploaded_cases_index** (RAM): ~1-5 MB per active case

**Total Disk Usage**: ~1-3 GB (static, built once)

---

## Cache Invalidation Rules 🔄

### When is data cleared?

| Cache | Cleared When | Restored From |
|-------|-------------|---------------|
| `session_case_ids` | `/reset` endpoint called | Not restored (user must re-upload) |
| `case_summary_sections` | Server restart | SQL (`cases.case_summary`) on first access |
| `pending_clarifications` | User submits clarification answers | Not needed (one-time use) |
| `session_history` | `/reset` endpoint called | SQL (`question_answers`) on first access |
| `uploaded_cases_index` | Server restart | Reconstructed from SQL summary on access |

### Data Persistence

| Storage Type | Survives Server Restart? | Survives DB Reset? |
|--------------|-------------------------|-------------------|
| SQL Database | ✅ Yes | ❌ No (deleted) |
| In-Memory Caches | ❌ No (rebuilt) | ❌ No |
| ChromaDB (static) | ✅ Yes | ✅ Yes (separate files) |
| ChromaDB (uploaded) | ❌ No (in-memory) | ❌ No |

---

## Debugging & Inspection 🔍

### Check SQL Data
```bash
# Install sqlite3 command line tool
sqlite3 app.db

# List tables
.tables

# View users
SELECT * FROM users;

# View cases
SELECT id, filename, user_id, substr(case_summary, 1, 50) FROM cases;

# View Q&A history
SELECT id, case_id, substr(question, 1, 30), substr(answer, 1, 30) FROM question_answers;
```

### Check In-Memory Caches
Add a debug endpoint in `app/api/routes.py`:
```python
@router.get("/debug/cache")
def debug_cache():
    return {
        "active_sessions": len(session_case_ids),
        "cached_cases": len(case_summary_sections),
        "pending_clarifications": len(pending_clarifications),
        "session_histories": len(session_history),
    }
```

### Check ChromaDB Collections
```python
from app.core.models import model_manager

# Get collection counts
model_manager.init_models()
model_manager.create_or_load_cases_index()
print(f"Cases: {model_manager.cases_collection.count()}")
print(f"Statutes: {model_manager.statutes_collection.count()}")
```

---

## Summary

✅ **SQL Database**: 3 tables (users, cases, question_answers) - Persistent  
✅ **In-Memory Caches**: 4 dictionaries - Session state & fast lookups  
✅ **ChromaDB Collections**: 4 indices (3 persistent, 1 in-memory) - Semantic search

**Total Storage**:
- SQL: ~250 MB for 100 users
- RAM: ~1-2 MB for 50 sessions
- Vector DBs: ~1-3 GB (one-time build)

**Data Flow**: Upload → SQL → Summary → Cache + Embeddings → Query → RAG → Answer → SQL
