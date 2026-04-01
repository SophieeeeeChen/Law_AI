# Law AI Backend — Data Storage Architecture

## Overview

The backend uses three storage layers:

1. **SQL database** (SQLite/PostgreSQL) for persistent relational data  
2. **In‑memory caches** (Python dicts) for fast session state  
3. **Vector database** (ChromaDB) for semantic search (RAG)

---
## 1) SQL Database

The SQL database serves as the primary persistent storage layer, maintaining all critical relational data including user accounts, case documents, and conversation history.

**Database**: SQLite (development) / PostgreSQL (production)  
**File location**: `app.db`

### Schema Design

#### users
User account management and authentication.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | PRIMARY KEY | Internal user identifier |
| external_id | String(255) | UNIQUE, NOT NULL | External authentication ID |
| created_at | DateTime | DEFAULT NOW | Account creation timestamp |

#### cases
Uploaded legal cases with full text and AI-generated summaries.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | PRIMARY KEY | Case identifier |
| user_id | Integer | FOREIGN KEY → users.id | Case owner |
| filename | String(512) | NOT NULL | Original upload filename |
| case_summary | Text | | Structured JSON summary (sections, facts, issues) |
| created_at | DateTime | DEFAULT NOW | Upload timestamp |

**Indexes**: `user_id`, `created_at`

#### question_answers
Complete audit trail of all user-AI interactions per case.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | PRIMARY KEY | Q&A record identifier |
| case_id | Integer | FOREIGN KEY → cases.id | Associated case |
| user_id | Integer | FOREIGN KEY → users.id | Question author |
| question | Text | NOT NULL | User's question |
| answer | Text | NOT NULL | AI-generated response |
| topic | String(50) | | Conversation topic/category |
| sources | JSON | | RAG retrieval sources and citations |
| context_snapshot | Text | | Summary context used for answer |
| created_at | DateTime | DEFAULT NOW | Interaction timestamp |

**Indexes**: `case_id`, `user_id`, `created_at`

### Design Rationale

**Why SQL?**
- **Durability**: Survives server restarts; single source of truth
- **Auditability**: Complete history of uploads and conversations
- **Relational integrity**: Enforces user-case-conversation relationships
- **Query flexibility**: Supports filtering, pagination, and analytics
- **Portability**: Easy migration from SQLite (dev) to PostgreSQL (prod)

### Data Retention

- User accounts: retained indefinitely
- Cases: retained until user deletion
- Q&A history: retained for audit compliance (configurable retention period)

## 2) In‑Memory Caches

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
  "user_id": {
    "case_id": {
      "facts": "- Fact: The parties were in a relationship (including marriage) for approximately six and a half years.\n- Fact: They have two children aged under 10.\n- Fact: Parenting proceedings have been resolved by consent and the children live in an equal shared care arrangement.\n- Fact: There is no dispute that it is just and equitable to alter property interests.\n- Fact: At trial the court identifies a single net property pool of approximately $2.9 million including superannuation and notional property.\n- Fact: Existing assets are approximately $2.36 million and include sale proceeds of the former matrimonial home held in trust ($145,630), the husband's superannuation (~$236,000), the wife's superannuation (~$45,000), the husband's business/investment interests including trusts/companies (~$1.6 million), and a boat and boat licence held by the wife (~$295,000).\n- Fact: Liabilities include the wife's credit card debt of approximately $5,500.\n- Fact: Loans from parents incurred by both parties to fund legal fees are excluded from the pool.\n- Fact: Notional property/add-backs total approximately $545,000 and include interim distributions to the wife (~$370,000) and to the husband (~$110,000), added back by agreement, and a further ~$65,000 trust distribution to the husband's mother in dispute as an add-back.\n- Fact: The husband entered the relationship with very substantial initial financial contributions estimated at approximately $1.7\u20131.8 million including property, business interests, trust structures, loans and superannuation.\n- Fact: The wife entered the relationship with negligible assets and minimal superannuation.\n- Fact: During the relationship the parties adopted traditional roles: husband as primary income earner and business operator; wife primarily responsible for homemaking and parenting and largely ceased paid work shortly before the first child's birth.\n- Fact: Post-separation, the husband continued to generate income and control business/trust structures, while the wife relied on interim distributions, government benefits, child support and spousal maintenance orders.\n- Fact: The husband is in his late 40s, in good health, and continues to earn income through businesses and entities he controls.\n- Fact: The wife is in her mid-30s, in good health, has been out of the workforce for several years and is retraining through tertiary study with anticipated graduate income of about $60,000 per annum in the medium term.\n- Fact: There is a significant short- to medium-term income disparity.",
      "issues": "- Issue: Property settlement under s 79 Family Law Act 1975 (Cth)\n- Issue: Identification and valuation of the net property pool including superannuation and notional property/add-backs\n- Issue: Assessment of contributions (initial, financial, non-financial, homemaker/parenting) in a relatively short relationship\n- Issue: Whether and to what extent to make an adjustment for future needs under s 75(2)\n- Issue: Whether an additional add-back should be made for a trust distribution to the husband's mother\n- Issue: Spousal maintenance (claimed by the wife) alongside property adjustment",
      "property_division": "- Asset Pool: Single identified pool including superannuation and notional property, net approximately $2.9 million.\n- Asset Pool: Existing assets (~$2.36m): sale proceeds held in trust ($145,630 joint); husband's superannuation (~$236k); wife's superannuation (~$45k); husband's business/investment interests via personal and trust/company structures (~$1.6m); wife's boat and boat licence (~$295k); minor accounts/personal property nominal.\n- Asset Pool: Liabilities: wife's credit card ~ $5,500.\n- Asset Pool: Excluded: parental loans used for legal fees (excluded from the pool).\n- Asset Pool: Notional property/add-backs (~$545k): wife's interim distributions ~ $370k (added back by agreement); husband's interim distributions ~ $110k (added back by agreement); husband's trust distribution to his mother ~ $65k (in dispute as an add-back).\n- Contributions: Husband's initial financial contributions were very substantial (~$1.7\u20131.8m) including real property, business interests, trust structures, loans and superannuation.\n- Contributions: Wife's initial contributions were negligible with minimal superannuation.\n- Contributions: During the relationship, husband's ongoing contributions included income earning, operating/growing business interests, and some home improvements.\n- Contributions: During the relationship, wife's non-financial contributions included primary homemaking and parenting, with cessation of paid employment shortly before first child.\n- Contributions: Post-separation, husband continued to control and benefit from business/trust structures and generate income.\n- Contributions: Post-separation, wife relied on interim distributions and support (government benefits, child support, spousal maintenance orders).\n- Future Needs: Husband: late 40s, good health, continuing income from businesses and controlled entities; asserted capacity to influence distributions and obtain indirect benefits.\n- Future Needs: Wife: mid-30s, good health, out of workforce for years; retraining via tertiary study; anticipated graduate income ~ $60,000 per annum in medium term.\n- Future Needs: Significant short- to medium-term income disparity is identified.\n- Future Needs: Equal shared care may affect each party's household costs but specific expense evidence is not provided in the text.\n- Future Needs: Wife's new relationship is raised; absence of evidence about cohabitation/financial support may permit adverse inference (Jones v Dunkel) relevant to needs.\n- Just & Equitable: It is not in dispute that it is just and equitable to alter property interests.\n- Just & Equitable: Central evaluative question is the balance between strong initial contributions by the husband in a relatively short relationship and the wife's substantial homemaker/parenting contributions plus any s 75(2) adjustment.\n- Living Arrangements: Children are in an equal shared care arrangement by consent.\n- Living Arrangements: Wife has commenced a new relationship but details of cohabitation and financial support are not provided.\n- Existing Agreements: Interim distributions to both parties are agreed to be added back (~$370k to wife; ~$110k to husband).\n- Existing Agreements: Parenting arrangements are resolved by consent.\n- Existing Agreements: No financial agreement (prenup/postnup) is described.",
      "spousal_maintenance": "- Need: Wife seeks spousal maintenance and has relied post-separation on interim distributions, government benefits, child support and spousal maintenance orders.\n- Need: Wife has been out of the workforce for several years and is retraining; anticipated income ~ $60,000 per annum in medium term suggests claimed short- to medium-term need.\n- Capacity to Pay: Husband continues to earn income through businesses and entities he controls, with asserted capacity to manipulate distributions and derive indirect benefits.\n- Statutory Factors: Income disparity between the parties in the short to medium term is identified as relevant.\n- Statutory Factors: Wife's new relationship may be relevant to maintenance needs but there is insufficient evidence about financial support.\n- Income & Expenses: Specific income figures (beyond wife's anticipated ~$60,000 medium term) and detailed budgets/expenses are not stated.\n- Income & Expenses: Husband's income is described qualitatively as continuing through controlled business structures.\n- Earning Capacity: Wife: currently retraining through tertiary study; anticipated graduate income about $60,000 per annum in medium term.\n- Earning Capacity: Husband: ongoing earning capacity via business interests and controlled entities.\n- Health & Care: Both parties are stated to be in good health.\n- Relationship Length: Relationship duration approximately six and a half years.\n- Standard of Living: Not specifically stated.",
      "children_parenting": "- Child Ages: Two children aged under 10.\n- Current Arrangements: Parenting proceedings resolved by consent with equal shared care.\n- Caregiver History: During the relationship the wife was primarily responsible for parenting and homemaking.\n- Availability: Equal shared care implies both parents have regular availability; further detail is not stated.\n- Best Interests: Parenting is resolved by consent; no best-interests reasoning is provided in the text.\n- Orders: Equal shared care arrangement by consent (details not specified).",
      "impact_analysis": "- Pivotal Finding: Whether the husband's initial contributions are accepted at ~$1.7\u20131.8m and how heavily they are weighted given the relationship length.\n- Pivotal Finding: How the wife's homemaker and primary parenting contributions are assessed in circumstances where she largely ceased paid employment and cared for two young children.\n- Pivotal Finding: Whether the husband's business/trust control supports findings about his true financial resources and capacity to benefit from entity structures beyond stated income.\n- Pivotal Finding: Whether the ~$65,000 trust distribution to the husband's mother is added back as notional property (including whether it is treated as a premature disposition to defeat the other party's claim or as a legitimate repayment).\n- Pivotal Finding: Whether an adverse inference (Jones v Dunkel) is drawn from the wife's failure to provide evidence of her new relationship and any financial support, and how that affects s 75(2) adjustment and/or maintenance.\n- Statutory Pivot: Family Law Act 1975 (Cth) s 79 (alteration of property interests; overall requirement of just and equitable outcome).\n- Statutory Pivot: Family Law Act 1975 (Cth) s 79(4) (contributions: financial, non-financial, homemaker/parenting; post-separation matters).\n- Statutory Pivot: Family Law Act 1975 (Cth) s 75(2) (future needs factors applied via s 79(4)(e)).\n- Statutory Pivot: If maintenance remains in issue: Family Law Act 1975 (Cth) s 72 and s 74 (threshold need and capacity; discretionary factors via s 75(2)).",
      "uncertainties": "- Uncertainties: No decided orders are provided; this is framed as a 'likely outcome' question rather than a judgment.\n- Uncertainties: No valuation methodology or expert evidence is described for the husband's business, trust and company interests (~$1.6m), which is commonly a major battleground.\n- Uncertainties: The factual basis for adding back the $65,000 distribution to the husband's mother is contested and not detailed (timing, intent, and whether it was a legitimate repayment).\n- Uncertainties: The extent of the husband's actual income, benefits, and access to funds through controlled entities is not quantified."
    }
  }
}
```

### pending_clarifications
Tracks clarification flows.
pending_clarifications
{
  "6": {
    "5": {
      "question": "How can we structure a 'no-contact' or 'low-contact' parenting plan that prioritizes child safety and eliminates the need for face-to-face interaction between the parents?",
      "topic": "family_violence_safety",
      "questions": [
        "Could you describe any specific incidents of violence, threats, or coercive control?",
        "Are there currently any AVOs, IVOs, or other protection orders in place?",
        "Have there been any police reports filed or criminal charges laid related to family violence?",
        "Were the children present during or exposed to the effects of any violent incidents?",
        "Do you currently have a safety plan or support services in place?"
      ],
      "missing_fields": [
        "incidents",
        "protection_orders",
        "police_court",
        "child_exposure",
        "safety_plan"
      ]
    }
  }
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
Metsdata: {'ref_doc_id': '0528a1cb-a490-4d85-ad07-ed1c74500769', 'doc_id': '0528a1cb-a490-4d85-ad07-ed1c74500769', 'source': 'FamCA_2000_102', 'source_type': 'case', '_node_type': 'TextNode', '_node_content': '{"id_": "3e2775f3-7ba4-4f05-8d5c-d9342c236b89", "embedding": null, "metadata": {"source_type": "case", "source": "FamCA_2000_102", "format": "markdown"}, "excluded_embed_metadata_keys": [], "excluded_llm_metadata_keys": [], "relationships": {"1": {"node_id": "0528a1cb-a490-4d85-ad07-ed1c74500769", "node_type": "4", "metadata": {"source_type": "case", "source": "FamCA_2000_102", "format": "markdown"}, "hash": "5f0b9e8a2cbf369e397d755e300c27f904d3a5ca71826f91b724b866f4493468", "class_name": "RelatedNodeInfo"}, "3": {"node_id": "c43b4b74-13af-4c94-bfcb-81060f408a6d", "node_type": "1", "metadata": {}, "hash": "82b2ed9befb88673f97cb367c798b973378bc48d0daafe196b3babf16621612e", "class_name": "RelatedNodeInfo"}}, "metadata_template": "{key}: {value}", "metadata_separator": "\\n", "text": "", "mimetype": "text/plain", "start_char_idx": 0, "end_char_idx": 3441, "metadata_seperator": "\\n", "text_template": "{metadata_str}\\n\\n{content}", "class_name": "TextNode"}', 'format': 'markdown', 'document_id': '0528a1cb-a490-4d85-ad07-ed1c74500769'}
**Use**: precedent retrieval  
**Built by**: `scripts/build_embeddings.py`

### Collection: rules_statutes
Legislation and regulations.

**Use**: statute retrieval  
**Built by**: `scripts/build_embeddings.py --statutes`

### Collection: cases_summary
In jsonl file:
{
  "case_id": "FamCA_2023_12345",
  "case_name": "FamCA_2023_12345",
  "summary_text": [
    "--- Summary ---",
    "Brief overview of the case including key facts and legal issues...",
    "",
    "--- Key Legal Principles ---",
    "Main legal principles applied in this case...",
    "",
    "--- Outcome ---",
    "Court's decision and orders made...",
    ""
  ],
  "summary_sections": [
    {
      "section": "summary",
      "text": "Brief overview of the case including key facts and legal issues..."
    },
    {
      "section": "key_legal_principles",
      "text": "Main legal principles applied in this case..."
    },
    {
      "section": "outcome",
      "text": "Court's decision and orders made..."
    }
  ]
}

Topic‑section summaries of cases.
--updated on 6/03
Document(
text=section_text,
metadata={
    "case_name": row.get("case_id"),  # FamCA_2020_122
    "summary_section": section_name,  # facts, issues, etc.
},
)

**Use**: faster semantic retrieval  
**Metadata** includes `impact_analysis`

---

### Inmemory embeddings
embedding by the following:
{
  "facts": "- Fact: The parties were in a relationship (including marriage) for approximately six and a half years.\n- Fact: They have two children aged under 10.\n- Fact: Parenting proceedings were resolved by consent and the children live in an equal shared care arrangement.\n- Fact: There is no dispute that it is just and equitable to alter property interests.\n- Fact: At trial a single pool is identified including superannuation and notional property with a net value of approximately $2.9 million.\n- Fact: Existing assets total approximately $2.36 million including sale proceeds of the former matrimonial home held in trust ($145,630 joint), the husband's superannuation (~$236,000), the wife's superannuation (~$45,000), the husband's business/investment interests through personal/trust/company structures (~$1.6 million), and a boat and boat licence held by the wife (~$295,000).\n- Fact: Liabilities include the wife's credit card debt of approximately $5,500.\n- Fact: Loans from parents incurred by both parties to fund legal fees are excluded from the pool.\n- Fact: Notional property/add-backs are approximately $545,000 including interim distributions to the wife (~$370,000) and to the husband (~$110,000) added back by agreement, and a trust distribution from the husband to his mother (~$65,000) in dispute but sought to be added back.\n- Fact: The husband entered the relationship with substantial initial financial contributions valued approximately $1.7–1.8 million (real property, business interests, trust structures, loans, and superannuation).\n- Fact: The wife entered the relationship with negligible assets and minimal superannuation.\n- Fact: During the relationship the parties adopted traditional roles: husband as primary income earner operating and growing business interests and doing some home improvements; wife primarily responsible for homemaking and parenting and largely ceased paid work shortly before the first child's birth.\n- Fact: Post-separation the husband continued to generate income and control business and trust structures, while the wife relied on interim distributions, government benefits, child support and spousal maintenance orders.\n- Fact: The husband is in his late 40s and in good health.\n- Fact: The wife is in her mid-30s and in good health, has been out of the workforce for several years, and is retraining via tertiary study with an anticipated graduate income of about $60,000 per annum in the medium term.\n- Fact: There is a significant short- to medium-term income disparity.",
  "issues": "- Issue: Property settlement under s 79 Family Law Act 1975 (Cth)\n- Issue: Identification and valuation of the net property pool including superannuation and alleged notional property/add-backs\n- Issue: Assessment of contributions (initial, financial, non-financial, homemaker/parenting) in a short-to-medium length relationship\n- Issue: Adjustment for future needs under s 75(2) including income disparity, care of children, and earning capacity\n- Issue: Whether a trust distribution to the husband's mother should be added back as notional property\n- Issue: Whether the wife's new relationship affects s 75(2) assessment and/or gives rise to an adverse inference (Jones v Dunkel)\n- Issue: Whether spousal maintenance should be ordered",
  "property_division": "- Asset Pool: Net pool approximately $2.9 million (including superannuation and notional property/add-backs).\n- Asset Pool: Existing assets about $2.36 million including: sale proceeds of former matrimonial home held in trust ($145,630, joint); husband's superannuation (~$236,000); wife's superannuation (~$45,000); husband's business/investment interests via entities and trusts (~$1.6 million); boat and boat licence in wife's name (~$295,000); minor bank accounts and personal property.\n- Asset Pool: Liability: wife's credit card about $5,500.\n- Asset Pool: Excluded: loans from parents used to fund legal fees (excluded from pool).\n- Asset Pool: Add-backs/notional property about $545,000: wife's interim distributions (~$370,000) added back by agreement; husband's interim distributions (~$110,000) added back by agreement; disputed trust distribution to husband's mother (~$65,000) sought to be added back.\n- Contributions: Initial contributions: husband brought substantial property/business/trust interests and superannuation valued about $1.7–1.8 million; wife brought negligible assets and minimal superannuation.\n- Contributions: During relationship: husband was primary income earner and operated/grew business interests; wife performed primary homemaker and parenting role and largely ceased paid employment around the birth of the first child.\n- Contributions: Some non-financial contributions attributed to husband via home improvements are noted.\n- Contributions: Post-separation: husband continued earning and retained control of business/trust structures; wife relied on interim distributions and support payments, consistent with limited immediate earning capacity.\n- Future Needs: Husband: late 40s, good health, ongoing income generation through controlled businesses/entities; asserted capacity to influence distributions and derive indirect financial benefits.\n- Future Needs: Wife: mid-30s, good health, several years out of workforce, retraining through tertiary study; medium-term projected income around $60,000 per annum.\n- Future Needs: Short- to medium-term income disparity favours an adjustment to wife.\n- Future Needs: Equal shared care reduces (but does not necessarily eliminate) parenting-related earning constraints and household cost impacts.\n- Future Needs: Wife's new relationship is raised; lack of evidence may support adverse inference as to financial support or living arrangements (Jones v Dunkel), potentially reducing weight of her claimed need depending on what was not disclosed.\n- Just & Equitable: The parties agree it is just and equitable to alter existing property interests.\n- Just & Equitable: Key determinants are likely the weight given to the husband's large initial contributions in a relatively short relationship, balanced against the wife's substantial homemaker/parenting contributions and any s 75(2) adjustment for income disparity and retraining.\n- Living Arrangements: Children live in an equal shared care arrangement pursuant to consent parenting resolution.\n- Living Arrangements: Wife has commenced a new relationship but details of living arrangements are not evidenced in the text.\n- Existing Agreements: Parenting proceedings resolved by consent (equal shared care).\n- Existing Agreements: Add-back of interim distributions to both parties is by agreement.\n- Existing Agreements: No binding financial agreement is described.",
  "spousal_maintenance": "- Need: Wife has been out of the workforce for several years, is undertaking tertiary retraining, and has short- to medium-term income disparity.\n- Need: Wife relied on interim distributions, government benefits, child support and existing spousal maintenance orders post-separation (amounts and duration not stated).\n- Capacity to Pay: Husband continues to earn income through businesses and entities he controls and is asserted to have capacity to manipulate distributions and obtain indirect financial benefits.\n- Statutory Factors: s 75(2) factors are raised in relation to income disparity, earning capacity, and the wife's unclear new relationship circumstances (Jones v Dunkel inference).\n- Income & Expenses: Specific incomes, expenses, and budgets are not stated.\n- Income & Expenses: Husband pays child support per assessment.\n- Earning Capacity: Wife's earning capacity is reduced short-term due to absence from workforce and ongoing study; anticipated graduate income around $60,000 in the medium term.\n- Earning Capacity: Husband has ongoing business-derived earning capacity (precise level not stated).\n- Health & Care: Both parties are in good health.\n- Relationship Length: Approximately six and a half years.\n- Standard of Living: Not stated.",
  "children_parenting": "- Child Ages: Two children aged under 10.\n- Current Arrangements: Equal shared care arrangement by consent.\n- Caregiver History: During the relationship, the wife was primarily responsible for parenting and homemaking.\n- Availability: Not stated beyond the agreed equal shared care arrangement.\n- Best Interests: Parenting issues are stated to be resolved by consent; no best-interests analysis is provided in the text.\n- Orders: Parenting orders/arrangements by consent providing for equal shared care (details not stated).",
  "impact_analysis": "- Pivotal Finding: Whether the $65,000 trust distribution to the husband's mother is treated as a dissipation/add-back (and if so, on what basis) or excluded.\n- Pivotal Finding: The weight given to the husband's initial contributions of approximately $1.7–1.8 million relative to the size of the pool (~$2.9 million) and the 6.5-year duration.\n- Pivotal Finding: The extent to which the wife's homemaker and parenting contributions in a traditional role arrangement are assessed as equal in weight to the husband's income-earning contributions during the relationship.\n- Pivotal Finding: The practical significance of the equal shared care arrangement for future-needs (capacity to work, time constraints, and household costs).\n- Pivotal Finding: Whether the court draws an adverse inference from the wife's failure to adduce evidence about her new relationship (Jones v Dunkel) and how that affects s 75(2) adjustment.\n- Pivotal Finding: How the court accounts for the husband's asserted ability to control/stream income through entities and obtain indirect benefits when assessing future needs and capacity.\n- Statutory Pivot: Family Law Act 1975 (Cth) s 79 (alteration of property interests; just and equitable requirement; consideration of contributions and other factors).\n- Statutory Pivot: Family Law Act 1975 (Cth) s 79(4) (contributions: financial, non-financial, homemaker/parenting; effect of proposed order).\n- Statutory Pivot: Family Law Act 1975 (Cth) s 75(2) (future-needs factors applied via s 79(4)(e)).\n- Statutory Pivot: Principle concerning adverse inference from failure to call evidence: Jones v Dunkel (relevance to undisclosed circumstances such as a new relationship).",
  "uncertainties": "- Uncertainties: No actual reasons for judgment or orders are provided; the outcome is not determined in the text.\n- Uncertainties: Valuation methodology and evidentiary basis for the husband's business/trust/company interests (~$1.6 million) is not stated (e.g., expert valuation, financial statements, control, or liquidity).\n- Uncertainties: The legal and factual basis for adding back the $65,000 distribution to the husband's mother is not provided (e.g., intention, timing, necessity, or whether it was a bona fide debt repayment).\n- Uncertainties: The nature of the trust holding the former matrimonial home sale proceeds and any trust-related property/entitlement issues are not detailed."
}

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