# AustLII + Uploaded Case Processing

This document explains how AustLII case files and uploaded case files are processed, indexed, and used for retrieval and prompting, plus the key data formats used in the pipeline.

## 1) AustLII case files

### 1.1 Source
- Input: Plain text files under the AustLII case folder (for example: AustLII_cases_txt).
- Each file represents one decided case.

### 1.2 Summary generation
For each case file, the pipeline generates a structured summary:
- `generate_summary_dict(...)` produces a JSON object with structured sections.
- `summary_json_to_text(...)` produces a flattened summary text (bulleted list).
- `summary_json_to_sections(...)` produces a list of section blocks used for embeddings.

### 1.3 Summary sections and embeddings
Each summary section becomes its own `Document` and embedding:
- Stored in the summary vector collection: `cases_summary`.
- Each section is embedded separately to enable targeted retrieval by topic.

### 1.4 Summary collection structure
For each summary section, the vector store holds:
- `text`: the section text (what gets embedded)
- `metadata`:
  - `source_type`: "case_summary"
  - `case_id`: case identifier (usually filename stem)
  - `case_name`: same as case_id
  - `source_file`: full path to original AustLII file
  - `summary_section`: section name (e.g., property_division, children_parenting)
  
Case: [2008] FamCAFC 142
├── Document 1: "overview" section
│   ├── text: "- Fact: Party A and B married in 2000...\n- Issue: Division of $800k property..."
│   └── metadata:
│       ├── source_type: "case_summary"
│       ├── case_id: "[2008] FamCAFC 142"
│       ├── case_name: "[2008] FamCAFC 142"
│       ├── source_file: "C:/path/to/[2008] FamCAFC 142.txt"
│       └── summary_section: "overview"
│
├── Document 2: "property_division" section
│   ├── text: "- Asset pool: House $800k...\n- Contributions: Wife primary carer..."
│   └── metadata:
│       └── summary_section: "property_division"
│
├── Document 3: "spousal_maintenance" section
│   ├── text: "- Need: Wife unemployed...\n- Capacity to pay: Husband earns $120k..."
│   └── metadata:
│       └── summary_section: "spousal_maintenance"
│
├── Document 4: "children_parenting" section
│   ├── text: "- Child ages: 8, 5\n- Current arrangements: Mother primary carer..."
│   └── metadata:
│       └── summary_section: "children_parenting"
│
└── Document 5: "family_violence_safety" section
    ├── text: "- Incidents: 3 police callouts...\n- Protection orders: IVO in place..."
    └── metadata:
        └── summary_section: "family_violence"
        
### 1.5 Full case embeddings
In addition to summary sections, the full case text is chunked and embedded:
- Stored in the cases vector collection: `cases_full`.
- Chunking uses `SentenceSplitter` for manageable passage retrieval.

## 2) Uploaded case files

### 2.1 Source
- Input: user-uploaded file via the API endpoint.
- File size is validated, then decoded to text.

### 2.2 Summary generation
For each upload, the same summary pipeline is used:
- `generate_summary_dict(...)` produces structured JSON.
- `summary_json_to_text(...)` and `summary_json_to_sections(...)` are created.
- The JSON string is stored in SQL (the `Case` table) for persistence.

### 2.3 Uploaded-case embeddings (in memory)
Each uploaded summary section becomes a `Document` and is embedded into an in-memory vector index:
- Purpose: allow retrieval of “uploaded case snippets” (facts) alongside AustLII precedent retrieval.
- The in-memory index is reused for the same `case_id` in the current process.

### 2.4 Uploaded-case document metadata
For each section inserted into the in-memory index:
- `text`: section text (embedded)
- `metadata`:
  - `source_type`: "uploaded_case"
  - `case_id`: database case ID
  - `source`: original filename
  - `summary_section`: section name (e.g., property_division)

## 3) Retrieval + prompting flow

### 3.1 Similar case retrieval (AustLII)
- The system embeds the user’s question and retrieves nearest vectors from:
  - `cases_summary` (summary sections)
  - `cases_full` (full text chunks)
  - `rules_statutes` (statute chunks)

### 3.2 Uploaded case snippet retrieval
- If an uploaded case is available, the system also retrieves relevant snippets from the uploaded-case in-memory index.
- These are treated as facts, not authority.

### 3.3 Prompt composition
The final prompt includes:
- Uploaded case summary (facts)
- Uploaded case snippets (facts)
- Retrieved AustLII summaries/snippets (authority)
- Retrieved statutory materials (authority)

## 4) Data formats

### 4.1 Summary JSON (simplified)
The summary JSON produced by `generate_summary_dict(...)` has a structured shape with keys such as:
- `facts`, `issues`, `outcome_orders`, `reasons_rationale`, `uncertainties`
- `property`, `parenting`, `spousal_maintenance`, `family_violence_safety`, `prenup_postnup`
- `impact_analysis` (pivotal findings and statutory pivots)
- `parties`, `court`, `date`

Example (data for each part):
```json
{
  "facts": [
    "The parties separated in 2019 after a 12-year relationship.",
    "Two children were born in 2011 and 2014."
  ],
  "issues": [
    "Dispute over property pool valuation.",
    "Primary care and school choice for the children."
  ],
  "property": {
    "asset_pool": [
      "Former matrimonial home valued at $820,000 with a $300,000 mortgage.",
      "Respondent superannuation of $140,000."
    ],
    "contributions": [
      "Applicant made initial financial contribution of $120,000.",
      "Respondent provided primary homemaker contributions during 2014-2018."
    ],
    "future_needs": [
      "Applicant has ongoing health issues limiting full-time work."
    ],
    "just_equitable": [
      "Court found a 55/45 split just and equitable."
    ],
    "living_arrangements": [
      "Children reside primarily with the applicant in the former home."
    ],
    "existing_agreements": [
      "No binding financial agreement in place."
    ]
  },
  "spousal_maintenance": {
    "need": [
      "Applicant has weekly shortfall of $250."
    ],
    "capacity_to_pay": [
      "Respondent earns $120,000 per annum and can contribute."
    ],
    "statutory_factors": [
      "Section 75(2) factors favour short-term maintenance."
    ],
    "income_expenses": [
      "Applicant rent $450/week; respondent mortgage $520/week."
    ],
    "earning_capacity": [
      "Applicant can return to part-time work within 12 months."
    ],
    "health_care": [
      "Ongoing physiotherapy costs estimated at $80/week."
    ],
    "relationship_length": [
      "Relationship lasted approximately 12 years."
    ],
    "standard_of_living": [
      "Parties enjoyed a moderate middle-income standard."
    ]
  },
  "parenting": {
    "child_ages": [
      "Child A is 12; Child B is 9."
    ],
    "current_arrangements": [
      "Children live with applicant and spend alternate weekends with respondent."
    ],
    "caregiver_history": [
      "Applicant has been primary caregiver since separation."
    ],
    "availability": [
      "Respondent works shift hours and has limited weekday availability."
    ],
    "safety_concerns": [
      "No current safety concerns identified by the court."
    ],
    "child_views": [
      "Children expressed preference to remain at current school."
    ],
    "allegations": [
      "Respondent alleged communication difficulties; not substantiated."
    ],
    "expert_evidence": [
      "Family report recommended stability in primary care."
    ],
    "best_interests": [
      "Continuity of care and schooling favoured applicant."
    ],
    "orders": [
      "Shared parental responsibility; children live with applicant."
    ]
  },
  "family_violence_safety": {
    "incidents": [
      "One historical incident of verbal abuse in 2018."
    ],
    "protection_orders": [
      "No current protection orders."
    ],
    "police_court": [
      "No criminal proceedings on record."
    ],
    "child_exposure": [
      "Children were not present during the 2018 incident."
    ],
    "safety_plan": [
      "Handover via school to minimise conflict."
    ]
  },
  "prenup_postnup": {
    "agreement_date": [
      "No agreement executed."
    ],
    "legal_advice": [
      "Not applicable."
    ],
    "financial_disclosure": [
      "Not applicable."
    ],
    "pressure_duress": [
      "Not applicable."
    ],
    "changed_circumstances": [
      "Not applicable."
    ]
  },
  "outcome_orders": [
    "Property adjusted 55/45 in favour of applicant.",
    "Respondent to pay spousal maintenance for 12 months."
  ],
  "reasons_rationale": [
    "Court preferred applicant's evidence on contributions.",
    "Stability for children required primary residence with applicant."
  ],
  "impact_analysis": {
    "pivotal_findings": [
      "Applicant's initial contribution was significant."
    ],
    "statutory_pivots": [
      "Section 79(4) contributions were determinative."
    ]
  },
  "uncertainties": [
    "Exact valuation date of the property was disputed."
  ],
  "parties": [
    "Applicant (mother)",
    "Respondent (father)"
  ],
  "court": "Family Court of Australia",
  "date": "2023-08-14"
}
```

### 4.2 Summary sections list
`summary_json_to_sections(...)` returns a list like:
- `[{"section": "overview", "text": "..."}, {"section": "property_division", "text": "..."}, ...]`

Each item becomes a separate embedding.

### 4.3 Cached section examples (stored per section)
Below are examples of how each section looks when stored in cache (vector store or in-memory index).

**AustLII summary cache (`cases_summary`)**
```json
{
  "text": "- Fact: The parties separated in 2019 after a 12-year relationship.\n- Issue: Dispute over property pool valuation.",
  "metadata": {
    "source_type": "case_summary",
    "case_id": "[2013] FamCAFC 109",
    "case_name": "[2013] FamCAFC 109",
    "source_file": "AustLII_cases_txt/[2013] FamCAFC 109.txt",
    "summary_section": "overview"
  }
}
```

**Uploaded-case cache (in-memory, scoped by user)**
```json
case_summary_sections = {
  7: {
    101: {
      "overview": "Parties: John Doe and Jane Doe. Court: Federal Circuit and Family Court of Australia. Date: 2026-01-15.",
      "property_division": "- Asset pool: Family home in Richmond ($1.5M), Superannuation ($200k).\n- Contributions: John (Initial deposit), Jane (Primary caregiver for 10 years).\n- Future needs: Jane requires housing for child.",
      "impact_analysis": "- Pivotal Finding: The Richmond property is the sole significant asset.\n- Statutory Pivot: Section 79(4) (Contributions) and Section 75(2) (Future Needs).",
      "uncertainties": "- Gap: The current valuation of John's industry super fund is missing.\n- Gap: Jane's exact employment capacity post-separation is unclear.",
      "children_parenting": "- Child ages: 8 and 10.\n- Current arrangements: Week-about roster.",
      "family_violence_safety": "No incidents reported; no current protection orders."
    }
  }
}
```

**Example with embedding vector stored in cache**
```json
{
  "id": "case:42:property_division:0",
  "text": "- Asset pool: Former matrimonial home valued at $820,000 with a $300,000 mortgage.\n- Contributions: Applicant made initial financial contribution of $120,000.",
  "embedding": [0.0123, -0.0456, 0.0789, -0.0112, 0.0345],
  "metadata": {
    "source_type": "uploaded_case",
    "case_id": 42,
    "source": "user_upload_case.txt",
    "summary_section": "property_division"
  }
}
```
Note: `embedding` is typically a long float array (e.g., 768/1536 dimensions) and may be stored internally by the vector database rather than in JSON.

## 5) Where this is implemented
- Batch indexing for AustLII: build_embeddings.py
- Upload processing: main.py
- Retrieval logic: app/services/rag_service.py

