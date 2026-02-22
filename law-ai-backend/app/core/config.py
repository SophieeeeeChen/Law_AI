import os

def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Config:
    ENV = os.environ.get("ENV", "dev")
    DATABASE_URL = os.environ.get("DATABASE_URL")
    AUTH_MODE = os.environ.get("AUTH_MODE", "dev")
    DEV_DEFAULT_USER_ID = os.environ.get("DEV_DEFAULT_USER_ID") if ENV == "dev" else None
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS")
    CORS_ORIGINS_LIST = _parse_csv(CORS_ORIGINS)
    ENTRA_TENANT_ID = os.environ.get("ENTRA_TENANT_ID")
    ENTRA_CLIENT_ID = os.environ.get("ENTRA_CLIENT_ID")
    ENTRA_AUDIENCE = os.environ.get("ENTRA_AUDIENCE")
    ENTRA_ISSUER = os.environ.get("ENTRA_ISSUER")
    ENTRA_JWKS_URL = os.environ.get("ENTRA_JWKS_URL")
    # --- OpenAI Models ---
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")  # ChatGPT model
    OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # Embeddings model
    
    # --- Summary Generation ---
    # For large, pre-processed AustLII files
    AUSTLII_SUMMARY_TARGET_WORDS = int(os.environ.get("AUSTLII_SUMMARY_TARGET_WORDS", "1000"))
    AUSTLII_SUMMARY_MAX_WORDS = int(os.environ.get("AUSTLII_SUMMARY_MAX_WORDS", "1200"))
    # For user-uploaded files (will be dynamically adjusted)
    USER_SUMMARY_TARGET_WORDS = int(os.environ.get("USER_SUMMARY_TARGET_WORDS", "1000"))
    USER_SUMMARY_MAX_WORDS = int(os.environ.get("USER_SUMMARY_MAX_WORDS", "1200"))
    
    # --- Chunking Parameters ---
    CASE_CHUNK_SIZE = int(os.environ.get("CASE_CHUNK_SIZE", "1000"))
    CASE_CHUNK_OVERLAP = int(os.environ.get("CASE_CHUNK_OVERLAP", "200"))
    STATUTE_CHUNK_SIZE = int(os.environ.get("STATUTE_CHUNK_SIZE", "800"))
    STATUTE_CHUNK_OVERLAP = int(os.environ.get("STATUTE_CHUNK_OVERLAP", "120"))
    
    # --- Data & Storage ---
    if ENV == "dev":
        DATA_DIR = "./AustLII_cases_txt_tmp"
        VECTOR_DB_DIR = "./chroma_db"
        if DATABASE_URL is None:
            DATABASE_URL = "sqlite:///./app.db"
    else:  # prd
        DATA_DIR = "/mnt/data/AustLII_cases_txt"
        VECTOR_DB_DIR = "/mnt/data/chroma_db"
        if DATABASE_URL is None:
            raise RuntimeError("DATABASE_URL must be set when ENV=prd")
        if not CORS_ORIGINS_LIST:
            raise RuntimeError("CORS_ORIGINS must be set when ENV=prd")

    CASES_COLLECTION_NAME = os.environ.get("CASES_COLLECTION_NAME", "cases_full")
    STATUTES_COLLECTION_NAME = os.environ.get("STATUTES_COLLECTION_NAME", "rules_statutes")
    SUMMARY_COLLECTION_NAME = os.environ.get("SUMMARY_COLLECTION_NAME", "cases_summary")

    # --- Retrieval ---
    TOP_K = 5               # how many docs to retrieve initially
    MIN_RERANK_SCORE = 0.5   # threshold for filtering nodes
    HISTORY_MAX_TURNS = int(os.environ.get("HISTORY_MAX_TURNS", "6"))
    HYBRID_USE_RERANK = os.environ.get("HYBRID_USE_RERANK", "false").lower() == "true"
    RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "8"))
    BM25_TOP_K = int(os.environ.get("BM25_TOP_K", str(TOP_K)))
    HYBRID_VECTOR_WEIGHT = float(os.environ.get("HYBRID_VECTOR_WEIGHT", "0.6"))
    HYBRID_BM25_WEIGHT = float(os.environ.get("HYBRID_BM25_WEIGHT", "0.4"))

    # --- Prompt Template ---
    #this one is just for answer question without uploaded case
    QA_TEMPLATE = (
        "<|im_start|>system\n"
        "You are an Australian case law assistant.\n"
        "Follow these rules strictly:\n"
        "1) You MUST answer using ONLY the provided case law context snippets.\n"
        "2) Do NOT use any external knowledge or assumptions.\n"
        "3) If the context does NOT contain enough information to answer the question, "
        "clearly state that the answer cannot be determined from the provided materials.\n"
        "4) Base your reasoning explicitly on the context.\n"
        "5) Do NOT provide legal advice. This is for informational purposes only.\n\n"
        "Case law context snippets (total={context_count}):\n"
        "{context_str}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Question: {query_str}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
