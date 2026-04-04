import argparse
from datetime import datetime
import time
import json
import logging
import os
import re
import sys
import queue
import threading
from pathlib import Path
from typing import Iterable, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# NOW import from app (after path is set up)
from app.core.dev_logger import format_and_log


# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from bs4 import BeautifulSoup
import trafilatura
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import Config
from app.services.summary_service import (
    SUMMARY_LIST_LIMITS_FALLBACK,
    SUMMARY_LIST_LIMITS_PRIMARY,
    generate_summary_dict,
    summary_json_to_sections,
)


# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
CASES_TXT_DIR = str((Path(__file__).resolve().parent.parent / "AustLII_cases_md_famca_tree").resolve())
STATUTES_JSONL = str((Path(__file__).resolve().parent.parent / "Austlii_law_statutes/family_law_act_1975_sections.jsonl").resolve())

# Single ChromaDB instance
DB_DIR = str((Path(__file__).resolve().parent.parent / "chroma_db").resolve())

# Log file
LOG_PATH = str((Path(__file__).resolve().parent.parent / "logs" / "build_embeddings.log").resolve())
SCRIPT_LOG_DIR = str((Path(__file__).resolve().parent.parent / "logs").resolve())

# Collection names gpt
# CASES_COLLECTION_gpt5 = "cases_full"
# SUMMARY_COLLECTION_gpt5 = "cases_summary"
# STATUTES_COLLECTION_gpt5 = "rules_statutes"

#Collection names gemini
CASES_COLLECTION_GEMINI = "cases_full_gemini"
SUMMARY_COLLECTION_GEMINI = "cases_summary_gemini"
STATUTES_COLLECTION_GEMINI = "rules_statutes_gemini"

SUMMARY_GEMINI_JSONL = str((Path(__file__).resolve().parent.parent / "out_summaries" / "case_summaries_gemini_v3.jsonl").resolve())
SUMMARY_JSONL = str((Path(__file__).resolve().parent.parent / "out_summaries" / "case_summaries_gemini_v3.jsonl").resolve())

# Summary sizing controls - from Config
SUMMARY_INDEX_BATCH_SIZE = 30

"""Summary generation helpers are centralized in summary_pipeline.py.

This file focuses on IO (reading cases/statutes) and indexing into Chroma.
"""


def setup_logging() -> None:
    ensure_dir(os.path.dirname(LOG_PATH))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger("build_embeddings")

#logging functions for failed cases summaries&embeddings
def log_fullcase_embeddings_failure(path_stem: str, error_message: str):
    """Appends the failed filename and error to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(SCRIPT_LOG_DIR, "failed_ingestion_fullcaseembeddings.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] FILE: {path_stem} | ERROR: {error_message}\n")
        
def log_summaries_failure(path_stem: str, error_message: str):
    """Appends the failed filename and error to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(SCRIPT_LOG_DIR, "failed_case_summaries.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] FILE: {path_stem} | ERROR: {error_message}\n")

def log_summaries_embedding_failure(path_stem: str, section_name: str, error_message: str):
    """Appends the failed filename and error to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(SCRIPT_LOG_DIR, "failed_ingestion_summariesembedding.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] FILE: {path_stem} |{section_name}| ERROR: {error_message}\n")

def load_cases_documents(folder: str, batch_size: int = 20, existing_fullcase_chunk: set = None) -> Iterable[List[Document]]:
    # We use a Markdown-specific splitter to keep headers and paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,  
        chunk_overlap=300, # Small overlap to maintain context between chunks
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
    )
    current_batch = []
    existing_fullcase_chunk = existing_fullcase_chunk or set()
    for path in Path(folder).rglob("*.md"):
        print(f"Checking file: {path.name}")
        if path.stem not in existing_fullcase_chunk:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                # --- 3. Semantic Chunking ---
                # This turns one 5,000-word file into multiple searchable Document objects
                chunks = text_splitter.split_text(text)
                
                for i, chunk_content in enumerate(chunks):
                    current_batch.append(Document(
                            text=f"SOURCE CASE: {path.stem}\n---\n{chunk_content}",
                            metadata={"case_name": path.stem, "chunk_index": i}
                        ))
                    
                if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
            except Exception as e:
                raise e
    if current_batch:
        yield current_batch

def get_existing_case_sections_inCollection(chroma_collection) -> set[tuple[str, str]]:
    """
    Retrieves all unique (case_name, summary_section) pairs from the ChromaDB collection.
    This allows us to skip only sections that are already embedded, not entire cases.
    """
    existing_pairs = set()
    
    count = chroma_collection.count()
    if count == 0:
        return existing_pairs

    print(f"Fetching metadata for {count} chunks to identify existing case sections...")
    
    batch_size = 5000 
    for offset in range(0, count, batch_size):
        results = chroma_collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset
        )
        
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                case_name = meta.get("case_name")
                section_name = meta.get("summary_section")
                if case_name and section_name:
                    existing_pairs.add((case_name, section_name))
                    
    print(f"Found {len(existing_pairs)} unique (case, section) pairs already in the index.")
    return existing_pairs


def get_existing_case_names_inCollection(chroma_collection) -> set:
    """
    Retrieves all unique case_name values from the ChromaDB collection.
    """
    existing_names = set()
    
    # We use .get() but only include 'metadatas' to keep the response lightweight
    # By default, Chroma has a limit, so we check if we need to paginate
    count = chroma_collection.count()
    if count == 0:
        return existing_names

    print(f"Fetching metadata for {count} chunks to identify existing cases...")
    
    # Fetch in batches if your collection is very large (e.g., > 10,000 chunks)
    batch_size = 5000 
    for offset in range(0, count, batch_size):
        results = chroma_collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset
        )
        
        # Extract the 'case_name' from each chunk's metadata
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if "case_name" in meta:
                    existing_names.add(meta["case_name"])
                    
    print(f"Found {len(existing_names)} unique cases already in the index.")
    return existing_names

def get_existing_case_ids_from_jsonl(jsonl_path: str) -> set:
    """
    Extract all case_id values from a JSONL file.
    
    Args:
        jsonl_path: Path to the case_summaries.jsonl file
        
    Returns:
        Set of case_id strings found in the JSONL file
    """
    existing_ids = set()
    jsonl_file = Path(jsonl_path)
    
    if not jsonl_file.exists():
        print(f"JSONL file not found: {jsonl_path}")
        return existing_ids
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'case_id' in data:
                        existing_ids.add(data['case_id'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
    
    return existing_ids

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_case_files(folder: str) -> List[Path]:
    txt_files = list(Path(folder).rglob("*.txt"))
    md_files = list(Path(folder).rglob("*.md"))
    return sorted(txt_files + md_files)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_existing_summary_ids(path: str) -> set[str]:
    """Read existing case_ids from the summaries JSONL to avoid re-processing."""
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                case_id = row.get("case_id")
                if case_id:
                    ids.add(str(case_id))
            except json.JSONDecodeError:
                continue
    return ids


def write_jsonl(path: str, rows: List[dict], *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_existing_case_sources(db_path: str, collection_name: str) -> set[str]:
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    total_count = collection.count()
    print(f"Collection '{collection_name}' has {total_count} total entries.")
    existing_sources: set[str] = set()
    
    if total_count == 0:
        return existing_sources

    # FIX: Explicitly set the limit to the total count of the collection
    results = collection.get(
        include=["metadatas"],
        limit=total_count  # This ensures you don't stop at 100
    )
    
    existing_sources = {
        str(meta.get("case_name")) 
        for meta in results.get("metadatas", []) 
        if meta and meta.get("case_name")
    }
            
    print(f"Verified {len(existing_sources)} unique cases already in {collection_name}")
    return existing_sources

#embeddings creation
def create_embeddings_for_full_cases_gemini(folder: str) -> None:
    """
    Build the full case collection using Gemini Embedding 2 (3072-dim).
    Includes a skip-logic to avoid re-paying for existing embeddings.
    """
    # 1. Global Gemini Configuration
    # Using 'preview' for the latest v2 high-accuracy model
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=f"models/{Config.GEMINI_EMBED_MODEL}",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        output_dimensionality=3072  # Explicitly set for maximum legal accuracy
    )
    
    db = chromadb.PersistentClient(path=DB_DIR)

    # 2. Setup/Connect to the Gemini Collection
    chroma_collection = db.get_or_create_collection(
        name=CASES_COLLECTION_GEMINI, 
        metadata={"hnsw:space": "cosine"}
    )
    embedded_case_names = get_existing_case_names_inCollection(chroma_collection)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    full_storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Initialize the Index
    full_index = VectorStoreIndex(
        nodes=[], 
        storage_context=full_storage_context
    )

    # 4. Processing Loop
    for batch_docs in load_cases_documents(folder, batch_size=50, existing_fullcase_chunk=embedded_case_names):
        # We check the first document in the batch; if it's already indexed, 
        try:
            # This triggers the API call to Gemini
            full_index.insert_nodes(batch_docs)
        except Exception as e:
            failed_stems = list(set([doc.metadata.get("case_name", "Unknown") for doc in batch_docs]))
            error_msg = f"Gemini API/DB Error: {str(e)}"
            # Log failure for manual review
            log_fullcase_embeddings_failure(str(failed_stems), error_msg)

    # 5. Final Persist
    full_index.storage_context.persist(persist_dir=DB_DIR)
    print(f"Index complete. Total cases in {CASES_COLLECTION_GEMINI}: {chroma_collection.count()}")

# This is to create the case summaries embeddings based on json file
def build_summary_embeddings_from_jsonl(jsonl_path: str) -> None:
    """
    Stream-read case summaries from JSONL and embed directly — no bulk loading.
    """
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"JSONL file not found: {jsonl_path}")
        return

    # 1. Initialize Gemini Embedding Model
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=f"models/{Config.GEMINI_EMBED_MODEL}",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        output_dimensionality=3072,
    )

    # 2. Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name=SUMMARY_COLLECTION_GEMINI,
        metadata={"hnsw:space": "cosine"},
    )

    # 3. Get existing (case_name, section_name) pairs — section-level skip logic
    existing_sections = get_existing_case_sections_inCollection(collection)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = None
    batch: List[Document] = []
    total_docs = 0
    skipped = 0

    # 4. Stream directly from file — no rows list in memory
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log_summaries_embedding_failure(f"line_{line_num}", "json_parse", str(e))
                continue

            case_id = row.get("case_id", "unknown")

            # Yield sections directly into the batch, skipping already-embedded sections
            summary_sections = row.get("summary_sections") or {}
            for section_name, section_text in summary_sections.items():
                if not section_text or not isinstance(section_text, str):
                    continue

                # Skip this specific (case, section) if already embedded
                if (case_id, section_name) in existing_sections:
                    skipped += 1
                    continue

                batch.append(Document(
                    text=section_text,
                    metadata={
                        "case_name": case_id,
                        "summary_section": section_name,
                    },
                ))

            # Flush batch when full
            if len(batch) >= SUMMARY_INDEX_BATCH_SIZE:
                try:
                    if index is None:
                        index = VectorStoreIndex.from_documents(
                            batch, storage_context=storage_context, show_progress=True,
                        )
                    else:
                        index.insert_nodes(batch)
                    total_docs += len(batch)
                    print(f"  Embedded {total_docs} sections so far...")
                except Exception as e:
                    for failed_doc in batch:
                        log_summaries_embedding_failure(
                            failed_doc.metadata.get("case_name", "unknown"),
                            failed_doc.metadata.get("summary_section", "unknown"),
                            str(e),
                        )
                batch = []

    # Handle remaining
    if batch:
        try:
            if index is None:
                index = VectorStoreIndex.from_documents(batch, storage_context=storage_context)
            else:
                index.insert_nodes(batch)
            total_docs += len(batch)
        except Exception as e:
            for failed_doc in batch:
                log_summaries_embedding_failure(
                    failed_doc.metadata.get("case_name", "unknown"),
                    failed_doc.metadata.get("summary_section", "unknown"),
                    str(e),
                )

    if index is None:
        print("No new summaries to index.")
        return

    index.storage_context.persist(persist_dir=DB_DIR)
    print(f"SUCCESS: Embedded {total_docs} sections (skipped {skipped} existing cases)")

def build_statutes_embeddings_gemini(jsonl_path: str = STATUTES_JSONL) -> None:
    """
    Build the rules_statutes_gemini collection from a statutes JSONL file.
    """
    # 1. Initialize Gemini Embedding Model
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=f"models/{Config.GEMINI_EMBED_MODEL}",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        output_dimensionality=3072,
    )

    # 2. Load all statute documents
    docs = load_statutes_documents(jsonl_path)
    if not docs:
        print("No statute documents found.")
        return

    print(f"Loaded {len(docs)} statute sections from {jsonl_path}")

    # 3. Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name=STATUTES_COLLECTION_GEMINI,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} entries. Skipping rebuild.")
        print("Run reset_collection() first if you want to rebuild.")
        return

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Batch-insert into index
    index = None
    batch_size = 50
    total = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        try:
            if index is None:
                index = VectorStoreIndex.from_documents(
                    batch,
                    storage_context=storage_context,
                    show_progress=True,
                )
            else:
                index.insert_nodes(batch)
            total += len(batch)
            print(f"  Embedded {total}/{len(docs)} statute sections...")
        except Exception as e:
            failed_ids = [d.metadata.get("section_id", "?") for d in batch]
            logger.error(f"Failed to embed statute batch ({failed_ids}): {e}")
            time.sleep(2)

    if index is None:
        print("No statutes indexed.")
        return

    # 5. Persist
    index.storage_context.persist(persist_dir=DB_DIR)
    print(f"SUCCESS: Indexed {collection.count()} statute sections into {STATUTES_COLLECTION_GEMINI}")

def load_statutes_documents(jsonl_path: str) -> List[Document]:
    docs: List[Document] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    text=row["text"],
                    metadata={
                        "source_type": row.get("source_type", "statute"),
                        "act": row.get("act"),
                        "section_id": row.get("section_id"),
                        "section_title": row.get("section_title"),
                        "part": row.get("part"),
                        "division": row.get("division"),
                        "subdivision": row.get("subdivision"),
                        "chunk_id": row.get("chunk_id")
                    },
                )
            )
    return docs

def _get_summary_jsonl_for_dir(cases_dir: str, override_jsonl: str = None) -> str:
    """
    Derive the JSONL output path from the input directory name.
    e.g. .../FamCA_2018 → out_summaries/case_summaries_gemini_v3_2018.jsonl
         .../FamCA_2020 → out_summaries/case_summaries_gemini_v3_2020.jsonl
    Falls back to SUMMARY_GEMINI_JSONL if no year is found.
    """
    if override_jsonl:
        return override_jsonl

    import re
    folder_name = Path(cases_dir).name  # e.g. "FamCA_2018"
    match = re.search(r"(\d{4})", folder_name)
    if match:
        year = match.group(1)
        return str((Path(__file__).resolve().parent.parent / "out_summaries" / f"case_summaries_gemini_v3_{year}.jsonl").resolve())
    return SUMMARY_GEMINI_JSONL

def run_case_summaries_only(cases_dir: str = CASES_TXT_DIR, override_jsonl: str = None) -> None:
    """
    Step 1 ONLY: Generate summaries → write to JSONL.
    Fully resumable — skips cases already in the JSONL file.
    Does NOT embed. Run embed_summaries separately after this completes.
    """
    jsonl_path = _get_summary_jsonl_for_dir(cases_dir, override_jsonl)
    ensure_dir(os.path.dirname(jsonl_path))
    print(f"Output JSONL: {jsonl_path}")
    llm = GoogleGenAI(
            model=f"models/{Config.GEMINI_MODEL}",
            api_key=os.environ.get("GOOGLE_API_KEY"),
            
            model_kwargs={
                "thinking_config": {"thinking_level": "medium"},
                "response_mime_type": "application/json", # FORCES structured output
                "top_p": 0.2,                             # Relaxed slightly to allow for detailed "Entities"
                "max_output_tokens": 8192,                # Room for 1,200+ word summaries
            },
            
            temperature=0.2,      # Low enough for JSON logic, high enough for nuanced legal writing
            timeout=300.0,        # Thinking mode adds latency; 5 mins is the new safe standard
            max_retries=12,
            transport="rest"
        )

    # Fallback LLM for cases where Gemini refuses (no candidates / content filtering)
    fallback_llm = OpenAI(
        model=Config.OPENAI_MODEL,
        temperature=0.1,
        timeout=300.0,
        max_retries=2,
    )

    # Skip cases already summarized in JSONL
    existing_summary_ids = get_existing_case_ids_from_jsonl(jsonl_path)
    print(f"Found {len(existing_summary_ids)} existing summaries in JSONL. Will skip those.")

    processed = 0
    failed = 0
    skipped = 0

    for path in list_case_files(cases_dir):
        if path.stem in existing_summary_ids:
            skipped += 1
            continue
        try:
            text = read_text(path)
            if not text.strip():
                logger.warning("Empty case file skipped: %s", path)
                continue

            print(f"[{processed + failed + skipped + 1}] Processing {path.stem}...")
            summary = generate_summary_dict(
                text,
                path.stem,
                list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
                llm=llm,
                case_name=path.stem,
            )
            if Config.ENV == "dev":
                format_and_log(
                    "/create_summary",
                    action="generate_summary_dict using prompt",
                    data_name="create_summary",
                    data_content=summary,
                )
            if summary:
                summary_sections = summary_json_to_sections(summary)
                row = {
                    "case_id": path.stem,
                    "summary_sections": summary_sections,
                }
                # Append immediately — so if we crash, progress is saved
                write_jsonl(jsonl_path, [row], append=True)
                processed += 1
                print(f"[OK] {path.stem}")
            else:
                logger.warning("Empty summary returned for %s", path.stem)
                failed += 1

            time.sleep(10)  # Rate limit protection
        except Exception as e:
            error_msg = str(e).lower()
            # Retry with OpenAI fallback for Gemini-specific failures
            if "no candidates" in error_msg or "safety" in error_msg or "blocked" in error_msg or "google" in error_msg or "gemini" in error_msg or "max_tokens" in error_msg:
                logger.warning("Gemini failed for %s (%s). Falling back to OpenAI model: %s", path.stem, e, Config.OPENAI_MODEL)
                try:
                    summary = generate_summary_dict(
                        text,
                        path.stem,
                        list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
                        llm=fallback_llm,
                        case_name=path.stem,
                    )
                    if summary:
                        summary_sections = summary_json_to_sections(summary)
                        row = {
                            "case_id": path.stem,
                            "summary_sections": summary_sections,
                        }
                        write_jsonl(jsonl_path, [row], append=True)
                        processed += 1
                        print(f"[OK-FALLBACK] {path.stem} (via {Config.OPENAI_MODEL})")
                        time.sleep(10)
                        continue
                except Exception as fallback_err:
                    logger.error("Fallback also failed for %s: %s", path.stem, str(fallback_err))

            logger.error("Failed to process %s: %s", path.stem, error_msg)
            log_summaries_failure(path.stem, str(e))
            failed += 1
            time.sleep(15)

    print(f"\n{'='*60}")
    print(f"  SUMMARY GENERATION COMPLETE")
    print(f"  Processed: {processed} | Failed: {failed} | Skipped: {skipped}")
    print(f"{'='*60}")
    print(f"\nNext step: run --action embed_summaries to embed into ChromaDB")

#to delete certain collections
def reset_collection(db_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=db_path)
    try:
        # This deletes the collection and all its embeddings/metadata
        client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except ValueError:
        print(f"Collection {collection_name} did not exist. Nothing to delete.")

if __name__ == "__main__":
    ensure_dir(SCRIPT_LOG_DIR)
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="SophieAI batch processing CLI")
    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "summaries",
            "embed_summaries",
            "embed_full_cases",
            "embed_statutes",
            "reset_collection",
        ],
        help="Which function to run",
    )
    parser.add_argument("--input-dir", default=None, help="Override input cases directory")
    parser.add_argument("--jsonl", default=None, help="JSONL file path for embedding")
    parser.add_argument("--collection", default=None, help="Collection name (for reset_collection)")

    args = parser.parse_args()

    input_folder = args.input_dir or os.environ.get(
        "CASES_INPUT_DIR",
        CASES_TXT_DIR,
    )

    if args.action == "summaries":
        # Initialize Gemini LLM for summary generation (1M token context)
        run_case_summaries_only(input_folder, override_jsonl=args.jsonl)

    elif args.action == "embed_summaries":
        # Step 2 ONLY: embed from JSONL → ChromaDB (resumable via existing case names)
        jsonl = args.jsonl or SUMMARY_GEMINI_JSONL
        build_summary_embeddings_from_jsonl(jsonl)

    elif args.action == "embed_full_cases":
        create_embeddings_for_full_cases_gemini(input_folder)

    elif args.action == "embed_statutes":
        build_statutes_embeddings_gemini(STATUTES_JSONL)

    elif args.action == "reset_collection":
        coll = args.collection or SUMMARY_COLLECTION_GEMINI
        reset_collection(DB_DIR, coll)

