import argparse
import json
import logging
import os
import re
import sys
import queue
import threading
from pathlib import Path
from typing import Iterable, List

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from bs4 import BeautifulSoup
import trafilatura
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
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
CASES_TXT_DIR = "./AustLII_cases_txt_old"
STATUTES_JSONL = "./Austlii_law_statutes/family_law_act_1975_sections.jsonl"

# Single ChromaDB instance
DB_DIR = str((Path(__file__).resolve().parent.parent / "chroma_db").resolve())

# Log file
LOG_PATH = "./logs/build_embeddings_fortest.log"

# Collection names
CASES_COLLECTION = "cases_full"
STATUTES_COLLECTION = "rules_statutes"
SUMMARY_COLLECTION = "cases_summary"
SUMMARY_JSONL = "./out_summaries/case_summaries.jsonl"

# Summary sizing controls - from Config
SUMMARY_INDEX_BATCH_SIZE = 512

# HTML conversion defaults
HTML_INPUT_DIR = "./cases/FamCA"
HTML_MD_OUTPUT_DIR = "./AustLII_cases_md_famca"
HTML_MD_TREE_OUTPUT_DIR = "./AustLII_cases_md_famca_tree"

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


def init_embeddings():
    Settings.embed_model = OpenAIEmbedding(model=Config.OPENAI_EMBED_MODEL)


def austlii_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    article = soup.select_one("article.the-document")
    if not article:
        article = soup.body or soup

    for tag in article.select("script, style"):
        tag.decompose()

    def clean(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    lines = []

    for el in article.find_all(["p", "li"], recursive=True):
        if el.name == "p":
            cls = el.get("class") or []
            txt = clean(el.get_text(" ", strip=True))
            if not txt:
                continue

            if txt.lower().startswith("last updated:"):
                continue

            if "h1" in cls:
                lines.append(f"## {txt}")
            elif "h2" in cls:
                lines.append(f"### {txt}")
            else:
                lines.append(txt)

        elif el.name == "li":
            num = el.get("value")
            txt = clean(el.get_text(" ", strip=True))
            if not txt:
                continue

            if num is not None and str(num).isdigit():
                lines.append(f"[{num}] {txt}")
            else:
                lines.append(f"- {txt}")

    seen = set()
    deduped = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    text = "\n".join(deduped)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text

# --- add near the other conversion functions ---
def convert_html_tree_trafilatura(
    input_dir: str,
    output_dir: str,
    *,
    workers: int = 4,
    queue_size: int = 200,
) -> None:
    return convert_html_folder_tree(
        input_dir,
        output_dir,
        workers=workers,
        queue_size=queue_size,
    )

def extract_legal_catchwords(html_content: str) -> str:
    """
    Extracts the 'Catchwords' section commonly found in AustLII cases.
    Best-effort fallback if metadata misses it.
    """
    if "Catchwords:" in html_content:
        parts = html_content.split("Catchwords:", 1)
        catchwords = parts[1].split("<", 1)[0].strip()
        return catchwords
    return ""



def convert_html_folder_trafilatura(
    input_dir: str,
    output_dir: str,
    *,
    existing_dir: str | None = None,
    workers: int = 4,
    queue_size: int = 200,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    existing_path = Path(existing_dir) if existing_dir else None
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("HTML input folder does not exist: %s", input_path)
        return

    files = sorted(input_path.rglob("*.html"))
    logger.info("Found %s HTML files in %s", len(files), input_path)
    if not files:
        return

    prefix = input_path.name or "FamCA"
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    stop_token = object()

    def producer():
        for fp in files:
            q.put(fp)
        for _ in range(workers):
            q.put(stop_token)

    def consumer(worker_id: int):
        while True:
            item = q.get()
            if item is stop_token:
                q.task_done()
                break
            file_path: Path = item
            try:
                year = file_path.parent.name or "unknown"
                folder_name = f"{prefix}_{year}"
                out_dir = output_path / folder_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{folder_name}_{file_path.stem}.md"
                out_path = out_dir / out_name
                if existing_path:
                    existing_md = existing_path / out_name
                    if existing_md.exists():
                        logger.info("Skipped (already converted): %s", file_path)
                        continue
                if out_path.exists():
                    logger.info("Skipped (already converted): %s", file_path)
                    continue
                html_content = file_path.read_text(encoding="utf-8", errors="ignore")
                clean_text = trafilatura.extract(
                    html_content,
                    include_tables=True,
                    include_formatting=True,
                    output_format="markdown",
                )
                if not clean_text:
                    logger.warning("No content extracted from: %s", file_path)
                    continue

                metadata = trafilatura.extract_metadata(html_content)
                title = metadata.title if metadata and metadata.title else file_path.stem
                date = metadata.date if metadata and metadata.date else "Unknown"
                catchwords = extract_legal_catchwords(html_content)

                header_lines = [f"# {title}"]
                header_lines.append(f"**Date:** {date}")
                if catchwords:
                    header_lines.append(f"**Catchwords:** {catchwords}")
                header_lines.append("")

                markdown = "\n".join(header_lines) + clean_text.strip() + "\n"
                out_path.write_text(markdown, encoding="utf-8")
                logger.info("Converted to markdown: %s", file_path)
            except Exception:
                logger.exception("Failed to convert HTML file: %s", file_path)
            finally:
                q.task_done()

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    consumers = [threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(workers)]
    for c in consumers:
        c.start()

    q.join()
    prod.join()
    for c in consumers:
        c.join()

def convert_html_folder(
    input_dir: str,
    output_dir: str,
    existing_dir: str | None = None,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    existing_path = Path(existing_dir) if existing_dir else None

    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("HTML input folder does not exist: %s", input_path)
        return

    html_files = sorted(input_path.glob("*.html"))
    logger.info("Found %s HTML files in %s", len(html_files), input_path)

    for file_path in html_files:
        try:
            existing_md = existing_path / f"{file_path.stem}.md" if existing_path else None
            new_md = output_path / f"{file_path.stem}.md"
            if (existing_md and existing_md.exists()) or new_md.exists():
                logger.info("Skipped (already converted): %s", file_path.name)
                continue

            html_content = file_path.read_text(encoding="utf-8", errors="ignore")
            clean_text = trafilatura.extract(
                html_content,
                include_tables=True,
                include_formatting=True,
                output_format="markdown",
            )
            if not clean_text:
                logger.warning("No content extracted from: %s", file_path.name)
                continue

            metadata = trafilatura.extract_metadata(html_content)
            title = metadata.title if metadata and metadata.title else file_path.stem
            date = metadata.date if metadata and metadata.date else "Unknown"
            catchwords = extract_legal_catchwords(html_content)

            header_lines = [f"# {title}"]
            header_lines.append(f"**Date:** {date}")
            if catchwords:
                header_lines.append(f"**Catchwords:** {catchwords}")
            header_lines.append("")

            markdown = "\n".join(header_lines) + clean_text.strip() + "\n"
            new_md.write_text(markdown, encoding="utf-8")
            logger.info("Converted to markdown: %s", file_path.name)
        except Exception:
            logger.exception("Failed to convert HTML file: %s", file_path)


def convert_html_folder_tree(
    input_dir: str,
    output_dir: str,
    *,
    workers: int = 4,
    queue_size: int = 200,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("HTML input folder does not exist: %s", input_path)
        return

    files = sorted(input_path.rglob("*.html"))
    logger.info("Found %s HTML files in %s", len(files), input_path)
    if not files:
        return

    prefix = input_path.name or "FamCA"
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    stop_token = object()

    def producer():
        for fp in files:
            q.put(fp)
        for _ in range(workers):
            q.put(stop_token)

    def consumer(worker_id: int):
        while True:
            item = q.get()
            if item is stop_token:
                q.task_done()
                break
            file_path: Path = item
            try:
                year = file_path.parent.name or "unknown"
                folder_name = f"{prefix}_{year}"
                out_dir = output_path / folder_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{folder_name}_{file_path.stem}.md"
                out_path = out_dir / out_name

                html_content = file_path.read_text(encoding="utf-8", errors="ignore")
                clean_text = trafilatura.extract(
                    html_content,
                    include_tables=True,
                    include_formatting=True,
                    output_format="markdown",
                )
                if not clean_text:
                    logger.warning("No content extracted from: %s", file_path)
                    continue

                metadata = trafilatura.extract_metadata(html_content)
                title = metadata.title if metadata and metadata.title else file_path.stem
                date = metadata.date if metadata and metadata.date else "Unknown"
                catchwords = extract_legal_catchwords(html_content)

                header_lines = [f"# {title}"]
                header_lines.append(f"**Date:** {date}")
                if catchwords:
                    header_lines.append(f"**Catchwords:** {catchwords}")
                header_lines.append("")

                markdown = "\n".join(header_lines) + clean_text.strip() + "\n"
                out_path.write_text(markdown, encoding="utf-8")
                logger.info("Converted to markdown: %s", file_path)
            except Exception:
                logger.exception("Failed to convert HTML file: %s", file_path)
            finally:
                q.task_done()

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    consumers = [threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(workers)]
    for c in consumers:
        c.start()

    q.join()
    prod.join()
    for c in consumers:
        c.join()


def load_cases_documents(folder: str, skip_sources: set[str] | None = None) -> List[Document]:
    docs: List[Document] = []
    for path in Path(folder).rglob("*.txt"):
        if skip_sources and path.stem in skip_sources:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                logger.warning("Empty case file skipped: %s", path)
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "case",
                        "source": path.stem,
                    },
                )
            )
        except Exception:
            logger.exception("Failed to read case file: %s", path)
    for path in Path(folder).rglob("*.md"):
        if skip_sources and path.stem in skip_sources:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                logger.warning("Empty case file skipped: %s", path)
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "case",
                        "source": path.stem,
                        "format": "markdown",
                    },
                )
            )
        except Exception:
            logger.exception("Failed to read case file: %s", path)
    return docs


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
                        "compilation_date": row.get("compilation_date"),
                        "section_id": row.get("section_id"),
                        "section_title": row.get("section_title"),
                        "part": row.get("part"),
                        "division": row.get("division"),
                        "subdivision": row.get("subdivision"),
                        "chunk_id": row.get("chunk_id"),
                        "source_file": row.get("source_file"),
                    },
                )
            )
    return docs


def build_index(
    docs: Iterable[Document],
    db_path: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        list(docs),
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)],
    )
    index.storage_context.persist(persist_dir=db_path)
    print(f"Indexed {collection.count()} vectors into {db_path} ({collection_name})")


def get_existing_case_sources(db_path: str, collection_name: str) -> set[str]:
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    existing_sources: set[str] = set()
    if collection.count() == 0:
        return existing_sources

    results = collection.get(include=["metadatas"])
    for meta in results.get("metadatas", []) or []:
        if not meta:
            continue
        source = meta.get("source")
        if source:
            existing_sources.add(str(source))
    return existing_sources


def build_summary_index(rows: List[dict]) -> None:
    """
    Build the cases_summary collection with impact_analysis metadata.
    """
    Settings.embed_model = OpenAIEmbedding(model=Config.OPENAI_EMBED_MODEL)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name=SUMMARY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    def iter_summary_documents():
        for row in rows:
            # Extract impact_analysis and convert to string
            raw_summary = row.get("summary", {})
            impact_text = "No analysis provided."
            if isinstance(raw_summary, dict):
                impact_dict = raw_summary.get("impact_analysis", {})
                if isinstance(impact_dict, dict):
                    # Convert dict to readable string
                    parts = []
                    for finding in impact_dict.get("pivotal_findings", []):
                        if finding:
                            parts.append(f"Finding: {finding}")
                    for pivot in impact_dict.get("statutory_pivots", []):
                        if pivot:
                            parts.append(f"Pivot: {pivot}")
                    impact_text = "; ".join(parts) if parts else "No analysis provided."
                elif isinstance(impact_dict, str):
                    impact_text = impact_dict

            summary_sections = row.get("summary_sections") or []
            if isinstance(summary_sections, list) and summary_sections:
                for section in summary_sections:
                    text = section.get("text")
                    if not text: continue
                    yield Document(
                        text=text,
                        metadata={
                            "source_type": "case_summary",
                            "source": row["case_id"],
                            "case_id": row["case_id"],
                            "case_name": row["case_name"],
                            "summary_section": section.get("section", "unknown"),
                            "impact_analysis": impact_text,
                        },
                    )

    index = None
    batch: List[Document] = []
    for doc in iter_summary_documents():
        batch.append(doc)
        if len(batch) >= SUMMARY_INDEX_BATCH_SIZE:
            if index is None:
                index = VectorStoreIndex.from_documents(
                    batch,
                    storage_context=storage_context,
                )
            else:
                for doc in batch:
                    index.insert(doc)
            batch = []

    if batch:
        if index is None:
            index = VectorStoreIndex.from_documents(
                batch,
                storage_context=storage_context,
            )
        else:
            for doc in batch:
                index.insert(doc)

    if index is None:
        print("No summaries to index.")
        return

    index.storage_context.persist(persist_dir=DB_DIR)
    print(f"Indexed {collection.count()} summaries into {DB_DIR} ({SUMMARY_COLLECTION})")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_case_files(folder: str) -> List[Path]:
    txt_files = list(Path(folder).rglob("*.txt"))
    md_files = list(Path(folder).rglob("*.md"))
    return sorted(txt_files + md_files)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_case_summaries(
    cases_dir: str = CASES_TXT_DIR,
    skip_case_ids: set[str] | None = None,
) -> None:
    """
    Generate summaries for cases and build the cases_summary collection.
    """
    ensure_dir(os.path.dirname(SUMMARY_JSONL))
    Settings.llm = OpenAI(model=Config.OPENAI_MODEL, temperature=0.1)
    llm = Settings.llm

    rows = []
    for path in list_case_files(cases_dir):
        if skip_case_ids and path.stem in skip_case_ids:
            logger.info("Skipping summary for already-embedded case: %s", path.stem)
            continue
        try:
            text = read_text(path)
            if not text.strip():
                logger.warning("Empty case file skipped: %s", path)
                continue
            summary = generate_summary_dict(
                text,
                target_words=Config.AUSTLII_SUMMARY_TARGET_WORDS,
                max_words=Config.AUSTLII_SUMMARY_MAX_WORDS,
                list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
                list_limits_fallback=SUMMARY_LIST_LIMITS_FALLBACK,
                llm=llm,
                case_name=path.stem,
            )

            summary_sections = summary_json_to_sections(summary)
            summary_text = []
            for section in summary_sections:
                summary_text.append(f"--- {section.get('section', 'Unknown').replace('_', ' ').title()} ---")
                summary_text.append(section.get('text', ''))
                summary_text.append('')
            rows.append(
                {
                    "case_id": path.stem,
                    "case_name": path.stem,
                    "source_file": str(path.resolve()),
                    "summary": summary,
                    "summary_text": summary_text,
                    "summary_sections": summary_sections,
                }
            )
            print(f"[OK] {path.name}")
        except Exception:
            logger.exception("Failed to summarize case file: %s", path)

    write_jsonl(SUMMARY_JSONL, rows)
    build_summary_index(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embeddings for cases, statutes, and summaries.")
    parser.add_argument("--convert-html", action="store_true", help="Convert AustLII HTML cases to text.")
    parser.add_argument(
        "--convert-html-md",
        action="store_true",
        help="Convert AustLII HTML cases to markdown using Trafilatura.",
    )
    parser.add_argument(
        "--convert-html-md-tree",
        action="store_true",
        help="Convert HTML tree to markdown with FamCA_YYYY folders and filenames.",
    )
    parser.add_argument("--html-dir", help="Folder containing HTML files for conversion.")
    parser.add_argument("--converted-dir", help="Folder to write converted text files.")
    parser.add_argument("--existing-converted-dir", help="Folder containing already-converted text files to skip.")
    parser.add_argument("--cases", action="store_true", help="Build case embeddings.")
    parser.add_argument("--statutes", action="store_true", help="Build statute embeddings.")
    parser.add_argument("--summaries", action="store_true", help="Generate case summaries and build summary embeddings.")
    parser.add_argument("--cases-dir", help="Folder containing case .txt files for embeddings.")
    parser.add_argument("--summaries-dir", help="Folder containing case .txt files for summary generation.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases already embedded in the Chroma collection.",
    )
    # --- extend parse_args() ---
    parser.add_argument("--md-tree-output-dir", help="Output dir for HTML tree markdown conversion.")
    parser.add_argument("--workers", type=int, default=4, help="Number of conversion workers.")
    parser.add_argument("--queue-size", type=int, default=200, help="Queue size for conversion.")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    init_embeddings()
    summaries_dir = args.summaries_dir or CASES_TXT_DIR

    if args.convert_html_md_tree and not args.summaries_dir:
        summaries_dir = args.md_tree_output_dir or HTML_MD_TREE_OUTPUT_DIR
    if not (args.cases or args.statutes or args.summaries):
        args.cases = args.statutes = args.summaries = True

    if args.convert_html_md:
        html_dir = args.html_dir or HTML_INPUT_DIR
        converted_dir = args.converted_dir or HTML_MD_OUTPUT_DIR
        existing_dir = args.existing_converted_dir
        convert_html_folder_trafilatura(
            html_dir,
            converted_dir,
            existing_dir=existing_dir,
            workers=args.workers,
            queue_size=args.queue_size,
        )

    if args.convert_html_md_tree:
        html_dir = args.html_dir or HTML_INPUT_DIR
        converted_dir = args.md_tree_output_dir or HTML_MD_TREE_OUTPUT_DIR
        convert_html_tree_trafilatura(
            html_dir,
            converted_dir,
            workers=args.workers,
            queue_size=args.queue_size,
        )

    

    cases_dir = args.cases_dir or CASES_TXT_DIR
    if args.convert_html_md_tree and not args.cases_dir:
        cases_dir = args.md_tree_output_dir or HTML_MD_TREE_OUTPUT_DIR


    if args.cases:
        skip_sources = set()
        if args.skip_existing:
            skip_sources = get_existing_case_sources(DB_DIR, CASES_COLLECTION)
            print(f"Skipping {len(skip_sources)} already-embedded cases.")
        case_docs = load_cases_documents(cases_dir, skip_sources=skip_sources)
        try:
            build_index(
                case_docs,
                db_path=DB_DIR,
                collection_name=CASES_COLLECTION,
                chunk_size=Config.CASE_CHUNK_SIZE,
                chunk_overlap=Config.CASE_CHUNK_OVERLAP,
            )
        except Exception:
            logger.exception("Failed to build case embeddings")
            raise

    if args.statutes:
        statute_docs = load_statutes_documents(STATUTES_JSONL)
        try:
            build_index(
                statute_docs,
                db_path=DB_DIR,
                collection_name=STATUTES_COLLECTION,
                chunk_size=Config.STATUTE_CHUNK_SIZE,
                chunk_overlap=Config.STATUTE_CHUNK_OVERLAP,
            )
        except Exception:
            logger.exception("Failed to build statute embeddings")
            raise

    if args.summaries:
        try:
            skip_case_ids = set()
            if args.skip_existing:
                skip_case_ids = get_existing_case_sources(DB_DIR, SUMMARY_COLLECTION)
                print(f"Skipping {len(skip_case_ids)} already-embedded summaries.")
            run_case_summaries(cases_dir=summaries_dir, skip_case_ids=skip_case_ids)
        except Exception:
            logger.exception("Failed to build summary embeddings")
            raise

