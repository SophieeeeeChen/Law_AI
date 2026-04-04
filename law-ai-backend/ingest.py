"""
Ingestion script: loads statute JSONL into Qdrant vector database
and builds a BM25 index saved to disk.

Usage:
    python ingest.py --file Austlii_law_statutes/family_law_act_1975_sections.jsonl
"""

import json
import os
import pickle
import argparse
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION = os.getenv("QDRANT_COLLECTION", "australian_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BM25_INDEX_PATH = Path("bm25_index.pkl")


def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} chunks from {path}")
    return records


def build_embedding_text(record: dict) -> str:
    """
    Create an optimised text for embedding.
    Combines section title + part/division context + first ~500 chars of body.
    """
    parts = []
    if record.get("act"):
        parts.append(record["act"])
    if record.get("part"):
        parts.append(record["part"])
    if record.get("division"):
        parts.append(record["division"])
    if record.get("section_title"):
        parts.append(f"Section {record.get('section_id', '')} — {record['section_title']}")
    body = record.get("text", "")
    # For embedding, take a reasonable window
    parts.append(body[:1000])
    return " | ".join(parts)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def main():
    parser = argparse.ArgumentParser(description="Ingest law JSONL into Qdrant + BM25")
    parser.add_argument("--file", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()

    # --- Load data ---
    records = load_jsonl(args.file)

    # --- Embedding model ---
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # --- Qdrant setup ---
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if args.recreate:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Recreated collection '{COLLECTION}'")
    else:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION not in collections:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Created collection '{COLLECTION}'")
        else:
            print(f"Collection '{COLLECTION}' already exists, appending...")

    # --- Create payload index for filtering ---
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="source_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="act",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="section_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # Indexes may already exist

    # --- Prepare texts for embedding ---
    embedding_texts = [build_embedding_text(r) for r in records]

    # --- Encode in batches ---
    print("Encoding embeddings...")
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(embedding_texts), batch_size):
        batch = embedding_texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embs)
        print(f"  Encoded {min(i + batch_size, len(embedding_texts))}/{len(embedding_texts)}")

    # --- Upsert into Qdrant ---
    print("Upserting into Qdrant...")
    points = []
    for idx, (record, embedding) in enumerate(zip(records, all_embeddings)):
        payload = {
            "chunk_id": record.get("chunk_id", ""),
            "source_type": record.get("source_type", "statute"),
            "act": record.get("act", ""),
            "section_id": record.get("section_id", ""),
            "section_title": record.get("section_title", ""),
            "part": record.get("part", ""),
            "division": record.get("division", ""),
            "subdivision": record.get("subdivision", ""),
            "text": record.get("text", ""),
        }
        points.append(PointStruct(id=idx, vector=embedding.tolist(), payload=payload))

    # Upsert in batches
    for i in range(0, len(points), 100):
        batch = points[i : i + 100]
        client.upsert(collection_name=COLLECTION, points=batch)
        print(f"  Upserted {min(i + 100, len(points))}/{len(points)}")

    # --- Build BM25 index ---
    print("Building BM25 index...")
    corpus_tokens = [tokenize(r.get("text", "")) for r in records]
    bm25 = BM25Okapi(corpus_tokens)

    bm25_data = {
        "bm25": bm25,
        "chunk_ids": [r.get("chunk_id", "") for r in records],
        "texts": [r.get("text", "") for r in records],
        "records": records,
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

    print("\n✅ Ingestion complete!")
    print(f"   Qdrant: {len(points)} vectors in '{COLLECTION}'")
    print(f"   BM25:   {len(corpus_tokens)} documents indexed")


if __name__ == "__main__":
    main()
