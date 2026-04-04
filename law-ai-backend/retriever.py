"""
Retrieval module: hybrid search (vector + BM25) with reranking.
"""

import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION = os.getenv("QDRANT_COLLECTION", "australian_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", 10))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
BM25_INDEX_PATH = Path("bm25_index.pkl")

# Cross-encoder for reranking (small but effective)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading reranker model...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        print("Connecting to Qdrant...")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print("Loading BM25 index...")
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.bm25_chunk_ids = bm25_data["chunk_ids"]
            self.bm25_texts = bm25_data["texts"]
            self.bm25_records = bm25_data["records"]
        else:
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_INDEX_PATH}. Run ingest.py first."
            )

        print("✅ Retriever ready")

    def _vector_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Semantic search via Qdrant."""
        query_vec = self.embedder.encode(query).tolist()
        results = self.qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vec,
            limit=top_k,
        )
        hits = []
        for r in results:
            hit = {**r.payload, "score_vector": r.score}
            hits.append(hit)
        return hits

    def _bm25_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Keyword search via BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                record = self.bm25_records[idx]
                hit = {**record, "score_bm25": float(scores[idx])}
                hits.append(hit)
        return hits

    def _merge_results(
        self,
        vector_hits: list[dict],
        bm25_hits: list[dict],
        vector_weight: float = 0.4,
        bm25_weight: float = 0.6,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) to merge vector and BM25 results.
        For statutes, BM25 is weighted higher (0.6) by default.
        """
        k = 60  # RRF constant

        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit.get("chunk_id", str(rank))
            rrf = vector_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf
            docs[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit.get("chunk_id", str(rank))
            rrf = bm25_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf
            if cid not in docs:
                docs[cid] = hit

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        merged = []
        for cid in sorted_ids:
            doc = docs[cid]
            doc["score_rrf"] = scores[cid]
            merged.append(doc)
        return merged

    def _rerank(self, query: str, docs: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        """Rerank using cross-encoder."""
        if not docs:
            return []

        pairs = [(query, doc.get("text", "")[:512]) for doc in docs]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["score_rerank"] = float(score)

        reranked = sorted(docs, key=lambda x: x["score_rerank"], reverse=True)
        return reranked[:top_k]

    def search(self, query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
        """
        Full hybrid search pipeline:
        1. Vector search (semantic)
        2. BM25 search (keyword)
        3. RRF merge
        4. Cross-encoder rerank
        """
        vector_hits = self._vector_search(query, top_k=TOP_K)
        bm25_hits = self._bm25_search(query, top_k=TOP_K)

        # For statutes: BM25 weighted higher
        merged = self._merge_results(vector_hits, bm25_hits, vector_weight=0.4, bm25_weight=0.6)

        # Take top candidates for reranking
        candidates = merged[: TOP_K * 2]

        # Rerank
        reranked = self._rerank(query, candidates, top_k=top_k)
        return reranked
