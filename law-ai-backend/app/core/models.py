import os
import logging
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb
from app.core.logger import logger
from app.core.config import Config

logging.basicConfig(level=logging.INFO)

class ModelManager:
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self.cases_index = None
        self.statutes_index = None
        self.case_summaries_index = None
        self.uploaded_cases_index = None
        self.cases_collection = None
        self.statutes_collection = None
        self.case_summaries_collection = None
        self.uploaded_case_ids = set()

    def init_models(self):
        if self.llm is None or self.embed_model is None:
            logger.info("Initializing LLM and embeddings...")
            self.llm = OpenAI(model=Config.OPENAI_MODEL, temperature=0.1)
            self.embed_model = OpenAIEmbedding(model=Config.OPENAI_EMBED_MODEL)
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
        logger.info("LLM and embeddings ready.")
        return self.llm, self.embed_model

    def _db_dir(self) -> str:
        return Config.VECTOR_DB_DIR

    def create_or_load_cases_index(self):
        if self.cases_index is not None:
            return self.cases_index

        cases_db_dir = self._db_dir()
        cases_collection_name = Config.CASES_COLLECTION_NAME

        logger.info("Initializing cases vector index...")
        chroma_client = chromadb.PersistentClient(path=cases_db_dir)
        self.cases_collection = chroma_client.get_or_create_collection(
            name=cases_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        vector_store = ChromaVectorStore(chroma_collection=self.cases_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=cases_db_dir
        )

        self.cases_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
        logger.info("Cases vector count: %s", self.cases_collection.count())
        return self.cases_index

    def create_or_load_statutes_index(self):
        if self.statutes_index is not None:
            return self.statutes_index

        statutes_db_dir = self._db_dir()
        statutes_collection_name = Config.STATUTES_COLLECTION_NAME

        logger.info("Initializing statutes vector index...")
        chroma_client = chromadb.PersistentClient(path=statutes_db_dir)
        self.statutes_collection = chroma_client.get_or_create_collection(
            name=statutes_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        vector_store = ChromaVectorStore(chroma_collection=self.statutes_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=statutes_db_dir
        )

        self.statutes_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
        logger.info("Statutes vector count: %s", self.statutes_collection.count())
        return self.statutes_index

    def create_or_load_case_summaries_index(self):
        if self.case_summaries_index is not None:
            return self.case_summaries_index

        summaries_db_dir = self._db_dir()
        summaries_collection_name = Config.SUMMARY_COLLECTION_NAME

        logger.info("Initializing case summaries vector index...")
        chroma_client = chromadb.PersistentClient(path=summaries_db_dir)
        self.case_summaries_collection = chroma_client.get_or_create_collection(
            name=summaries_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        vector_store = ChromaVectorStore(chroma_collection=self.case_summaries_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=summaries_db_dir
        )

        self.case_summaries_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
        logger.info("Case summaries vector count: %s", self.case_summaries_collection.count())
        return self.case_summaries_index

    def create_or_load_uploaded_cases_index(self):
        if self.uploaded_cases_index is None:
            raise ValueError("Uploaded cases index not initialized.")
        return self.uploaded_cases_index

    def create_or_load_index(self):
        """
        Backward-compatible: return the cases index by default.
        """
        return self.create_or_load_cases_index()
    
    def add_case(self, case_id: str, text: str):
        if self.cases_index is None:
            raise ValueError("Cases index not initialized. Call create_or_load_cases_index() first.")

        # Split into chunks
        chunks = text.split("\n\n")  # simple chunking; can replace with SentenceSplitter
        documents = [Document(text=chunk, metadata={"source": case_id}) for chunk in chunks]

        # Add to vector index
        for doc in documents:
            self.cases_index.insert(doc)

        # Persist index to disk
        cases_db_dir = self._db_dir("./chroma_db", "CASES_VECTOR_DB_DIR")
        self.cases_index.storage_context.persist(persist_dir=cases_db_dir)

        logger.info(f"Added new case '{case_id}' with {len(chunks)} chunks to vector index.")

    def has_uploaded_case(self, case_id: str | int) -> bool:
        return str(case_id) in self.uploaded_case_ids

    def add_uploaded_case_documents(self, case_id: str | int, documents: list[Document], *, allow_existing: bool = False) -> None:
        if not documents:
            return
        case_id_str = str(case_id)
        if case_id_str in self.uploaded_case_ids and not allow_existing:
            logger.info("Uploaded case '%s' already embedded in memory; skipping.", case_id_str)
            return

        if self.uploaded_cases_index is None:
            # First uploaded case - create the in-memory index
            self.uploaded_cases_index = VectorStoreIndex.from_documents(
                documents,
                embed_model=Settings.embed_model,
            )
        else:
            for doc in documents:
                self.uploaded_cases_index.insert(doc)

        self.uploaded_case_ids.add(case_id_str)
        logger.info("Embedded uploaded case '%s' (%s chunks) in memory.", case_id_str, len(documents))

model_manager = ModelManager()
