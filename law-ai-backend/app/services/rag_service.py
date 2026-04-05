from ast import Dict
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from app.core.models import model_manager
from app.core.logger import logger
from app.core.dev_logger import format_and_log
from app.core.config import Config
from llama_index.core import get_response_synthesizer, PromptTemplate, Settings
from app.services.summary_service import SUMMARY_LIST_LIMITS_PRIMARY, generate_summary_dict
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from scipy.spatial.distance import cosine
import numpy as np

from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.llms.anthropic import Anthropic as AnthropicLLM
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator

from app.services.clarify_service import TOPIC_KEYWORDS
import re

from app.core.config import Config

my_prompt_template = PromptTemplate(Config.QA_TEMPLATE)


def _build_history_block(history_text: str | None) -> str:
    if not history_text:
        return ""
    return f"Conversation history:\n{history_text}\n\n"


def _node_id(node) -> str:
    return getattr(node, "node_id", None) or getattr(node, "id_", None) or str(id(node))


def _build_structured_query(
    question: str,
    case_section_text: str,
    history_text: str | None,
    topic: str | None = None,
) -> str:
    """Build a structured query combining question, case context, and history."""
    # Topic Header
    topic_label = topic.replace('_', ' ').title() if topic else 'General Family Law'
    
    # Select relevant keywords
    relevant_tags = []
    if topic and topic in TOPIC_KEYWORDS:
        keywords = TOPIC_KEYWORDS[topic]
        input_lower = question.lower()
        relevant_tags = [kw for kw in keywords if any(word in input_lower for word in kw.lower().split())]
        relevant_tags = relevant_tags[:8]  # Limit to top 8
    
    # Build query components
    components = [f"[{topic_label}]"]
    
    if relevant_tags:
        components.append(f"Legal terms: {', '.join(relevant_tags)}")
    
    components.append(f"Question: {question}")
    
    if case_section_text:
        components.append(f"Case context: {case_section_text}")
    
    if history_text:
        history_summary = history_text[-200:] + "..." if len(history_text) > 200 else history_text
        components.append(f"History: {history_summary}")
    
    return "\n".join(components)


def _log_retrieval(
    kind: str,
    question: str,
    topic: str | None,
    files: list[str],
    nodes: list[dict],
) -> None:
    """Log retrieval events for debugging and monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "kind": kind,
        "question": question,
        "topic": topic,
        "files_retrieved": files,
        "node_count": len(nodes),
        "nodes": nodes,
    }
    logger.info(f"Retrieval [{kind}]: {len(files)} files, {len(nodes)} nodes")
    logger.debug(f"Retrieval details: {json.dumps(log_entry, indent=2)}")


def _normalize_scores(nodes):
    scores = [n.score for n in nodes if n.score is not None]
    if not scores:
        return {}
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return {_node_id(n): 1.0 for n in nodes}
    return {_node_id(n): (n.score - min_s) / (max_s - min_s) for n in nodes}

def _merge_nodes(vector_nodes, bm25_nodes):
    merged = {}
    for node in vector_nodes + bm25_nodes:
        merged[_node_id(node)] = node
    return list(merged.values())

def _apply_manual_filter(nodes, filters):
    """Ensures BM25 nodes respect the metadata filters applied to Vector nodes."""
    filtered = []
    for node in nodes:
        match = True
        for f in filters.filters:
            if node.metadata.get(f.key) != f.value:
                match = False
                break
        if match:
            filtered.append(node)
    return filtered

def _hybrid_retrieve(
    index,
    query: str,
    use_rerank: bool,
    *,
    vector_top_k: int | None = None,
    metadata_filters=None,
    limit: int | None = None,
):
    """Vector retrieval + optional LLM rerank.
    """
    v_k = limit or vector_top_k or Config.TOP_K

    # 1. Vector Retrieval (ChromaDB handles filters natively)
    retriever_kwargs = {"similarity_top_k": v_k}
    if metadata_filters is not None:
        retriever_kwargs["filters"] = metadata_filters

    vector_retriever = index.as_retriever(**retriever_kwargs)
    vector_nodes = vector_retriever.retrieve(query)

    # 2. Rerank (or pass-through)
    if use_rerank and LLMRerank is not None:
        reranker = LLMRerank(top_n=Config.RERANK_TOP_N)
        vector_nodes = reranker.postprocess_nodes(vector_nodes, query_str=query)
    else:
        vector_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

    return vector_nodes[:limit] if limit else vector_nodes


# ---------------------------------------------------------------------------
#  Statute-specific hybrid: exact section_id keyword match + vector semantic
# ---------------------------------------------------------------------------

def _normalize_section_ref(raw: str) -> str:
    """Turn user notations into a Chroma-friendly section key.

    Examples:  's 79' → '79',  'S.79' → '79',  '79(1)' → '79',
               'section 79' → '79'
    """
    s = raw.lower().strip()
    s = re.sub(r"^(section|sect?\.?)\s*", "", s)
    s = re.sub(r"\(.*\)", "", s)  # drop sub-section
    return s.strip()


def _extract_section_refs(text: str) -> list[str]:
    """Pull all plausible FLA section references from the user query."""
    patterns = [
        r"(?:section|sect?\.?)\s*(\d+[A-Za-z]*)",
        r"\bs\s*(\d+[A-Za-z]*)\b",
        r"\bss\s*(\d+[A-Za-z]*)\b",
    ]
    refs: list[str] = []
    for p in patterns:
        refs.extend(re.findall(p, text, re.IGNORECASE))
    return [_normalize_section_ref(r) for r in refs]


def _keyword_search_statutes(collection, section_ids: list[str], top_k: int = 5) -> list[dict]:
    """Exact metadata match on ``section_id`` in the statutes ChromaDB collection."""
    results: list[dict] = []
    for sid in section_ids:
        hits = collection.get(
            where={"section_id": {"$eq": sid}},
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(hits.get("documents") or [], hits.get("metadatas") or []):
            results.append({"text": doc, **(meta or {})})
        if len(results) >= top_k:
            break
    return results[:top_k]


def _hybrid_retrieve_statutes(
    index,
    query: str,
    *,
    use_rerank: bool = True,
    limit: int = 3,
) -> list:
    """Statute retrieval: exact section-id keyword lookup + vector semantic,
    merged and optionally reranked.

    This is the only collection where keyword matching is used because the
    statute corpus is small and section-id exact matches are high-value.
    """
    import re as _re  # already imported at module level; local alias for clarity

    # --- A. Keyword hits (exact section_id from ChromaDB metadata) ---
    keyword_docs = []
    vector_store = index.vector_store
    chroma_collection = getattr(vector_store, "_collection", None)

    if chroma_collection is not None:
        section_refs = _extract_section_refs(query)
        if section_refs:
            raw_hits = _keyword_search_statutes(chroma_collection, section_refs, top_k=limit)
            # Wrap raw dicts into lightweight objects that look like retriever nodes
            # so they can be merged with vector nodes.
            from llama_index.core.schema import TextNode, NodeWithScore
            for hit in raw_hits:
                node = TextNode(
                    text=hit.pop("text", ""),
                    metadata=hit,
                )
                keyword_docs.append(NodeWithScore(node=node, score=1.0))  # max score for exact match

    # --- B. Vector (semantic) hits ---
    retriever = index.as_retriever(similarity_top_k=limit)
    vector_nodes = retriever.retrieve(query)

    # --- C. Merge & deduplicate by section_id ---
    merged = _merge_nodes(keyword_docs, vector_nodes)

    # --- D. Rerank or weighted sort ---
    if use_rerank and LLMRerank is not None:
        reranker = LLMRerank(top_n=Config.RERANK_TOP_N)
        merged = reranker.postprocess_nodes(merged, query_str=query)
    else:
        kw_norm = _normalize_scores(keyword_docs)
        vec_norm = _normalize_scores(vector_nodes)
        for node in merged:
            nid = _node_id(node)
            node.score = (
                Config.HYBRID_VECTOR_WEIGHT * vec_norm.get(nid, 0.0)
                + Config.HYBRID_BM25_WEIGHT * kw_norm.get(nid, 0.0)
            )
        merged.sort(key=lambda n: n.score or 0.0, reverse=True)

    return merged[:limit]


async def get_precedent_data(index, case_name: str, topic: str):
    """
    Directly retrieves the embedding vector for the 'facts' section of a specific case.
    """
    # 1. Build the filter for this specific case's facts
    vector_store = index.vector_store
    
    # Access ChromaDB collection directly
    if hasattr(vector_store, "_collection"):
        collection = vector_store._collection
        
        # Query by metadata only (no query embedding)
        results = collection.get(
            where={
                "$and": [
                    {"case_name": case_name},
                    {"summary_section": topic}
                ]
            },
            include=["embeddings", "documents", "metadatas"]
        )
        
        if not results or not results["ids"]:
            return None, "Information not available."
        
        # Extract embeddings and text
        embeddings = results["embeddings"]
        documents = results["documents"]
        
        if embeddings is None or len(embeddings) == 0:
            embedding = None
        elif len(embeddings) > 1:
            embedding = np.mean(embeddings, axis=0)
        else:
            embedding = embeddings[0]
        
        full_text = "\n".join(documents)
        
        if Config.ENV == "dev":
            format_and_log(
                "/get_precedent_data",
                f"Retrieved {len(documents)} node(s) for {case_name} | Section: {topic}",
                "data",
                {
                    "case_name": case_name,
                    "topic": topic,
                    "has_embedding": embedding is not None,
                    "text_length": len(full_text)
                }
            )
        
        return embedding, full_text
    
    # Fallback to old method if direct access not available
    raise ValueError("ChromaDB collection not accessible")


async def get_user_data(model_manager, case_id: int, topic: str):
    """Retrieves the embedding for the specified section of the uploaded case."""
    if model_manager.uploaded_cases_index is None:
        raise ValueError("Uploaded cases index is not initialized.")

    vector_store = model_manager.uploaded_cases_index.vector_store
    
    # Direct access to in-memory cache
    if not hasattr(vector_store, "data"):
        raise ValueError("Vector store format not supported")
    
    metadata_dict = vector_store.data.metadata_dict
    embedding_dict = vector_store.data.embedding_dict
    
    # Find matching node by metadata
    for node_id, metadata in metadata_dict.items():
        if (metadata.get("case_id") == str(case_id) and 
            metadata.get("summary_section") == topic):
            
            embedding = embedding_dict.get(node_id)
            # Get text from docstore
            text_content = ""
            if hasattr(model_manager.uploaded_cases_index, "docstore"):
                doc = model_manager.uploaded_cases_index.docstore.get_document(node_id)
                if doc:
                    text_content = doc.get_text()
            
            if Config.ENV == "dev":
                format_and_log(
                    "/get_user_data",
                    f"Extracted User Case {case_id} | Section: {topic}",
                    "node_data",
                    {
                        "case_id": case_id,
                        "section": topic,
                        "has_vector": embedding is not None,
                        "char_count": len(text_content),
                    },
                )
            
            return embedding, text_content
    return None, ""

def calculate_similarity(user_fact_vec: list[float] | np.ndarray, precedent_fact_vec: list[float] | np.ndarray) -> float:
    """
    Calculates similarity between two pre-computed vectors.
    """
    if user_fact_vec is None or precedent_fact_vec is None:
        return 0.0

    # Ensure inputs are numpy arrays for the distance calculation
    vec_a = np.array(user_fact_vec)
    vec_b = np.array(precedent_fact_vec)

    # 1 - cosine distance = cosine similarity
    # We use '1 -' because cosine distance is 0 when vectors are identical
    similarity = 1 - cosine(vec_a, vec_b)

    # Handle potential NaN if a vector is all zeros
    return float(similarity) if not np.isnan(similarity) else 0.0

#TODO   to be updated
def answer_query_with_trace_withoutUploadFile(
    query: str,
    history_text: str | None = None,
    topic: str | None = None,
):
    """
    Answer a question using the RAG system with logging and node filtering.
    Uses the global model_manager to avoid reloading models or index each time.
    
    Returns:
        response_text (str): The AI answer.
        formatted_nodes (list): List of retrieved nodes with metadata and content.
    """
    logger.info(f"Processing query: {query}")
    index = model_manager.create_or_load_index()

    retrieval_query = _build_structured_query(query, "", history_text, topic=topic)
    initial_nodes = _hybrid_retrieve(index, retrieval_query, use_rerank=Config.HYBRID_USE_RERANK)
    if not initial_nodes:
        logger.warning("No nodes retrieved from vector index.")
        return f"⚠️ No nodes retrieved for your query: '{query}'", []

    filtered_nodes = [node for node in initial_nodes if node.score >= Config.MIN_RERANK_SCORE]
    if not filtered_nodes:
        logger.warning("No nodes passed the minimum rerank score threshold.")
        return f"⚠️ No relevant nodes found for your query: '{query}'", []

    formatted_nodes = [
        {
            "file_name": node.metadata.get("source", "Unknown"),
            "score": node.score,
            "content": node.get_text()
        }
        for node in filtered_nodes
    ]
    _log_retrieval(
        kind="general",
        question=query,
        topic=topic,
        files=[n["file_name"] for n in formatted_nodes],
        nodes=[
            {
                "file_name": n["file_name"],
                "score": n["score"],
                "content": n["content"][:500],
            }
            for n in formatted_nodes
        ],
    )

    # Prepare context string for the AI prompt
    context_str = "\n---\n".join([node.get_text() for node in filtered_nodes])
    context_count = len(filtered_nodes)
    history_block = _build_history_block(history_text)
    formatted_prompt = my_prompt_template.format(
        context_count=context_count,
        context_str=context_str,
        query_str=f"{history_block}{query}"
    )

    #Generate the AI response
    response_synthesizer = get_response_synthesizer(verbose=True)
    response = response_synthesizer.synthesize(formatted_prompt, nodes=filtered_nodes)
    response_text = response.response

    logger.info(f"Query processed. Retrieved {len(filtered_nodes)} nodes. Answer length: {len(response_text)} chars.")

    return response_text, formatted_nodes


def compress_case_facts(case_text: str, max_chars: int = 2000) -> str:
    from app.core.config import Config
    from app.core.logger import logger
    import re
    
    try:
        summary_obj = generate_summary_dict(
            case_text,
            "uploaded_case",
            list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
            llm=Settings.llm,
            case_name="uploaded_case",
        )
    except (ValueError, Exception) as e:
        error_msg = str(e).lower()
        if "no candidates" in error_msg or "safety" in error_msg or "blocked" in error_msg or "google" in error_msg or "gemini" in error_msg:
            logger.warning(f"Gemini failed for summary generation ({e}). Falling back to OpenAI model: {Config.OPENAI_MODEL}")
            from llama_index.llms.openai import OpenAI as OpenAILLM
            fallback_llm = OpenAILLM(
                model=Config.OPENAI_MODEL,
                temperature=0.1,
                timeout=300.0,
                max_retries=2,
            )
            summary_obj = generate_summary_dict(
                case_text,
                "uploaded_case",
                list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
                llm=fallback_llm,
                case_name="uploaded_case",
            )
        else:
            raise

    # Uploaded files are often hypotheticals/undecided scenarios.
    # If the uploaded text doesn't look like it contains an actual judgment/outcome,
    # keep `outcome_orders` as null (so we don't store a hallucinated "likely outcome"
    # inside the persisted summary).
    def _looks_like_decided_case(text: str) -> bool:
        t = (text or "").lower()
        # Neutral citation style: [YYYY] Something N
        if re.search(r"\[\d{4}\]\s*[a-z]{2,}\s*\d+", t):
            return True
        decided_markers = (
            "final orders",
            "orders made",
            "the court orders",
            "the court ordered",
            "it is ordered",
            "judgment",
            "reasons for judgment",
            "appeal allowed",
            "appeal dismissed",
            "orders of the court",
            "held that",
        )
        return any(m in t for m in decided_markers)

    if isinstance(summary_obj, dict) and not _looks_like_decided_case(case_text):
        summary_obj["outcome_orders"] = None

    if str(getattr(Config, "ENV", "")).lower() in {"dev", "development"}:
        logger.debug(f"Compressed case facts summary_obj keys: {list(summary_obj.keys()) if isinstance(summary_obj, dict) else type(summary_obj)}")
    return json.dumps(summary_obj, ensure_ascii=False)


async def answer_case_question_withuploadFile(
    question: str,
    case_section_text: str,  # Facts from the user's uploaded file
    history_text: str,       # Previous chat turns from cache
    topic: str = None,
    case_id: int | None = None
) -> tuple[str, list[dict]]:
    """
    Final Triple-Thread RAG: Statutes + Strategic Impact + Full Text.
    Returns: (UI_Answer, Citations_List)
    """
    
    # 1. RETRIEVE CASE SUMMARIES (The Strategic Child Search)
    # We search the summary collection to get the 'impact_analysis' metadata.
    summary_index = model_manager.create_or_load_case_summaries_index()
    summary_filters = None
    # It builds a metadata filter so summary retrieval only returns nodes whose summary_section equals the topic (e.g., property_division). Without it, summaries from all sections can be returned.
    if topic:
        summary_filters = MetadataFilters(filters=[
            ExactMatchFilter(key="summary_section", value=topic)
        ])

    logger.debug(f"Summary retrieval filters: {summary_filters.filters if summary_filters else 'None'}")
    summary_query = _build_structured_query(question, case_section_text, history_text, topic)
    summary_nodes = _hybrid_retrieve(
        summary_index, 
        summary_query, 
        use_rerank=True, 
        metadata_filters=summary_filters, 
        limit=10
    )
    if Config.ENV == "dev":
        format_and_log(
            "/answer_case_question_withuploadFile",
            "get similar summary_nodes",
            "summary_nodes",
            {
                "topic": topic,
                "summary_query": summary_query,
                "filters": (
                    [{"key": f.key, "value": f.value} for f in summary_filters.filters]
                    if summary_filters else []
                ),
                "count": len(summary_nodes),
                "samples": [
                    {
                        "case_name": (n.metadata or {}).get("case_name"),
                        "summary_section": (n.metadata or {}).get("summary_section"),
                        "score": round(n.score, 4) if n.score is not None else None,
                    }
                    for n in summary_nodes
                ],
            },
        )
    # 2. RETRIEVE DEEP PRECEDENT (The Parent Search)
    # Using case_ids found in summaries to pull granular details/analogies from full judgments.
    #TODO To be updated - we should pull out the 'facts'&'impact_analysis' from the summary metadata and include it in the prompt for each case, rather than just logging it here. This way, the AI can directly use the strategic insights from the summaries when analyzing each precedent.
    reranked_nodes = []
    # user_fact_vec, user_fact_text = await get_user_data(model_manager, case_id=case_id, topic="facts")
    user_topic_vec, user_topic_text = await get_user_data(model_manager, case_id=case_id, topic=topic)  # e.g., "property_division"
    user_impact_vec, user_impact_text = await get_user_data(model_manager, case_id=case_id, topic="overall_impact_analysis")
    user_uncert_vec, user_uncert_text = await get_user_data(model_manager, case_id=case_id, topic="general_credibility_risk")
    
    #3.RETRIEVE STATUTES (The Statutory Search)
    statutes_index = model_manager.create_or_load_statutes_index()
    statute_query_parts = [topic or '']
    statute_query_parts.append(question)
    if user_topic_text:
        statute_query_parts.append(user_topic_text)   # truncate to avoid overly long query
    if user_impact_text:
        statute_query_parts.append(user_impact_text)
    statute_query = " ".join(statute_query_parts)

    statute_nodes = _hybrid_retrieve_statutes(
        statutes_index,
        statute_query,
        use_rerank=True,
        limit=3,
    )
    if Config.ENV == "dev":
        format_and_log(
            "/answer_case_question_withuploadFile",
            "get similar statute_nodes",
            "statute_nodes",
            statute_nodes[:3],  # Log top 3 statute nodes
        )
    context_statutes = "\n".join([f"- {n.get_text()}" for n in statute_nodes])

    for s_node in summary_nodes:
        case_name = s_node.metadata.get("case_name")
        # precedent_fact_vec, p_fact_text = await get_precedent_data(summary_index, case_name, "facts")
        precedent_topic_vec, p_topic_text = await get_precedent_data(summary_index, case_name, topic)
        precedent_impact_vec, p_impact_text = await get_precedent_data(summary_index, case_name, "overall_impact_analysis")
        precedent_uncert_vec, p_uncert_text = await get_precedent_data(summary_index, case_name, "general_credibility_risk")
        precedent_outcome_vec, p_outcome_text = await get_precedent_data(summary_index, case_name, "outcome_orders")
        precedent_reasons_vec, p_reasons_text = await get_precedent_data(summary_index, case_name, "reasons_rationale")
        # fact_score = calculate_similarity(user_fact_vec, precedent_fact_vec)
        topic_score = calculate_similarity(user_topic_vec, precedent_topic_vec) if precedent_topic_vec is not None else 0.0
        impact_score = calculate_similarity(user_impact_vec, precedent_impact_vec) if precedent_impact_vec is not None else 0.0
        uncert_score = calculate_similarity(user_uncert_vec, precedent_uncert_vec) if precedent_uncert_vec is not None else 0.0

        final_score = (topic_score * 0.7) + (impact_score * 0.2) + (uncert_score * 0.1)
        # 5. Store the score in metadata and keep the node
        s_node.metadata.update({
        "rerank_score": final_score,
        "full_facts": p_topic_text,
        "overall_impact_analysis": p_impact_text,
        "general_credibility_risk": p_uncert_text,
        "outcome_orders": p_outcome_text,
        "reasons_rationale": p_reasons_text
    })
        reranked_nodes.append(s_node)
    reranked_nodes.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

    # 7. Take the top 3-5 for the LLM
    top_precedents = reranked_nodes[:3]
    full_index = model_manager.create_or_load_cases_index()
    precedent_blocks = []
    for s_node in top_precedents:
        case_name = s_node.metadata.get("case_name")
        
        # --- THE DEEP DIVE ---
        # We already have the 'Logic' (Summary). 
        # Now we get the 'Evidence' (Full Text) for the specific USER QUESTION.
        ft_filters = MetadataFilters(filters=[ExactMatchFilter(key="case_name", value=case_name)])
        ft_retriever = full_index.as_retriever(filters=ft_filters, similarity_top_k=2)
        
        # This finds the exact paragraphs in the 50-page judgment that answer the question
        evidence_nodes = await ft_retriever.aretrieve(question)
        evidence_text = "\n".join([f"  - [Direct Judgment Quote]: {en.text}" for en in evidence_nodes])
        if Config.ENV == "dev":
            format_and_log(
                "/answer_case_question_withuploadFile",
                f"Retrieved evidence nodes from full case for {case_name}",
                "evidence_nodes",
                {
                    "case_name": case_name,
                    "samples": evidence_text,
                }
            )
        # Assemble the final block for the Prompt
        block = f"""
        ### PRECEDENT CASE: {case_name} (Relevance Score: {s_node.metadata.get('rerank_score', 0):.2f})

        **1. SUMMARY OF FACTS:** {s_node.metadata.get('full_facts', 'N/A')}

        **2. JUDICIAL REASONS & RATIONALE:** {s_node.metadata.get('reasons_rationale', 'N/A')}

        **3. FINAL OUTCOME & ORDERS:** {s_node.metadata.get('outcome_orders', 'N/A')}

        **4. STRATEGIC IMPACT:** {s_node.metadata.get('overall_impact_analysis', 'N/A')}

        **5. UNCERTAINTIES & LIMITATIONS:** {s_node.metadata.get('general_credibility_risk', 'N/A')}

        **6. DIRECT EVIDENCE FROM FULL JUDGMENT:**
        {evidence_text if evidence_text else "No specific quotes found for this specific query."}
        """
        precedent_blocks.append(block)

        if Config.ENV == "dev": 
            format_and_log(
                "/answer_case_question_withuploadFile",
                f"Compiled precedent block for {case_name}",
                "precedent_block",
                {
                    "case_name": case_name,
                    "rerank_score": s_node.metadata.get("rerank_score"),
                    "full_facts": s_node.metadata.get("full_facts"),
                    "overall_impact_analysis": s_node.metadata.get("overall_impact_analysis"),
                    "general_credibility_risk": s_node.metadata.get("general_credibility_risk"),
                    "outcome_orders": s_node.metadata.get("outcome_orders"),
                    "reasons_rationale": s_node.metadata.get("reasons_rationale"),
                    "evidence_nodes_count": len(evidence_nodes),
                }
            )
        """
          now the case summary embeddings are built by each section of summary_sections:
            {
            "case_id": "FamCA_2018_257",
            "summary_sections": {
                "facts": "- Fact: The respondent commenced Supreme Court proceedings seeking possession of her property at Suburb B, where costs orders were made and were pending assessment in that jurisdiction.\n- Fact: The applicant filed a Family Court initiating application on 18 August 2015 seeking a declaration of a de facto relationship and relief under s 90SM of the Family Law Act 1975 (Cth).\n- Fact: On 28 October 2015 the respondent sought dismissal of the Family Court application.\n- Fact: In February 2016 the respondent made a written offer to settle both the Supreme Court and Family Court proceedings for $200,000; the applicant did not accept the offer.\n- Fact: On 27 March 2017 Justice Watts made costs orders against the applicant in relation to injunctive relief sought by the applicant.\n- Fact: On 27 February 2018 the Family Court application was dismissed for want of jurisdiction.\n- Fact: The respondent then applied for costs, initially seeking indemnity costs (and alternatives including party-party costs) and sought orders to secure payment of costs.\n- Fact: Evidence referred to in the costs hearing indicated the respondent had limited income (pension) and significant assets after sale of the Suburb B property.\n- Fact: The applicant provided no direct evidence of current financial circumstances in the costs application; material from the proceedings indicated limited income and that he undertook professional work.\n- Fact: The applicant had assets formerly tied to a property at L Town and admitted it had been sold; he asserted sale price was $300,000 and that he held approximately $150,000-$200,000 cash after paying mortgage and expenses.",
                "issues": "- Issue: Costs following dismissal of de facto property/jurisdiction application (s 90SM) for want of jurisdiction\n- Issue: Whether costs should be indemnity costs or party-party costs\n- Issue: Whether costs should be fixed in a lump sum or assessed\n- Issue: Whether costs should include costs of the costs application\n- Issue: Security/enforcement-type measures to secure payment of costs (deposit into trust, restraint on overseas travel, injunctions re dealing with property sale proceeds)\n- Issue: Certification for senior counsel",
                "outcome_orders": "- Outcome: The applicant to pay the respondent’s costs of and incidental to the proceedings commenced by initiating application filed 18 August 2015 on a party-party basis, as agreed or assessed, within 14 days of agreement or assessment.\n- Outcome: The costs order excludes the part of the proceedings relating to injunction proceedings before Justice Watts for which a costs order had already been made.\n- Outcome: Mr Brownless to deposit $127,000 into the trust account of Blackman Legal Pty Ltd by 4:00pm Monday 23 April 2018.\n- Outcome: Mr Brownless restrained from leaving Australia pending compliance with the costs payment order and the deposit order.\n- Outcome: Request that the Australian Federal Police place Mr Brownless on the Airport Watch List until the Court orders removal (to give effect to the travel restraint).\n- Outcome: Respondent’s solicitor to advise chambers when Mr Brownless complies with the deposit order, at which point the travel restraint and watch list request to be discharged in chambers.\n- Outcome: Injunction restraining the applicant from selling, transferring, encumbering, alienating or adversely dealing with his interest in the L Town property pending compliance with the deposit order and until further order.\n- Outcome: Mr S restrained from doing any act causing monies to be paid to Mr Brownless under any contract for sale of the L Town property pending Mr Brownless’s compliance with the deposit order.\n- Outcome: The injunctions relating to the L Town property contingent on the respondent filing an undertaking as to damages by 2:00pm on 20 April 2018; failure to lodge results in immediate discharge of those injunctions.\n- Outcome: A sealed copy of the orders to be served on Mr S by post to specified addresses and by SMS to a specified number.",
                "reasons_rationale": "- Reasons: The Court began from the statutory starting point under s 117 that each party bears their own costs, but found it appropriate to order costs against the applicant due to the combination of complete lack of success and rejection of a settlement offer that would likely have produced a better outcome for him than dismissal in both jurisdictions.\n- Reasons: Indemnity costs were refused: the Court accepted that indemnity costs require exceptional circumstances and found that imprudence or mere failure is insufficient; the Court was not persuaded the applicant’s case was objectively hopeless from the outset given the evaluative nature of de facto relationship criteria.\n- Reasons: The Court distinguished between the applicant’s conduct toward the respondent and his conduct of the litigation, and confined the costs consequence to party-party costs as the appropriate standard.\n- Reasons: A fixed sum was declined because the evidence did not permit the Court to be confident in the proposed quantification, even if the figure might ultimately be reasonable.\n- Reasons: Costs of the costs application were awarded to the respondent despite the failure to obtain indemnity costs, due to the strong connection with the respondent’s success overall.\n- Reasons: Protective/enforcement measures were made to secure payment, including a substantial trust deposit order, travel restraint/watch list request, and interim injunctions concerning dealing with the L Town property and sale proceeds, with an undertaking-as-to-damages condition for the property-related injunctions.\n- Reasons: The Court certified senior counsel as justified due to factual difficulty and the need to weigh competing inferences in the jurisdiction question, and the desirability of continuity.",
                "impact_analysis": "- Pivotal Finding: The applicant was wholly unsuccessful because the Family Court application was dismissed for want of jurisdiction.\n- Pivotal Finding: A written settlement offer to resolve both jurisdictions’ proceedings for $200,000 was made and not accepted; the judge considered acceptance would have been materially more favourable to the applicant than the outcome.\n- Pivotal Finding: The applicant’s conduct of the proceedings justified a departure from the usual costs position, but did not justify indemnity costs.\n- Pivotal Finding: The Court was not satisfied that the applicant objectively should have known he had no prospects, noting indicia of a de facto relationship may have been present and the test involves weighing criteria rather than a bright-line rule.\n- Pivotal Finding: The Court was not satisfied the proceedings were brought for an ulterior motive or in wilful disregard of known facts.\n- Pivotal Finding: A fixed lump sum costs order was not made because there was insufficient material to assess whether the proposed amount was logical, fair and reasonable.\n- Pivotal Finding: The respondent was awarded costs of the costs application notwithstanding partial failure (indemnity costs refused), due to the close connection between the costs application and the respondent’s overall success.\n- Pivotal Finding: Certification for senior counsel was granted due to factual difficulty and the need to weigh competing inferences in the jurisdiction dispute, and for continuity of representation.\n- Statutory Pivot: Family Law Act 1975 (Cth) s 117 (costs discretion; starting point each party bear own costs)\n- Statutory Pivot: Family Law Act 1975 (Cth) s 90SM (de facto financial cause provision referenced as the basis of the dismissed application)\n- Statutory Pivot: Family Law Act 1975 (Cth) s 121(9)(g) (approval to publish under pseudonym noted in the reasons)",
                "uncertainties": "- Uncertainties: The reasons state that financial circumstances evidence was limited for the purpose of the costs application; the applicant did not give direct evidence of his current circumstances.\n- Uncertainties: The Court did not determine or state the final quantum of costs payable, as costs were ordered on a party-party basis to be agreed or assessed.\n- Uncertainties: The material did not allow the Court to fix costs in a lump sum; the sufficiency and detail of costs evidence was a limiting factor.\n- Uncertainties: The underlying factual basis for the jurisdiction dismissal (why jurisdiction failed) is not detailed in these ex tempore costs reasons beyond being 'want of jurisdiction'."
            }
            }  
        """
    if topic == "property_division":
        topic_instruction = "Apply the 'Four-Step Process' (Pool, Contributions, s 75(2) Future Needs, and Just & Equitable)."
    elif topic == "children_parenting":
        topic_instruction = "Apply the 'Best Interests of the Child' framework (Section 60CC), focusing on safety, developmental needs, and the benefit of a relationship with both parents."
    elif topic == "spousal_maintenance":
        topic_instruction = "Apply the 'Threshold Test' (Section 72): One party's inability to support themselves vs. the other party's capacity to pay."
    else:
        topic_instruction = "Assess the situation based on the relevant sections of the Family Law Act 1975."
        # 4. FINAL SYNTHESIS PROMPT
    formatted_statutes = []
    for n in statute_nodes:
        raw_text = n.get_text()
        # Try metadata keys first, then extract from text as fallback
        section = (
            n.metadata.get("section_id")
            or n.metadata.get("section")
            or n.metadata.get("Section")
            or ""
        )
        title = (
            n.metadata.get("title")
            or n.metadata.get("Title")
            or n.metadata.get("header")
            or ""
        )
        # Fallback: parse section & title from the node text itself
        # e.g. "Family Law Act 1975 — Part VIII—Property... — s 79 Alteration of property interests"
        if not section or not title:
            import re
            # Try to extract "s 79", "s 60CC" etc.
            sec_match = re.search(r'\bs\s*(\d+[A-Z]*(?:\([^)]*\))?)\b', raw_text)
            if sec_match and not section:
                section = f"s {sec_match.group(1)}"
            # Try to extract the title after the section number
            title_match = re.search(r'\bs\s*\d+[A-Z]*(?:\([^)]*\))?\s+(.+?)(?:\n|$)', raw_text)
            if title_match and not title:
                title = title_match.group(1).strip()[:120]
            if not title:
                # Use first line as title
                first_line = raw_text.split('\n')[0].strip()
                title = first_line[:120] if first_line else "Statute"

        formatted_statutes.append({
            "section": section,
            "title": title,
            "text": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
            "score": round(n.score, 2) if n.score is not None else None,
        })

    if Config.ENV == "dev":
        format_and_log(
            "/answer_case_question_withuploadFile",
            "Formatted statutes for response",
            "formatted_statutes",
            formatted_statutes,
        )
    precedent_context = "\n\n" + "="*30 + "\n\n".join(precedent_blocks)
    prompt = f"""
        ROLE: Senior Australian Family Law Specialist.

        ### 1. STATUTORY BASIS
        {context_statutes}

        ### 2. CLIENT'S CURRENT PROFILE (From Upload)
        - **Facts:** {case_section_text}
        - **Impact Analysis:** {user_impact_text}
        - **Identified Uncertainties:** {user_uncert_text}

        ### 3. CHAT HISTORY CONTEXT
        {history_text}

        ### 4. RELEVANT AUSTLII PRECEDENTS & JUDICIAL EVIDENCE
        {precedent_context}

        ### 5. USER QUESTION: 
        {question}

        ---

        ### INSTRUCTIONS:
        Provide a comprehensive legal analysis using the following structured format. Ensure you integrate the 'Direct Evidence' from the precedents to provide a high-fidelity answer.

        ## Direct Answer
        Provide a concise summary of the legal position addressing the user's question directly.

        ## Analysis of Similar Decided Cases
        For each AustLII precedent provided above:
        - **Judicial Reasoning:** Explain how the judge linked facts to the legal outcome using the provided 'REASONS' and 'STRATEGIC IMPACT'.
        - **Evidence-Based Support:** Cite specific details from the 'DIRECT EVIDENCE FROM JUDGMENT' to validate why this case is a relevant benchmark for the client.

        ## Likely Assessment & Range of Outcomes
        - {topic_instruction}
        - Predict the likely range of outcomes (percentages, specific orders, or arrangements) based on the most similar 'FINAL ORDERS' in the precedents.
        - **CRITICAL: Any predicted percentage range MUST be narrow — no more than 10 percentage points wide** (e.g., "55% to 62%", NOT "50% to 70%"). A wider range indicates insufficient analysis. If you cannot narrow it to within 10 points, identify exactly which missing facts prevent a tighter prediction and state your best estimate with caveats.
        - Present ranges for each party that are consistent (e.g., Wife: 58%–65%, Husband: 35%–42%).
        - Justify the range by highlighting specific similarities or differences between the client's facts and the decided cases that push the prediction toward one end of the range.

        ## Critical Risks & Missing Information
        - Compare the client's 'UNCERTAINTIES' with those found in the precedents.
        - Identify specific missing facts that, if clarified, would most significantly shift this prediction.

        ---CACHE_SUMMARY---
        [Technical summary for system memory: Include statutory sections applied, and the specific outcome range predicted.]
        """
    if Config.ENV == "dev": 
        format_and_log(
                "/answer_case_question_withuploadFile",
                "Final synthesis prompt",
                "final_prompt",
                prompt
        )

    # Use a dedicated LLM for the final synthesis prompt.
    # The long, multi-source prompt benefits from a model with strong reasoning
    # and a large context window, which may differ from the embedding/summary LLM.
    synthesis_llm = Settings.llm  # default fallback
    synthesis_model = Config.SYNTHESIS_LLM  # e.g. "openai", "anthropic", or None/"gemini"

    if synthesis_model == "openai" and OpenAILLM is not None:
        try:
            synthesis_llm = OpenAILLM(
                model=Config.SYNTHESIS_OPENAI_LLM_MODEL,
                temperature=0.1,
                timeout=300.0,
                max_retries=2,
            )
            logger.info(f"Using OpenAI ({synthesis_llm.model}) for final synthesis.")
        except Exception as e:
            logger.warning(f"Failed to initialise OpenAI synthesis LLM ({e}); falling back to default.")
    elif synthesis_model == "anthropic" and AnthropicLLM is not None:
        try:
            synthesis_llm = AnthropicLLM(
                model=Config.SYNTHESIS_ANTHROPIC_LLM_MODEL,
                temperature=0.1,
                timeout=300.0,
                max_retries=2,
            )
            logger.info(f"Using Anthropic ({synthesis_llm.model}) for final synthesis.")
        except Exception as e:
            logger.warning(f"Failed to initialise Anthropic synthesis LLM ({e}); falling back to default.")

    try:
        response = await synthesis_llm.acomplete(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        # If Gemini blocks the prompt (safety/content filters), retry with OpenAI
        if any(kw in error_msg for kw in ("no candidates", "safety", "blocked", "google", "gemini")):
            if OpenAILLM is None:
                logger.error("Primary synthesis LLM failed and OpenAI fallback is not available (package not installed).")
                raise
            logger.warning(f"Primary synthesis LLM failed ({e}). Falling back to OpenAI model: {Config.SYNTHESIS_OPENAI_LLM_MODEL}")
            fallback_llm = OpenAILLM(
                model=Config.SYNTHESIS_OPENAI_LLM_MODEL,
                temperature=0.1,
                timeout=300.0,
                max_retries=2,
            )
            response = await fallback_llm.acomplete(prompt)
        else:
            raise

    return response.text, formatted_statutes



