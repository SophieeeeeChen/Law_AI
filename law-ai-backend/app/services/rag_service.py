from ast import Dict
import json
from datetime import datetime
from pathlib import Path

from app.core.models import model_manager
from app.core.logger import logger
from app.core.config import Config
from llama_index.core import get_response_synthesizer, PromptTemplate, Settings
from app.services.summary_service import generate_summary_dict
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from app.services.clarify_service import TOPIC_KEYWORDS

try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception:
    try:
        from llama_index.core.retrievers import BM25Retriever
    except Exception:  # pragma: no cover - optional dependency
        BM25Retriever = None

try:
    from llama_index.core.postprocessor import LLMRerank
except Exception:  # pragma: no cover - optional dependency
    LLMRerank = None

try:
    from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
except Exception:  # pragma: no cover - optional dependency
    MetadataFilter = MetadataFilters = FilterOperator = None
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
        case_excerpt = case_section_text[:300] + "..." if len(case_section_text) > 300 else case_section_text
        components.append(f"Case context: {case_excerpt}")
    
    if history_text:
        history_summary = history_text[:200] + "..." if len(history_text) > 200 else history_text
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
    bm25_top_k: int | None = None,
    metadata_filters=None,
    limit: int | None = None, # Added a generic limit for easier use
):
    # Use the 'limit' if provided, otherwise fallback to Config
    v_k = limit or vector_top_k or Config.TOP_K
    b_k = limit or bm25_top_k or Config.BM25_TOP_K

    # 1. Vector Retrieval (Handles Filters natively)
    retriever_kwargs = {"similarity_top_k": v_k}
    if metadata_filters is not None:
        retriever_kwargs["filters"] = metadata_filters
    
    vector_retriever = index.as_retriever(**retriever_kwargs)
    vector_nodes = vector_retriever.retrieve(query)

    # 2. BM25 Retrieval
    bm25_nodes = []
    # FIX: Allow BM25 to run even if filters exist. 
    # Note: LlamaIndex BM25 doesn't always support complex filters natively 
    # as easily as Vector search, but we should at least try to get keyword matches.
    if BM25Retriever is not None and b_k > 0:
        try:
            # Check if docstore has documents before attempting BM25
            if hasattr(index, 'docstore') and hasattr(index.docstore, 'docs') and len(index.docstore.docs) > 0:
                bm25 = BM25Retriever.from_defaults(
                    docstore=index.docstore,
                    similarity_top_k=b_k,
                )
                bm25_nodes = bm25.retrieve(query)
                
                # OPTIONAL: If filters exist, manually filter BM25 results to match
                if metadata_filters:
                    bm25_nodes = _apply_manual_filter(bm25_nodes, metadata_filters)
            else:
                logger.debug("BM25Retriever skipped: docstore is empty.")
                
        except Exception as exc:
            logger.warning(f"BM25Retriever failed ({exc}); falling back to vector-only.")
    
    # 3. Merge and Rerank
    merged_nodes = _merge_nodes(vector_nodes, bm25_nodes)

    if use_rerank and LLMRerank is not None:
        reranker = LLMRerank(top_n=Config.RERANK_TOP_N)
        merged_nodes = reranker.postprocess_nodes(merged_nodes, query_str=query)
    else:
        # Standard weighted score merge
        vec_norm = _normalize_scores(vector_nodes)
        bm25_norm = _normalize_scores(bm25_nodes)
        for node in merged_nodes:
            nid = _node_id(node)
            node.score = (
                Config.HYBRID_VECTOR_WEIGHT * vec_norm.get(nid, 0.0)
                + Config.HYBRID_BM25_WEIGHT * bm25_norm.get(nid, 0.0)
            )
        merged_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

    return merged_nodes[:limit] if limit else merged_nodes

# to be updated
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
    
    #word_count = len(case_text.split())
    target_words = Config.USER_SUMMARY_TARGET_WORDS
    max_words = target_words + 200

    summary_obj = generate_summary_dict(
        case_text,
        target_words=target_words,
        max_words=max_words,
        raw_excerpt_chars=max_chars,
    )

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
    impact_analysis: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Final Triple-Thread RAG: Statutes + Strategic Impact + Full Text.
    Returns: (UI_Answer, Citations_List)
    """
    
    # 1. RETRIEVE STATUTES (The Legal Foundation)
    # We search the rules_statutes collection for black-letter law.
    statutes_index = model_manager.create_or_load_statutes_index()
    statute_nodes = _hybrid_retrieve(
        statutes_index, 
        f"{topic or ''} {question}", 
        use_rerank=True, 
        limit=3
    )
    context_statutes = "\n".join([f"- {n.get_text()}" for n in statute_nodes])

    # 2. RETRIEVE CASE SUMMARIES (The Strategic Child Search)
    # We search the summary collection to get the 'impact_analysis' metadata.
    summary_index = model_manager.create_or_load_case_summaries_index()
    summary_filters = None
    if topic:
        summary_filters = MetadataFilters(filters=[
            ExactMatchFilter(key="summary_section", value=topic)
        ])

    summary_query = _build_structured_query(question, case_section_text, history_text, topic)
    summary_nodes = _hybrid_retrieve(
        summary_index, 
        summary_query, 
        use_rerank=True, 
        metadata_filters=summary_filters, 
        limit=2
    )

    # 3. RETRIEVE DEEP PRECEDENT (The Parent Search)
    # Using case_ids found in summaries to pull granular details/analogies from full judgments.
    full_index = model_manager.create_or_load_cases_index()
    precedent_blocks = []
    citations = []

    # Add Legislation to Citations
    for n in statute_nodes:
        citations.append({
            "source": n.metadata.get("section_title", "Family Law Act 1975"),
            "type": "Legislation",
            "id": n.metadata.get("section_id")
        })

    for s_node in summary_nodes:
        cid = s_node.metadata.get("case_id")
        impact = s_node.metadata.get("impact_analysis", "Analyzing case significance...")
        reasons_rationale = s_node.metadata.get("reasons_rationale","No detailed reasoning available.")
        outcome_orders = s_node.metadata.get("outcome_orders", "No specific orders reported.")
        uncertainties = s_node.metadata.get("uncertainties", "No uncertainties reported.")
        # Pull details only for this case to avoid "hallucinating" facts between different cases
        case_filters = MetadataFilters(filters=[ExactMatchFilter(key="case_id", value=cid)])
        detail_nodes = _hybrid_retrieve(full_index, question, use_rerank=False, metadata_filters=case_filters, limit=2)
        
        detail_text = "\n".join([dn.get_text() for dn in detail_nodes])
        precedent_blocks.append(
            f"CASE: {s_node.metadata.get('case_name')}\n"
            f"STRATEGIC IMPACT: {impact}\n",
            f"REASONS & RATIONALE: {reasons_rationale}\n",
            f"OUTCOME/ORDERS: {outcome_orders}\n",
            f"FULL TEXT SNIPPET: {detail_text}"
        )
        citations.append({
            "source": s_node.metadata.get("case_name"),
            "type": "Case Law",
            "id": cid,
            "url": f"https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/FedCFamC1F/{cid}.html" # Example dynamic link
        })
    if topic == "property_division":
        topic_instruction = "Apply the 'Four-Step Process' (Pool, Contributions, s 75(2) Future Needs, and Just & Equitable)."
    elif topic == "children_parenting":
        topic_instruction = "Apply the 'Best Interests of the Child' framework (Section 60CC), focusing on safety, developmental needs, and the benefit of a relationship with both parents."
    elif topic == "spousal_maintenance":
        topic_instruction = "Apply the 'Threshold Test' (Section 72): One party's inability to support themselves vs. the other party's capacity to pay."
    else:
        topic_instruction = "Assess the situation based on the relevant sections of the Family Law Act 1975."
        # 4. FINAL SYNTHESIS PROMPT
    precedent_context = "\n\n---\n\n".join(precedent_blocks)
    impact_block = impact_analysis or  "No specific impact analysis provided for this case."
    prompt = f"""
    ROLE: Senior Australian Family Law Specialist.
    
    STATUTORY BASIS:
    {context_statutes}
    
    CLIENT'S CURRENT CASE FACTS (From Upload):
    {case_section_text}
    
    CLIENT'S CURRENT CASE IMPACT ANALYSIS:
    {impact_block}

    CHAT HISTORY CONTEXT:
    {history_text}
    
    RELEVANT AUSTLII PRECEDENTS & IMPACT ANALYSIS:
    {precedent_context}

    USER QUESTION: {question}
    
    INSTRUCTIONS:
    Provide a comprehensive legal analysis in the following structured format:
    
    ## Direct Answer
    Provide a concise summary of the legal position addressing the user's question directly.
    
    ## Similar Decided Cases
    For each AustLII precedent provided above:
    - Write a bullet point explaining the judicial reasoning
    - Show how the judge linked facts to a legal outcome
    - Explicitly mention the 'STRATEGIC IMPACT' from the metadata
    
    ## Likely Assessment
    - {topic_instruction}
    - Predict the likely range of outcomes based on the client's specific facts
    - Be specific about percentages, orders, or arrangements where appropriate
    
    ## Uncertainties & Missing Information
    Identify what facts are missing that would significantly shift this prediction.
    
    ---CACHE_SUMMARY---
    [Provide a technical summary of this advice for conversation memory]
    """

    response = await Settings.llm.acomplete(prompt)
    return response.text, citations


# def _node_case_id(node) -> str | None:
#     meta = getattr(node, "metadata", None) or {}
#     for key in ("case_id", "case_name"):
#         value = meta.get(key)
#         if value:
#             return str(value)
#     source = meta.get("source") or meta.get("source_file") or meta.get("path")
#     if source:
#         try:
#             return Path(str(source)).stem
#         except Exception:
#             return str(source)
#     return None


# def _aggregate_case_ids(nodes, max_cases: int) -> list[str]:
#     scores: dict[str, float] = {}
#     for node in nodes:
#         case_id = _node_case_id(node)
#         if not case_id:
#             continue
#         score = node.score if node.score is not None else 0.0
#         if case_id not in scores or score > scores[case_id]:
#             scores[case_id] = score
#     ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
#     return [case_id for case_id, _ in ranked[:max_cases]]


# def _format_nodes(nodes, source_type: str) -> list[dict]:
#     formatted = []
#     for node in nodes:
#         meta = node.metadata or {}
#         formatted.append(
#             {
#                 "file_name": meta.get("source") or meta.get("source_file") or meta.get("path") or "Unknown",
#                 "score": node.score,
#                 "content": node.get_text(),
#                 "source_type": source_type,
#                 "summary_section": meta.get("summary_section"),
#                 "case_id": _node_case_id(node),
#             }
#         )
#     return formatted


# def _dedupe_nodes(nodes):
#     unique = {}
#     for node in nodes:
#         unique[_node_id(node)] = node
#     return list(unique.values())


# def _topic_to_summary_section(topic: str | None) -> str | None:
#     if not topic:
#         return None
#     mapping = {
#         "property_division": "property_division",
#         "children_parenting": "children_parenting",
#         "spousal_maintenance": "spousal_maintenance",
#         "family_violence_safety": "family_violence_safety",
#         "prenup_postnup": "prenup_postnup",
#     }
#     return mapping.get(topic)


# def _filter_summary_nodes_by_section(nodes, section: str | None):
#     if not section:
#         return list(nodes)
#     filtered = [n for n in nodes if (n.metadata or {}).get("summary_section") == section]
#     return filtered or list(nodes)


# def _build_case_metadata_filters(case_ids: list[str]):
#     if not case_ids or MetadataFilters is None or MetadataFilter is None or FilterOperator is None:
#         return None
#     sources = [f"{case_id}.txt" for case_id in case_ids]
#     return MetadataFilters(
#         filters=[MetadataFilter(key="source", operator=FilterOperator.IN, value=sources)]
#     )


# def _build_summary_metadata_filters(case_ids: list[str], summary_section: str | None = None):
#     if not case_ids or MetadataFilters is None or MetadataFilter is None or FilterOperator is None:
#         return None
#     filters = [MetadataFilter(key="case_id", operator=FilterOperator.IN, value=case_ids)]
#     if summary_section:
#         filters.append(
#             MetadataFilter(key="summary_section", operator=FilterOperator.EQ, value=summary_section)
#         )
#     return MetadataFilters(filters=filters)


