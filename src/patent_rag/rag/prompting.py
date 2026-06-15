"""Prompt construction for patent RAG answers."""

from __future__ import annotations

from patent_rag.llm import ChatMessage
from patent_rag.rag.graph_context import GraphEvidence
from patent_rag.retrieval import SearchHit

SYSTEM_PROMPT = """你是一个严谨的中文专利知识库助手。
你必须只依据用户提供的专利证据回答问题。
如果证据不足以支持结论，请明确说明“当前证据不足”。
回答中需要引用证据编号，例如 [S1]、[S2]、[G1]。
不要编造专利号、申请人、技术效果或未出现在证据中的细节。
图谱证据 [G] 只能用于说明实体关系、关联关键词、申请人、IPC、章节和权利要求数量；
涉及具体技术方案和效果时，优先引用原文证据 [S]。
"""


def build_rag_messages(
    question: str,
    hits: list[SearchHit],
    graph_evidence: list[GraphEvidence] | None = None,
) -> list[ChatMessage]:
    """Build chat messages for a source-grounded patent RAG answer."""

    evidence_context = render_evidence_context(hits)
    graph_context = render_graph_evidence_context(graph_evidence or [])
    user_prompt = f"""用户问题：
{question}

专利原文证据：
{evidence_context}

知识图谱证据：
{graph_context}

请基于以上证据回答用户问题。要求：
1. 先给出直接结论。
2. 说明相关专利和技术方案。
3. 使用 [S1]、[G1] 这样的证据编号标注来源。
4. 如果证据不足，请明确指出不足之处。
"""
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]


def render_evidence_context(hits: list[SearchHit], max_snippet_chars: int = 700) -> str:
    """Render retrieved chunks as numbered evidence blocks."""

    if not hits:
        return "未检索到相关专利证据。"

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        claim_text = f"权利要求{hit.claim_number}" if hit.claim_number is not None else hit.section
        snippet = hit.snippet.strip().replace("\r\n", "\n")
        if len(snippet) > max_snippet_chars:
            snippet = f"{snippet[:max_snippet_chars]}..."
        blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"专利：{hit.title}",
                    f"专利号：{hit.patent_id}",
                    f"位置：{claim_text}",
                    f"相关性分数：{hit.score}",
                    f"证据内容：{snippet}",
                ]
            )
        )
    return "\n\n".join(blocks)


def render_graph_evidence_context(
    graph_evidence: list[GraphEvidence],
    max_relation_chars: int = 500,
) -> str:
    """Render graph neighborhoods as numbered evidence blocks."""

    if not graph_evidence:
        return "未检索到相关知识图谱证据。"

    blocks: list[str] = []
    for evidence in graph_evidence:
        relation_summary = evidence.relation_summary.strip()
        if len(relation_summary) > max_relation_chars:
            relation_summary = f"{relation_summary[:max_relation_chars]}..."
        matched_terms = "、".join(evidence.matched_terms) if evidence.matched_terms else "无"
        supporting_chunks = (
            "、".join(evidence.supporting_chunk_ids[:5])
            if evidence.supporting_chunk_ids
            else "无"
        )
        blocks.append(
            "\n".join(
                [
                    f"[{evidence.source_id}]",
                    f"专利：{evidence.title}",
                    f"专利号：{evidence.patent_id}",
                    f"图谱相关性分数：{evidence.score}",
                    f"问题匹配词：{matched_terms}",
                    f"关联原文 chunk：{supporting_chunks}",
                    f"图谱关系：{relation_summary}",
                ]
            )
        )
    return "\n\n".join(blocks)
