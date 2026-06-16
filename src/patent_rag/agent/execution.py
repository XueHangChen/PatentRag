"""Shared Agent execution helpers."""

from __future__ import annotations

from patent_rag.agent.planner import extract_patent_id
from patent_rag.agent.schemas import (
    AgentPlan,
    GraphSearchToolResult,
    PatentSearchToolResult,
    PatentSummaryToolResult,
)
from patent_rag.rag import RagAnswer
from patent_rag.rag.service import RetrievalMode
from patent_rag.retrieval import SearchHit


def choose_summary_identifier(
    query: str,
    search_result: PatentSearchToolResult | None,
) -> str:
    patent_id = extract_patent_id(query)
    if patent_id:
        return patent_id
    if search_result and search_result.hits:
        return search_result.hits[0].patent_id
    return query


def choose_graph_keyword(
    query: str,
    search_result: PatentSearchToolResult | None,
    summary_result: PatentSummaryToolResult | None,
) -> str:
    if summary_result and summary_result.title:
        return summary_result.title
    if search_result and search_result.hits:
        return top_hit_title_or_patent_id(search_result.hits[0])
    patent_id = extract_patent_id(query)
    if patent_id:
        return patent_id
    return query.strip()


def top_hit_title_or_patent_id(hit: SearchHit) -> str:
    return hit.title if hit.title else hit.patent_id


def build_answer_query(
    query: str,
    plan: AgentPlan,
    summary_result: PatentSummaryToolResult | None,
    graph_result: GraphSearchToolResult | None,
) -> str:
    context_lines = [query]
    if plan.intent == "idea_analysis":
        context_lines.append("请重点分析现有专利方案、可借鉴结构和潜在改进方向。")
    if summary_result and summary_result.found:
        context_lines.append(f"已定位专利：{summary_result.title} [{summary_result.patent_id}]。")
    if graph_result and graph_result.matches:
        titles = "；".join(
            f"{match.title} [{match.patent_id}]"
            for match in graph_result.matches[:3]
        )
        context_lines.append(f"知识图谱关联专利：{titles}。")
    return "\n".join(context_lines)


def search_observation(result: PatentSearchToolResult) -> str:
    if not result.hits:
        return "检索工具未找到相关 chunk。"
    top_hit = result.hits[0]
    return (
        f"检索到 {len(result.hits)} 条候选 chunk，最高命中为 "
        f"{top_hit.title} [{top_hit.patent_id}]。"
    )


def summary_observation(result: PatentSummaryToolResult) -> str:
    if not result.found:
        return "未能从结构化专利数据中定位到可总结的专利。"
    return (
        f"已读取 {result.title} [{result.patent_id}]，包含 "
        f"{result.claim_count} 项权利要求和 {len(result.section_names)} 个章节。"
    )


def graph_observation(result: GraphSearchToolResult) -> str:
    if not result.matches:
        return f"知识图谱中未找到与“{result.keyword}”直接关联的专利。"
    return f"知识图谱中找到 {len(result.matches)} 个相关专利节点。"


def rag_observation(result: RagAnswer) -> str:
    return (
        f"GraphRAG 返回回答，包含 {len(result.sources)} 条原文证据和 "
        f"{len(result.graph_sources)} 条图谱证据。"
    )


def fallback_answer(
    query: str,
    retrieval_mode: RetrievalMode,
    top_k: int,
    use_graph: bool,
) -> RagAnswer:
    return RagAnswer(
        question=query,
        answer="Agent 未执行到最终问答工具，因此没有生成可用答案。",
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        use_graph=use_graph,
    )
