"""Patent Agent orchestration service."""

from __future__ import annotations

from pathlib import Path

from patent_rag.agent.planner import PatentAgentPlanner, extract_patent_id
from patent_rag.agent.schemas import (
    AgentPlan,
    AgentRunResult,
    AgentToolStep,
    GraphSearchToolResult,
    PatentSearchToolResult,
    PatentSummaryToolResult,
)
from patent_rag.agent.tools import PatentAgentTools
from patent_rag.llm import ChatClient
from patent_rag.rag import RagAnswer
from patent_rag.rag.service import RetrievalMode
from patent_rag.retrieval import SearchHit


class PatentAgentService:
    """Plan and execute multi-tool patent Agent workflows."""

    def __init__(
        self,
        *,
        planner: PatentAgentPlanner | None = None,
        tools: PatentAgentTools | None = None,
        patents_path: Path = Path("data") / "processed" / "patents.jsonl",
        chunks_path: Path = Path("data") / "processed" / "chunks.jsonl",
        index_path: Path = Path("data") / "indexes" / "chroma",
        graph_path: Path = Path("data") / "graph" / "patent_graph.json",
        collection_name: str = "patent_chunks",
        chat_client: ChatClient | None = None,
        embedding_provider: str | None = None,
        embedding_model_name: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        self.planner = planner or PatentAgentPlanner()
        self.tools = tools or PatentAgentTools(
            patents_path=patents_path,
            chunks_path=chunks_path,
            index_path=index_path,
            graph_path=graph_path,
            collection_name=collection_name,
            chat_client=chat_client,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            embedding_dimension=embedding_dimension,
        )

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "hybrid",
        use_graph: bool = True,
        graph_top_k: int = 3,
    ) -> AgentRunResult:
        """Run the planned Agent workflow."""

        plan = self.planner.plan(query)
        steps: list[AgentToolStep] = []
        search_result: PatentSearchToolResult | None = None
        summary_result: PatentSummaryToolResult | None = None
        graph_result: GraphSearchToolResult | None = None
        rag_answer: RagAnswer | None = None

        for planned_step in plan.steps:
            if planned_step.tool_name == "search_patents":
                search_result = self.tools.search_patents(
                    query,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                )
                steps.append(
                    AgentToolStep(
                        tool_name=planned_step.tool_name,
                        tool_input={
                            "query": query,
                            "top_k": top_k,
                            "retrieval_mode": retrieval_mode,
                        },
                        observation=_search_observation(search_result),
                        output=search_result.model_dump(mode="json"),
                    )
                )
                continue

            if planned_step.tool_name == "summarize_patent":
                identifier = _choose_summary_identifier(query, search_result)
                summary_result = self.tools.summarize_patent(identifier)
                steps.append(
                    AgentToolStep(
                        tool_name=planned_step.tool_name,
                        tool_input={"identifier": identifier},
                        observation=_summary_observation(summary_result),
                        output=summary_result.model_dump(mode="json"),
                    )
                )
                continue

            if planned_step.tool_name == "graph_search":
                keyword = _choose_graph_keyword(query, search_result, summary_result)
                graph_result = self.tools.graph_search(keyword, limit=max(graph_top_k, 1))
                steps.append(
                    AgentToolStep(
                        tool_name=planned_step.tool_name,
                        tool_input={"keyword": keyword, "limit": max(graph_top_k, 1)},
                        observation=_graph_observation(graph_result),
                        output=graph_result.model_dump(mode="json"),
                    )
                )
                continue

            if planned_step.tool_name == "graph_rag_answer":
                answer_query = _build_answer_query(query, plan, summary_result, graph_result)
                rag_answer = self.tools.graph_rag_answer(
                    answer_query,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    use_graph=use_graph,
                    graph_top_k=graph_top_k,
                )
                steps.append(
                    AgentToolStep(
                        tool_name=planned_step.tool_name,
                        tool_input={
                            "question": answer_query,
                            "top_k": top_k,
                            "retrieval_mode": retrieval_mode,
                            "use_graph": use_graph,
                            "graph_top_k": graph_top_k,
                        },
                        observation=_rag_observation(rag_answer),
                        output=rag_answer.model_dump(mode="json"),
                    )
                )

        if rag_answer is None:
            rag_answer = _fallback_answer(query, retrieval_mode, top_k, use_graph)

        return AgentRunResult.from_rag_answer(
            query=query,
            intent=plan.intent,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_graph=use_graph,
            plan=plan,
            steps=steps,
            rag_answer=rag_answer,
        )


def run_patent_agent(
    query: str,
    *,
    chat_client: ChatClient | None = None,
    patents_path: Path = Path("data") / "processed" / "patents.jsonl",
    chunks_path: Path = Path("data") / "processed" / "chunks.jsonl",
    index_path: Path = Path("data") / "indexes" / "chroma",
    graph_path: Path = Path("data") / "graph" / "patent_graph.json",
    collection_name: str = "patent_chunks",
    top_k: int = 5,
    retrieval_mode: RetrievalMode = "hybrid",
    use_graph: bool = True,
    graph_top_k: int = 3,
    embedding_provider: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
) -> AgentRunResult:
    """Convenience helper for one-shot Agent runs."""

    service = PatentAgentService(
        patents_path=patents_path,
        chunks_path=chunks_path,
        index_path=index_path,
        graph_path=graph_path,
        collection_name=collection_name,
        chat_client=chat_client,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
    )
    return service.run(
        query,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        use_graph=use_graph,
        graph_top_k=graph_top_k,
    )


def _choose_summary_identifier(
    query: str,
    search_result: PatentSearchToolResult | None,
) -> str:
    patent_id = extract_patent_id(query)
    if patent_id:
        return patent_id
    if search_result and search_result.hits:
        return search_result.hits[0].patent_id
    return query


def _choose_graph_keyword(
    query: str,
    search_result: PatentSearchToolResult | None,
    summary_result: PatentSummaryToolResult | None,
) -> str:
    if summary_result and summary_result.title:
        return summary_result.title
    if search_result and search_result.hits:
        return _top_hit_title_or_patent_id(search_result.hits[0])
    patent_id = extract_patent_id(query)
    if patent_id:
        return patent_id
    return query.strip()


def _top_hit_title_or_patent_id(hit: SearchHit) -> str:
    return hit.title if hit.title else hit.patent_id


def _build_answer_query(
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


def _search_observation(result: PatentSearchToolResult) -> str:
    if not result.hits:
        return "检索工具未找到相关 chunk。"
    top_hit = result.hits[0]
    return (
        f"检索到 {len(result.hits)} 条候选 chunk，最高命中为 "
        f"{top_hit.title} [{top_hit.patent_id}]。"
    )


def _summary_observation(result: PatentSummaryToolResult) -> str:
    if not result.found:
        return "未能从结构化专利数据中定位到可总结的专利。"
    return (
        f"已读取 {result.title} [{result.patent_id}]，包含 "
        f"{result.claim_count} 项权利要求和 {len(result.section_names)} 个章节。"
    )


def _graph_observation(result: GraphSearchToolResult) -> str:
    if not result.matches:
        return f"知识图谱中未找到与“{result.keyword}”直接关联的专利。"
    return f"知识图谱中找到 {len(result.matches)} 个相关专利节点。"


def _rag_observation(result: RagAnswer) -> str:
    return (
        f"GraphRAG 返回回答，包含 {len(result.sources)} 条原文证据和 "
        f"{len(result.graph_sources)} 条图谱证据。"
    )


def _fallback_answer(
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
