"""Rule-based planning for the first patent Agent workflow."""

from __future__ import annotations

import re

from patent_rag.agent.schemas import AgentIntent, AgentPlan, AgentPlannedStep

PATENT_ID_PATTERN = re.compile(r"CN\s*\d+(?:\s*\d+)?\s*[A-Z]", re.IGNORECASE)

IDEA_TERMS = (
    "设计",
    "想法",
    "创意",
    "改进",
    "创新",
    "规避",
    "优化",
    "方案",
)
SUMMARY_TERMS = (
    "总结",
    "概括",
    "介绍",
    "说明",
    "这篇专利",
    "该专利",
)


class PatentAgentPlanner:
    """Create a deterministic plan before Agent tool execution."""

    def plan(self, query: str) -> AgentPlan:
        """Plan which tools should be used for a user request."""

        intent = classify_intent(query)
        if intent == "patent_summary":
            return AgentPlan(
                query=query,
                intent=intent,
                rationale="用户请求中包含专利号或总结类表达，需要先定位专利，再组织摘要和证据。",
                steps=[
                    AgentPlannedStep(
                        tool_name="search_patents",
                        reason="先检索相关专利，为后续摘要和问答提供候选对象。",
                    ),
                    AgentPlannedStep(
                        tool_name="summarize_patent",
                        reason="读取结构化专利数据，生成可解释的专利摘要观察结果。",
                    ),
                    AgentPlannedStep(
                        tool_name="graph_search",
                        reason="查询知识图谱中该专利或相关关键词的实体关系。",
                    ),
                    AgentPlannedStep(
                        tool_name="graph_rag_answer",
                        reason="综合原文证据和图谱证据生成最终回答。",
                    ),
                ],
            )

        if intent == "idea_analysis":
            return AgentPlan(
                query=query,
                intent=intent,
                rationale="用户提出了设计、改进或创意分析需求，需要先查现有方案，再生成启发式回答。",
                steps=[
                    AgentPlannedStep(
                        tool_name="search_patents",
                        reason="检索与用户想法最相关的现有专利片段。",
                    ),
                    AgentPlannedStep(
                        tool_name="graph_search",
                        reason="查询相关专利在知识图谱中的关键词和实体关系。",
                    ),
                    AgentPlannedStep(
                        tool_name="graph_rag_answer",
                        reason="基于现有专利证据，分析可参考方案和改进方向。",
                    ),
                ],
            )

        return AgentPlan(
            query=query,
            intent=intent,
            rationale="用户请求更接近专利知识问答，直接执行检索、图谱查询和 GraphRAG 回答。",
            steps=[
                AgentPlannedStep(
                    tool_name="search_patents",
                    reason="检索能够回答问题的专利原文片段。",
                ),
                AgentPlannedStep(
                    tool_name="graph_search",
                    reason="查询与问题或命中专利相关的图谱关系。",
                ),
                AgentPlannedStep(
                    tool_name="graph_rag_answer",
                    reason="把原文证据和图谱证据合并后生成回答。",
                ),
            ],
        )


def classify_intent(query: str) -> AgentIntent:
    """Classify a patent task using transparent rules."""

    normalized = query.strip()
    if PATENT_ID_PATTERN.search(normalized) or any(term in normalized for term in SUMMARY_TERMS):
        return "patent_summary"
    if any(term in normalized for term in IDEA_TERMS):
        return "idea_analysis"
    return "patent_qa"


def extract_patent_id(query: str) -> str | None:
    """Extract a normalized Chinese patent id from text."""

    match = PATENT_ID_PATTERN.search(query)
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(0)).upper()
