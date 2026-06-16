"""LangGraph-backed patent Agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from langgraph.graph import END, StateGraph

from patent_rag.agent.execution import (
    build_answer_query,
    choose_graph_keyword,
    choose_summary_identifier,
    fallback_answer,
    graph_observation,
    rag_observation,
    search_observation,
    summary_observation,
)
from patent_rag.agent.langgraph_state import (
    AgentState,
    append_node_trace,
    new_checkpoint_id,
    snapshot_graph_state,
)
from patent_rag.agent.llm_planner import LlmIntentPlanner
from patent_rag.agent.planner import PatentAgentPlanner
from patent_rag.agent.replanner import AgentReplanner, RuleOnlyReplanner
from patent_rag.agent.schemas import (
    AgentPlannedStep,
    AgentRunResult,
    AgentToolStep,
)
from patent_rag.agent.tools import PatentAgentTools
from patent_rag.config import get_settings
from patent_rag.llm import ChatClient, create_chat_client
from patent_rag.rag.service import RetrievalMode


class LangGraphPatentAgentService:
    """Run patent Agent workflows through a typed LangGraph state graph."""

    def __init__(
        self,
        *,
        planner: PatentAgentPlanner | None = None,
        replanner: AgentReplanner | None = None,
        tools: PatentAgentTools | None = None,
        patents_path: Path = Path("data") / "processed" / "patents.jsonl",
        chunks_path: Path = Path("data") / "processed" / "chunks.jsonl",
        index_path: Path = Path("data") / "indexes" / "chroma",
        graph_path: Path | None = None,
        collection_name: str = "patent_chunks",
        chat_client: ChatClient | None = None,
        embedding_provider: str | None = None,
        embedding_model_name: str | None = None,
        embedding_dimension: int | None = None,
        llm_planner: LlmIntentPlanner | None = None,
        enable_llm_planner: bool | None = None,
        llm_planner_min_confidence: float | None = None,
    ) -> None:
        settings = get_settings()
        self.enable_llm_planner = (
            bool(llm_planner) or settings.enable_llm_planner
            if enable_llm_planner is None
            else enable_llm_planner
        )
        self.llm_planner_min_confidence = (
            settings.llm_planner_min_confidence
            if llm_planner_min_confidence is None
            else llm_planner_min_confidence
        )
        self.llm_planner_error: str | None = None
        self.llm_planner = llm_planner or self._build_llm_planner(chat_client)
        self.planner = planner or PatentAgentPlanner()
        self.replanner = replanner or RuleOnlyReplanner()
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
        self.graph = self._build_graph()

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "hybrid",
        use_graph: bool = True,
        graph_top_k: int = 3,
    ) -> AgentRunResult:
        """Run the Agent workflow and preserve the legacy public response shape."""

        initial_state: AgentState = {
            "query": query,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "use_graph": use_graph,
            "graph_top_k": graph_top_k,
            "planner_mode": "rule",
            "pending_steps": [],
            "completed_steps": [],
            "errors": [],
            "node_trace": [],
            "planning_metadata": {"source": "rule", "fallback_used": False},
            "runtime_checkpoint_id": new_checkpoint_id(),
        }
        final_state = self.graph.invoke(initial_state)
        plan = final_state["plan"]
        rag_answer = final_state.get("rag_answer") or fallback_answer(
            query,
            retrieval_mode,
            top_k,
            use_graph,
        )
        return AgentRunResult.from_rag_answer(
            query=query,
            intent=plan.intent,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_graph=use_graph,
            plan=plan,
            steps=final_state.get("completed_steps", []),
            rag_answer=rag_answer,
            planner_mode=final_state.get("planner_mode", "rule"),
            node_trace=final_state.get("node_trace", []),
            graph_state=snapshot_graph_state(final_state),
            checkpoint_id=final_state.get("runtime_checkpoint_id"),
        )

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("rule_plan", self._rule_plan)
        workflow.add_node("llm_plan", self._llm_plan)
        workflow.add_node("llm_replan", self._llm_replan)
        workflow.add_node("execute_search", self._execute_search)
        workflow.add_node("execute_summary", self._execute_summary)
        workflow.add_node("execute_graph_search", self._execute_graph_search)
        workflow.add_node("execute_graph_rag", self._execute_graph_rag)
        workflow.add_node("self_check", self._self_check)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "rule_plan")
        workflow.add_edge("rule_plan", "llm_plan")
        workflow.add_edge("llm_plan", "llm_replan")
        workflow.add_conditional_edges(
            "llm_replan",
            self._route_next_tool,
            {
                "execute_search": "execute_search",
                "execute_summary": "execute_summary",
                "execute_graph_search": "execute_graph_search",
                "execute_graph_rag": "execute_graph_rag",
                "self_check": "self_check",
            },
        )
        for node in ("execute_search", "execute_summary", "execute_graph_search"):
            workflow.add_conditional_edges(
                node,
                self._route_next_tool,
                {
                    "execute_search": "execute_search",
                    "execute_summary": "execute_summary",
                    "execute_graph_search": "execute_graph_search",
                    "execute_graph_rag": "execute_graph_rag",
                    "self_check": "self_check",
                },
            )
        workflow.add_edge("execute_graph_rag", "self_check")
        workflow.add_edge("self_check", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _initialize(self, state: AgentState) -> AgentState:
        return self._with_trace(
            state,
            "initialize",
            lambda active_state: {
                **active_state,
                "completed_steps": list(active_state.get("completed_steps", [])),
                "errors": list(active_state.get("errors", [])),
                "node_trace": list(active_state.get("node_trace", [])),
            },
        )

    def _rule_plan(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            plan = self.planner.plan(active_state["query"])
            return {
                **active_state,
                "plan": plan,
                "rule_plan": plan,
                "pending_steps": list(plan.steps),
                "planning_metadata": {"source": "rule", "fallback_used": False},
            }

        return self._with_trace(state, "rule_plan", run)

    def _llm_plan(self, state: AgentState) -> AgentState:
        started = perf_counter()
        if not self.enable_llm_planner or self.llm_planner is None:
            details = {
                **self._trace_details(state),
                "enabled": self.enable_llm_planner,
                "fallback_used": True,
            }
            if self.llm_planner_error:
                details["skip_reason"] = self.llm_planner_error
            return append_node_trace(
                state,
                node_name="llm_plan",
                status="skipped",
                details=details,
                latency_ms=(perf_counter() - started) * 1000,
            )

        def run(active_state: AgentState) -> AgentState:
            result = self.llm_planner.plan(active_state["query"])
            metadata = dict(result.metadata)
            next_state: AgentState = {
                **active_state,
                "llm_plan_result": result,
                "planning_metadata": metadata,
            }
            if result.accepted and result.plan is not None:
                metadata["fallback_used"] = False
                next_state["plan"] = result.plan
                next_state["pending_steps"] = list(result.plan.steps)
                next_state["planner_mode"] = "llm"
                next_state["planning_metadata"] = metadata
                return next_state

            metadata["fallback_used"] = True
            metadata["rejection_reason"] = result.rejection_reason or "llm_plan_rejected"
            next_state["planning_metadata"] = metadata
            return next_state

        return self._with_trace(state, "llm_plan", run)

    def _llm_replan(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            decision = self.replanner.replan(active_state)
            planner_mode = decision.planner_mode
            if active_state.get("planner_mode") == "llm" and planner_mode == "rule":
                planner_mode = "llm"
            return {
                **active_state,
                "planner_mode": planner_mode,
                "pending_steps": list(decision.steps),
            }

        return self._with_trace(state, "llm_replan", run)

    def _execute_search(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            step, next_state = self._pop_next_step(active_state, "search_patents")
            if step is None:
                return next_state
            result = self.tools.search_patents(
                active_state["query"],
                top_k=active_state["top_k"],
                retrieval_mode=active_state["retrieval_mode"],
            )
            return {
                **next_state,
                "search_result": result,
                "completed_steps": [
                    *next_state.get("completed_steps", []),
                    AgentToolStep(
                        tool_name=step.tool_name,
                        tool_input={
                            "query": active_state["query"],
                            "top_k": active_state["top_k"],
                            "retrieval_mode": active_state["retrieval_mode"],
                        },
                        observation=search_observation(result),
                        output=result.model_dump(mode="json"),
                    ),
                ],
            }

        return self._tool_node(state, "execute_search", "search_patents", run)

    def _execute_summary(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            step, next_state = self._pop_next_step(active_state, "summarize_patent")
            if step is None:
                return next_state
            identifier = choose_summary_identifier(
                active_state["query"],
                active_state.get("search_result"),
            )
            result = self.tools.summarize_patent(identifier)
            return {
                **next_state,
                "summary_result": result,
                "completed_steps": [
                    *next_state.get("completed_steps", []),
                    AgentToolStep(
                        tool_name=step.tool_name,
                        tool_input={"identifier": identifier},
                        observation=summary_observation(result),
                        output=result.model_dump(mode="json"),
                    ),
                ],
            }

        return self._tool_node(state, "execute_summary", "summarize_patent", run)

    def _execute_graph_search(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            step, next_state = self._pop_next_step(active_state, "graph_search")
            if step is None:
                return next_state
            keyword = choose_graph_keyword(
                active_state["query"],
                active_state.get("search_result"),
                active_state.get("summary_result"),
            )
            result = self.tools.graph_search(keyword, limit=max(active_state["graph_top_k"], 1))
            return {
                **next_state,
                "graph_result": result,
                "completed_steps": [
                    *next_state.get("completed_steps", []),
                    AgentToolStep(
                        tool_name=step.tool_name,
                        tool_input={
                            "keyword": keyword,
                            "limit": max(active_state["graph_top_k"], 1),
                        },
                        observation=graph_observation(result),
                        output=result.model_dump(mode="json"),
                    ),
                ],
            }

        return self._tool_node(state, "execute_graph_search", "graph_search", run)

    def _execute_graph_rag(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            step, next_state = self._pop_next_step(active_state, "graph_rag_answer")
            if step is None:
                return next_state
            plan = active_state["plan"]
            question = build_answer_query(
                active_state["query"],
                plan,
                active_state.get("summary_result"),
                active_state.get("graph_result"),
            )
            result = self.tools.graph_rag_answer(
                question,
                top_k=active_state["top_k"],
                retrieval_mode=active_state["retrieval_mode"],
                use_graph=active_state["use_graph"],
                graph_top_k=active_state["graph_top_k"],
            )
            return {
                **next_state,
                "rag_answer": result,
                "answer": result.answer,
                "sources": result.sources,
                "graph_sources": result.graph_sources,
                "completed_steps": [
                    *next_state.get("completed_steps", []),
                    AgentToolStep(
                        tool_name=step.tool_name,
                        tool_input={
                            "question": question,
                            "top_k": active_state["top_k"],
                            "retrieval_mode": active_state["retrieval_mode"],
                            "use_graph": active_state["use_graph"],
                            "graph_top_k": active_state["graph_top_k"],
                        },
                        observation=rag_observation(result),
                        output=result.model_dump(mode="json"),
                    ),
                ],
            }

        return self._tool_node(state, "execute_graph_rag", "graph_rag_answer", run)

    def _self_check(self, state: AgentState) -> AgentState:
        def run(active_state: AgentState) -> AgentState:
            if active_state.get("rag_answer") is not None:
                return active_state
            rag_answer = fallback_answer(
                active_state["query"],
                active_state["retrieval_mode"],
                active_state["top_k"],
                active_state["use_graph"],
            )
            return {
                **active_state,
                "rag_answer": rag_answer,
                "answer": rag_answer.answer,
            }

        return self._with_trace(state, "self_check", run)

    def _finalize(self, state: AgentState) -> AgentState:
        return self._with_trace(state, "finalize", lambda active_state: active_state)

    def _tool_node(
        self,
        state: AgentState,
        node_name: str,
        tool_name: str,
        func: Callable[[AgentState], AgentState],
    ) -> AgentState:
        started = perf_counter()
        try:
            next_state = func(state)
            return append_node_trace(
                next_state,
                node_name=node_name,
                status="success",
                details=self._trace_details(next_state),
                latency_ms=(perf_counter() - started) * 1000,
            )
        except Exception as exc:
            next_state = self._record_tool_error(state, tool_name, exc)
            return append_node_trace(
                next_state,
                node_name=node_name,
                status="error",
                details=self._trace_details(next_state),
                latency_ms=(perf_counter() - started) * 1000,
                error=str(exc),
            )

    def _with_trace(
        self,
        state: AgentState,
        node_name: str,
        func: Callable[[AgentState], AgentState],
    ) -> AgentState:
        started = perf_counter()
        try:
            next_state = func(state)
            return append_node_trace(
                next_state,
                node_name=node_name,
                status="success",
                details=self._trace_details(next_state),
                latency_ms=(perf_counter() - started) * 1000,
            )
        except Exception as exc:
            next_state = dict(state)
            next_state["errors"] = [*state.get("errors", []), f"{node_name}: {exc}"]
            return append_node_trace(
                next_state,
                node_name=node_name,
                status="error",
                details=self._trace_details(next_state),
                latency_ms=(perf_counter() - started) * 1000,
                error=str(exc),
            )

    def _pop_next_step(
        self,
        state: AgentState,
        expected_tool: str,
    ) -> tuple[AgentPlannedStep | None, AgentState]:
        pending_steps = list(state.get("pending_steps", []))
        if not pending_steps or pending_steps[0].tool_name != expected_tool:
            return None, state
        return pending_steps[0], {**state, "pending_steps": pending_steps[1:]}

    def _record_tool_error(self, state: AgentState, tool_name: str, exc: Exception) -> AgentState:
        pending_steps = list(state.get("pending_steps", []))
        step: AgentPlannedStep | None = None
        if pending_steps and pending_steps[0].tool_name == tool_name:
            step = pending_steps.pop(0)
        error_message = str(exc)
        return {
            **state,
            "pending_steps": pending_steps,
            "errors": [*state.get("errors", []), f"{tool_name}: {error_message}"],
            "completed_steps": [
                *state.get("completed_steps", []),
                AgentToolStep(
                    tool_name=step.tool_name if step else tool_name,
                    status="error",
                    observation=f"{tool_name} 执行失败：{error_message}",
                    output={},
                ),
            ],
        }

    def _route_next_tool(self, state: AgentState) -> str:
        pending_steps = state.get("pending_steps", [])
        if not pending_steps:
            return "self_check"
        next_tool = pending_steps[0].tool_name
        if next_tool == "search_patents":
            return "execute_search"
        if next_tool == "summarize_patent":
            return "execute_summary"
        if next_tool == "graph_search":
            return "execute_graph_search"
        if next_tool == "graph_rag_answer":
            return "execute_graph_rag"
        return "self_check"

    def _trace_details(self, state: AgentState) -> dict[str, object]:
        planning_metadata = state.get("planning_metadata", {})
        details: dict[str, object] = {
            "pending_step_count": len(state.get("pending_steps", [])),
            "completed_step_count": len(state.get("completed_steps", [])),
            "errors": len(state.get("errors", [])),
            "has_answer": bool(state.get("answer")),
            "fallback_used": bool(planning_metadata.get("fallback_used", False)),
        }
        if planning_metadata.get("source"):
            details["planning_source"] = planning_metadata["source"]
        if planning_metadata.get("confidence") is not None:
            details["confidence"] = planning_metadata["confidence"]
        if planning_metadata.get("confidence_label"):
            details["confidence_label"] = planning_metadata["confidence_label"]
        if planning_metadata.get("rejection_reason"):
            details["rejection_reason"] = planning_metadata["rejection_reason"]
        return details

    def _build_llm_planner(self, chat_client: ChatClient | None) -> LlmIntentPlanner | None:
        if not self.enable_llm_planner:
            return None
        planner_client = chat_client
        if planner_client is None:
            try:
                planner_client = create_chat_client(temperature=0.0)
            except Exception as exc:  # pragma: no cover - depends on local provider config.
                self.llm_planner_error = str(exc)
                return None
        return LlmIntentPlanner(
            planner_client,
            min_confidence=self.llm_planner_min_confidence,
        )
