# Agent Evaluation and Trace Design

Date: 2026-06-15
Project: Patent KG Agent
Status: Proposed

## 1. Purpose

The project already has patent ingestion, hybrid retrieval, GraphRAG, a rule-based Agent planner, Agent tools, FastAPI endpoints, and a React Agent workspace. The next stage is to make the Agent measurable and debuggable.

This design introduces an Agent evaluation and trace layer with two goals:

- Record what an Agent run did: query, intent, plan, tool calls, observations, evidence, answer, and latency.
- Evaluate whether the Agent followed expected behavior across a fixed task set.

The evaluation strategy is:

- Offline by default: no Qwen call, stable and cheap for frequent regression checks.
- Online when `--online` is passed: runs the full Agent chain and validates answer-level signals.

## 2. Scope

In scope:

- Agent task dataset in `evals/agent_tasks.jsonl`.
- Trace schemas and JSONL writer in `src/patent_rag/observability/trace.py`.
- Agent evaluation schemas and evaluator in `src/patent_rag/evaluation/agent_eval.py`.
- CLI entrypoint in `scripts/evaluate_agent.py`.
- Unit tests for trace writing and offline evaluation behavior.
- JSON reports saved to `logs/evals/`.
- JSONL traces saved to `logs/traces/`.

Out of scope for this phase:

- LLM-as-judge scoring.
- Ragas or DeepEval integration.
- LangGraph migration.
- SSE or WebSocket streaming.
- Frontend evaluation dashboard.
- Long-term database-backed trace storage.

## 3. Proposed Files

```text
src/patent_rag/observability/
  __init__.py
  trace.py

src/patent_rag/evaluation/
  __init__.py
  agent_eval.py

evals/
  agent_tasks.jsonl

scripts/
  evaluate_agent.py

tests/
  test_observability_trace.py
  test_agent_evaluation.py
```

## 4. Architecture

The design separates factual trace recording from metric calculation.

```text
evals/agent_tasks.jsonl
        |
        v
scripts/evaluate_agent.py
        |
        v
AgentEvaluator
        |
        +--> PatentAgentPlanner
        +--> PatentAgentTools
        +--> PatentAgentService when --online is enabled
        |
        v
AgentEvaluationReport
        |
        +--> logs/evals/agent_eval_*.json
        +--> logs/traces/agent_traces.jsonl
```

Responsibilities:

- `observability.trace`: Defines Agent trace records and writes JSONL trace rows.
- `evaluation.agent_eval`: Reads tasks, runs offline or online evaluation, calculates metrics, and writes reports.
- `scripts.evaluate_agent`: Parses CLI arguments and calls the evaluation module.

## 5. Agent Task Dataset

`evals/agent_tasks.jsonl` contains one JSON object per task.

Example:

```json
{
  "task_id": "agent_001",
  "query": "低温样本运输时如何避免容器碰撞？",
  "expected_intent": "patent_qa",
  "expected_tools": ["search_patents", "graph_search", "graph_rag_answer"],
  "expected_patent_ids": ["CN206539886U"],
  "expected_keywords": ["液氮罐", "固定架", "碰撞"],
  "requires_graph": true,
  "notes": "应召回车载液氮罐固定架，并说明固定架用于防止运输碰撞。"
}
```

Field meanings:

- `task_id`: Stable task identifier.
- `query`: User task sent to the Agent.
- `expected_intent`: Expected planner intent, such as `patent_qa`, `patent_summary`, or `idea_analysis`.
- `expected_tools`: Tools that should be covered by the generated plan.
- `expected_patent_ids`: Patent ids expected in retrieved sources or generated evidence.
- `expected_keywords`: Keywords expected in evidence, and in the final answer during online evaluation.
- `requires_graph`: Whether graph usage is expected.
- `notes`: Human-readable task rationale.

The initial dataset should contain 6 to 8 tasks across:

- Patent QA.
- Patent summary.
- Idea analysis.

## 6. Trace Structure

`AgentTrace` records one Agent run. It is internal observability data, not the public API response.

Core fields:

```text
trace_id
created_at
query
mode
intent
plan
steps
answer
sources
graph_sources
latency_ms
metadata
```

`AgentTraceStep` records one tool execution.

Core fields:

```text
step_id
tool_name
status
started_at
ended_at
latency_ms
tool_input
observation
output_summary
error
```

Trace rows are appended to:

```text
logs/traces/agent_traces.jsonl
```

The trace should keep concise summaries rather than full raw chunk payloads. It should include enough information for debugging and evaluation: tool input, observations, source ids, graph source ids, patent ids, status, and errors.

## 7. Offline Evaluation

Default command:

```powershell
python scripts\evaluate_agent.py
```

Offline evaluation does not call Qwen. It evaluates Agent decisions and evidence signals.

Flow:

1. Read `evals/agent_tasks.jsonl`.
2. Run `PatentAgentPlanner` for each task.
3. Compare detected intent with `expected_intent`.
4. Compare planned tools against `expected_tools` by coverage, not strict order.
5. Use controlled local tools to inspect retrieval and graph signals.
6. Check whether expected patents appear in sources or graph sources.
7. Write per-task trace rows.
8. Write aggregate JSON report.

Offline metrics:

- `intent_accuracy`.
- `tool_coverage_rate`.
- `patent_recall_at_k`.
- `graph_usage_rate`.
- `source_presence_rate`.
- `graph_source_presence_rate`.

## 8. Online Evaluation

Online command:

```powershell
python scripts\evaluate_agent.py --online
```

Online evaluation runs the full Agent chain and can call Qwen. It should only be used when the environment has the required DashScope configuration.

Additional checks:

- Final answer is non-empty.
- Final answer contains citation markers like `[S1]` or `[G1]`.
- Final answer covers expected keywords.

Additional metrics:

- `answer_presence_rate`.
- `citation_presence_rate`.
- `keyword_coverage_rate`.

If the API key is missing, online evaluation should fail early with a clear message.

## 9. Report Structure

Reports are saved under:

```text
logs/evals/agent_eval_YYYYMMDD_HHMMSS.json
```

Aggregate report fields:

```text
mode
task_count
metrics
task_results
created_at
```

Per-task result fields:

```text
task_id
query
passed
intent_matched
tool_coverage
patent_recall
graph_used
source_present
graph_source_present
answer_present
citation_present
keyword_coverage
errors
trace_id
```

Terminal output should print a concise summary of aggregate metrics and the report path.

## 10. Error Handling

Per-task failures should not stop the whole evaluation run. A failed task records errors and evaluation continues.

Global failures should stop early with a clear message. Examples:

- Missing `evals/agent_tasks.jsonl`.
- Missing `data/processed/chunks.jsonl`.
- Missing Chroma index when vector or hybrid retrieval is required.
- Missing graph file when graph evaluation is required.
- Missing DashScope API key in online mode.

Trace write failures should be recorded in the report and surfaced in terminal output.

## 11. Testing Strategy

No automated test should call Qwen.

`tests/test_observability_trace.py` should cover:

- Building a trace from an Agent run result.
- Writing JSONL traces.
- Ensuring trace rows include query, intent, plan, steps, sources, and graph sources.

`tests/test_agent_evaluation.py` should cover:

- Loading task JSONL.
- Correct `intent_accuracy` calculation.
- Correct expected tool coverage calculation.
- Correct expected patent recall calculation.
- Offline evaluator creates a report with fake or controlled tools.
- A single task failure does not interrupt the full evaluation.

Online mode should be tested only for argument propagation and missing-key behavior.

## 12. Success Criteria

This phase is complete when:

- `python scripts\evaluate_agent.py` runs without calling Qwen and prints aggregate metrics.
- `python scripts\evaluate_agent.py --online` runs the full Agent chain when the API key is configured.
- Reports are saved in `logs/evals/`.
- Traces are appended to `logs/traces/agent_traces.jsonl`.
- Unit tests pass without external API calls.
- README explains how to run offline and online Agent evaluation.

## 13. Future Extensions

After this phase, the project can extend toward:

- Frontend evaluation dashboard.
- LangGraph workflow comparison.
- LLM-as-judge answer quality scoring.
- Trace visualization.
- Dataset expansion as more patents are added.
