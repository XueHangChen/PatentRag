import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import {
  askPatentRag,
  getGraphStats,
  getHealth,
  listPatents,
  runPatentAgent,
  searchGraph,
  searchPatents
} from "./api/client";
import type {
  AgentRunResult,
  GraphKeywordMatch,
  GraphStats,
  PatentListItem,
  RagAnswer,
  RagGraphSource,
  RagSource,
  SearchHit
} from "./api/client";
import { buildTechnicalEvidenceGroups, formatEvidenceScore } from "./evidence";

type ApiStatus = "checking" | "ok" | "offline";
type LoadingState = "idle" | "loading" | "error";

const sampleQueries = ["液氮罐运输固定", "心脏支架材料 生物降解", "小儿智能雾化器"];
const intentLabels: Record<AgentRunResult["intent"], string> = {
  patent_qa: "专利问答",
  patent_summary: "专利总结",
  idea_analysis: "创意分析"
};

function formatTraceCount(value: unknown): number | string {
  return typeof value === "number" || typeof value === "string" ? value : 0;
}

function renderChipGroup(label: string, values: string[]) {
  if (values.length === 0) {
    return null;
  }

  return (
    <div className="evidence-group" key={label}>
      <span className="evidence-group-label">{label}</span>
      <div className="keyword-row evidence-chip-row">
        {values.map((value, index) => (
          <span key={`${label}-${value}-${index}`}>{value}</span>
        ))}
      </div>
    </div>
  );
}

function renderTextSourceCard(source: RagSource) {
  return (
    <article className="source-item" key={source.source_id}>
      <div className="result-title-row">
        <h3>
          [{source.source_id}] {source.title}
        </h3>
        <span>{formatEvidenceScore(source.score)}</span>
      </div>
      <div className="result-meta">
        <span>{source.patent_id}</span>
        <span>{source.section}</span>
        {source.claim_number !== null ? <span>权利要求 {source.claim_number}</span> : null}
      </div>
      <p>{source.snippet}</p>
    </article>
  );
}

function renderSearchHit(hit: SearchHit) {
  return (
    <article className="result-item" key={hit.chunk_id}>
      <div className="result-title-row">
        <h3>{hit.title}</h3>
        <span>{formatEvidenceScore(hit.score)}</span>
      </div>
      <div className="result-meta">
        <span>{hit.patent_id}</span>
        <span>{hit.section}</span>
        {hit.claim_number !== null ? <span>权利要求 {hit.claim_number}</span> : null}
      </div>
      <p>{hit.snippet}</p>
    </article>
  );
}

function renderGraphEvidenceCard(source: RagGraphSource) {
  const technicalGroups = buildTechnicalEvidenceGroups(source);

  return (
    <article className="source-item graph-source-item" key={source.source_id}>
      <div className="result-title-row">
        <h3>
          [{source.source_id}] {source.title}
        </h3>
        <span>{formatEvidenceScore(source.score)}</span>
      </div>
      <div className="result-meta">
        <span>{source.patent_id}</span>
        <span>{source.keywords.length} 个关键词</span>
        <span>{source.supporting_chunk_ids.length} 个支撑片段</span>
        <span>{source.claim_numbers.length} 项权利要求</span>
      </div>
      {source.relation_summary ? <p>{source.relation_summary}</p> : null}
      {technicalGroups.length > 0 ? (
        <div className="evidence-groups">
          {technicalGroups.map((group) => renderChipGroup(group.label, group.values))}
        </div>
      ) : null}
      {renderChipGroup("命中词", source.matched_terms)}
      {source.supporting_chunk_ids.length > 0 ? (
        <div className="supporting-chunks">
          <span>支撑片段</span>
          <code>{source.supporting_chunk_ids.join(" / ")}</code>
        </div>
      ) : null}
    </article>
  );
}

export default function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [patents, setPatents] = useState<PatentListItem[]>([]);
  const [patentState, setPatentState] = useState<LoadingState>("idle");
  const [searchState, setSearchState] = useState<LoadingState>("idle");
  const [query, setQuery] = useState(sampleQueries[0]);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [message, setMessage] = useState("");
  const [ragState, setRagState] = useState<LoadingState>("idle");
  const [ragQuestion, setRagQuestion] = useState("低温样本运输时如何避免容器磕碰？");
  const [ragAnswer, setRagAnswer] = useState<RagAnswer | null>(null);
  const [ragMessage, setRagMessage] = useState("");
  const [graphRagEnabled, setGraphRagEnabled] = useState(true);
  const [graphState, setGraphState] = useState<LoadingState>("idle");
  const [graphSearchState, setGraphSearchState] = useState<LoadingState>("idle");
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [graphKeyword, setGraphKeyword] = useState("液氮罐");
  const [graphMatches, setGraphMatches] = useState<GraphKeywordMatch[]>([]);
  const [graphMessage, setGraphMessage] = useState("");
  const [agentState, setAgentState] = useState<LoadingState>("idle");
  const [agentGoal, setAgentGoal] = useState(
    "我想设计一个低温样本运输装置，现有专利有哪些方案？还能怎么改进？"
  );
  const [agentResult, setAgentResult] = useState<AgentRunResult | null>(null);
  const [agentMessage, setAgentMessage] = useState("");
  const [agentGraphEnabled, setAgentGraphEnabled] = useState(true);

  useEffect(() => {
    getHealth()
      .then(() => setApiStatus("ok"))
      .catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => {
    setPatentState("loading");
    listPatents()
      .then((items) => {
        setPatents(items);
        setPatentState("idle");
      })
      .catch(() => {
        setPatentState("error");
      });
  }, []);

  useEffect(() => {
    setGraphState("loading");
    getGraphStats()
      .then((stats) => {
        setGraphStats(stats);
        setGraphState("idle");
      })
      .catch(() => {
        setGraphState("error");
      });
  }, []);

  useEffect(() => {
    setGraphSearchState("loading");
    searchGraph("液氮罐", 8)
      .then((response) => {
        setGraphMatches(response.matches);
        setGraphSearchState("idle");
      })
      .catch(() => {
        setGraphMatches([]);
        setGraphSearchState("error");
      });
  }, []);

  const patentTypeSummary = useMemo(() => {
    return patents.reduce<Record<string, number>>((summary, patent) => {
      summary[patent.patent_type] = (summary[patent.patent_type] ?? 0) + 1;
      return summary;
    }, {});
  }, [patents]);

  async function handleSearch(event?: FormEvent<HTMLFormElement>, nextQuery = query) {
    event?.preventDefault();
    const trimmedQuery = nextQuery.trim();
    if (!trimmedQuery) {
      setMessage("请输入检索词");
      return;
    }

    setQuery(trimmedQuery);
    setMessage("");
    setSearchState("loading");
    try {
      const response = await searchPatents(trimmedQuery, 8);
      setHits(response.hits);
      setSearchState("idle");
    } catch {
      setSearchState("error");
      setHits([]);
    }
  }

  async function handleAsk(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmedQuestion = ragQuestion.trim();
    if (!trimmedQuestion) {
      setRagMessage("请输入问题");
      return;
    }

    setRagQuestion(trimmedQuestion);
    setRagMessage("");
    setRagState("loading");
    try {
      const response = await askPatentRag(trimmedQuestion, 5, graphRagEnabled);
      setRagAnswer(response);
      setRagState("idle");
    } catch {
      setRagState("error");
      setRagAnswer(null);
    }
  }

  async function handleGraphSearch(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmedKeyword = graphKeyword.trim();
    if (!trimmedKeyword) {
      setGraphMessage("请输入图谱关键词");
      return;
    }

    setGraphKeyword(trimmedKeyword);
    setGraphMessage("");
    setGraphSearchState("loading");
    try {
      const response = await searchGraph(trimmedKeyword, 8);
      setGraphMatches(response.matches);
      setGraphSearchState("idle");
    } catch {
      setGraphMatches([]);
      setGraphSearchState("error");
    }
  }

  async function handleRunAgent(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmedGoal = agentGoal.trim();
    if (!trimmedGoal) {
      setAgentMessage("请输入 Agent 任务");
      return;
    }

    setAgentGoal(trimmedGoal);
    setAgentMessage("");
    setAgentState("loading");
    try {
      const response = await runPatentAgent(trimmedGoal, 5, agentGraphEnabled);
      setAgentResult(response);
      setAgentState("idle");
    } catch {
      setAgentState("error");
      setAgentResult(null);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace-header">
        <div>
          <p className="eyebrow">Patent KG Agent</p>
          <h1>专利知识图谱 Agent 工作台</h1>
          <p className="subtitle">
            面向专利检索、总结、对话、创意分析和图谱探索的前后端分离应用。
          </p>
        </div>
        <div className={`status-pill status-${apiStatus}`}>
          API {apiStatus === "checking" ? "checking" : apiStatus}
        </div>
      </section>

      <section className="summary-grid" aria-label="数据概览">
        <article className="summary-tile">
          <span>专利总数</span>
          <strong>{patentState === "loading" ? "..." : patents.length}</strong>
        </article>
        <article className="summary-tile">
          <span>发明申请</span>
          <strong>{patentTypeSummary.invention_application ?? 0}</strong>
        </article>
        <article className="summary-tile">
          <span>实用新型</span>
          <strong>{patentTypeSummary.utility_model ?? 0}</strong>
        </article>
      </section>

      <section className="panel agent-panel" aria-label="Agent 工作台">
        <div className="panel-header">
          <h2>Agent 工作台</h2>
          <span>{agentResult ? intentLabels[agentResult.intent] : "Agent"}</span>
        </div>

        <form className="agent-form" onSubmit={handleRunAgent}>
          <textarea
            value={agentGoal}
            onChange={(event) => setAgentGoal(event.target.value)}
            placeholder="输入一个复杂专利任务"
            rows={3}
          />
          <button type="submit" disabled={agentState === "loading"}>
            {agentState === "loading" ? "执行中" : "运行"}
          </button>
          <label className="qa-toggle">
            <input
              type="checkbox"
              checked={agentGraphEnabled}
              onChange={(event) => setAgentGraphEnabled(event.target.checked)}
            />
            <span>图谱增强</span>
          </label>
        </form>

        {agentMessage ? <p className="notice">{agentMessage}</p> : null}
        {agentState === "error" ? (
          <p className="notice error">Agent 暂不可用，请确认后端、索引、图谱和 Qwen 配置</p>
        ) : null}

        {agentResult ? (
          <div className="agent-result">
            <div className="agent-summary-row">
              <article>
                <span>意图</span>
                <strong>{intentLabels[agentResult.intent]}</strong>
              </article>
              <article>
                <span>计划步骤</span>
                <strong>{agentResult.plan.steps.length}</strong>
              </article>
              <article>
                <span>工具执行</span>
                <strong>{agentResult.steps.length}</strong>
              </article>
              <article>
                <span>证据</span>
                <strong>
                  {agentResult.sources.length} 原文 / {agentResult.graph_sources.length} 图谱
                </strong>
              </article>
              <article>
                <span>Planner</span>
                <strong>{agentResult.planner_mode ?? "rule"}</strong>
              </article>
              <article>
                <span>Checkpoint</span>
                <strong>{agentResult.checkpoint_id ?? "local-run"}</strong>
              </article>
            </div>

            <article className="agent-answer">
              <h3>回答</h3>
              <p>{agentResult.answer}</p>
            </article>

            <div className="agent-trace-grid">
              <section>
                <h3>Plan</h3>
                <div className="agent-step-list">
                  {agentResult.plan.steps.map((step, index) => (
                    <article className="agent-step" key={`${step.tool_name}-${index}`}>
                      <div className="result-title-row">
                        <h4>
                          {index + 1}. {step.tool_name}
                        </h4>
                      </div>
                      <p>{step.reason}</p>
                    </article>
                  ))}
                </div>
              </section>

              <section>
                <h3>Trace</h3>
                <div className="agent-step-list">
                  {agentResult.steps.map((step, index) => (
                    <article className="agent-step" key={`${step.tool_name}-${index}`}>
                      <div className="result-title-row">
                        <h4>{step.tool_name}</h4>
                        <span>{step.status}</span>
                      </div>
                      <p>{step.observation}</p>
                    </article>
                  ))}
                </div>
              </section>

              {agentResult.node_trace && agentResult.node_trace.length > 0 ? (
                <section>
                  <h3>LangGraph</h3>
                  <div className="agent-step-list">
                    {agentResult.node_trace.map((trace, index) => (
                      <article className="agent-step" key={`${trace.node_name}-${index}`}>
                        <div className="result-title-row">
                          <h4>{trace.node_name}</h4>
                          <span>{trace.status}</span>
                        </div>
                        <p>
                          {formatTraceCount(trace.details.completed_step_count)} completed /{" "}
                          {formatTraceCount(trace.details.pending_step_count)} pending
                        </p>
                        {trace.error ? <p>{trace.error}</p> : null}
                      </article>
                    ))}
                  </div>
                </section>
              ) : null}
            </div>

            {agentResult.sources.length > 0 ? (
              <section className="evidence-section">
                <h3>原文证据</h3>
                <div className="source-list">
                  {agentResult.sources.slice(0, 3).map((source) => renderTextSourceCard(source))}
                </div>
              </section>
            ) : null}

            {agentResult.graph_sources.length > 0 ? (
              <section className="evidence-section">
                <h3>图谱证据</h3>
                <div className="source-list agent-graph-source-list">
                  {agentResult.graph_sources.slice(0, 3).map((source) => renderGraphEvidenceCard(source))}
                </div>
              </section>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="panel graph-panel" aria-label="知识图谱探索">
        <div className="panel-header">
          <h2>知识图谱探索</h2>
          <span>{graphState === "loading" ? "加载中" : "Graph"}</span>
        </div>

        <div className="graph-layout">
          <div className="graph-stats">
            <article>
              <span>节点</span>
              <strong>{graphStats ? graphStats.node_count : "..."}</strong>
            </article>
            <article>
              <span>关系</span>
              <strong>{graphStats ? graphStats.edge_count : "..."}</strong>
            </article>
            <article>
              <span>专利节点</span>
              <strong>{graphStats?.node_labels.Patent ?? 0}</strong>
            </article>
            <article>
              <span>关键词节点</span>
              <strong>{graphStats?.node_labels.Keyword ?? 0}</strong>
            </article>
          </div>

          <form className="graph-form" onSubmit={handleGraphSearch}>
            <input
              value={graphKeyword}
              onChange={(event) => setGraphKeyword(event.target.value)}
              placeholder="输入图谱关键词，例如 液氮罐、固定架、支架"
            />
            <button type="submit" disabled={graphSearchState === "loading"}>
              {graphSearchState === "loading" ? "查询中" : "查询"}
            </button>
          </form>
        </div>

        {graphMessage ? <p className="notice">{graphMessage}</p> : null}
        {graphState === "error" ? (
          <p className="notice error">知识图谱暂不可用，请先运行 build_graph 脚本</p>
        ) : null}
        {graphSearchState === "error" ? (
          <p className="notice error">图谱查询暂不可用，请确认后端图谱接口</p>
        ) : null}

        <div className="graph-match-list">
          {graphMatches.map((match) => (
            <article className="graph-match-item" key={match.patent_id}>
              <h3>{match.title}</h3>
              <div className="result-meta">
                <span>{match.patent_id}</span>
                <span>{match.keywords.length} 个关联关键词</span>
              </div>
              <div className="keyword-row">
                {match.keywords.map((keyword) => (
                  <span key={keyword}>{keyword}</span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="panel qa-panel" aria-label="专利智能问答">
        <div className="panel-header">
          <h2>专利智能问答</h2>
          <span>
            {ragAnswer
              ? `${ragAnswer.sources.length} 条原文 / ${ragAnswer.graph_sources.length} 条图谱`
              : "GraphRAG"}
          </span>
        </div>

        <form className="qa-form" onSubmit={handleAsk}>
          <textarea
            value={ragQuestion}
            onChange={(event) => setRagQuestion(event.target.value)}
            placeholder="输入一个需要基于专利知识库回答的问题"
            rows={3}
          />
          <button type="submit" disabled={ragState === "loading"}>
            {ragState === "loading" ? "生成中" : "开始问答"}
          </button>
          <label className="qa-toggle">
            <input
              type="checkbox"
              checked={graphRagEnabled}
              onChange={(event) => setGraphRagEnabled(event.target.checked)}
            />
            <span>图谱增强</span>
          </label>
        </form>

        {ragMessage ? <p className="notice">{ragMessage}</p> : null}
        {ragState === "error" ? (
          <p className="notice error">RAG 问答暂不可用，请确认后端和 Qwen 配置</p>
        ) : null}

        {ragAnswer ? (
          <div className="qa-answer">
            <article>
              <h3>回答</h3>
              <p>{ragAnswer.answer}</p>
            </article>

            {ragAnswer.sources.length > 0 ? (
              <section className="evidence-section">
                <h3>原文证据</h3>
                <div className="source-list">
                  {ragAnswer.sources.map((source) => renderTextSourceCard(source))}
                </div>
              </section>
            ) : null}

            {ragAnswer.graph_sources.length > 0 ? (
              <section className="evidence-section">
                <h3>图谱证据</h3>
                <div className="source-list graph-rag-source-list">
                  {ragAnswer.graph_sources.map((source) => renderGraphEvidenceCard(source))}
                </div>
              </section>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="workspace-grid">
        <section className="panel search-panel" aria-label="专利检索">
          <div className="panel-header">
            <h2>专利检索</h2>
            <span>{hits.length} 条命中</span>
          </div>

          <form className="search-form" onSubmit={handleSearch}>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="输入技术问题、部件或专利关键词"
            />
            <button type="submit" disabled={searchState === "loading"}>
              {searchState === "loading" ? "检索中" : "检索"}
            </button>
          </form>

          <div className="query-row" aria-label="示例检索">
            {sampleQueries.map((sample) => (
              <button type="button" key={sample} onClick={() => void handleSearch(undefined, sample)}>
                {sample}
              </button>
            ))}
          </div>

          {message ? <p className="notice">{message}</p> : null}
          {searchState === "error" ? <p className="notice error">后端检索接口暂不可用</p> : null}

          <div className="result-list">{hits.map((hit) => renderSearchHit(hit))}</div>
        </section>

        <aside className="panel patent-panel" aria-label="专利列表">
          <div className="panel-header">
            <h2>专利列表</h2>
            <span>{patentState === "loading" ? "加载中" : `${patents.length} 篇`}</span>
          </div>
          {patentState === "error" ? <p className="notice error">后端专利接口暂不可用</p> : null}
          <div className="patent-list">
            {patents.map((patent) => (
              <article className="patent-row" key={patent.patent_id}>
                <h3>{patent.title}</h3>
                <div>
                  <span>{patent.patent_id}</span>
                  <span>{patent.ipc_classes.join(" / ")}</span>
                </div>
              </article>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}
