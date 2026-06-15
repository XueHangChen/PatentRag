# 专利知识图谱 Agent 技术架构

版本：v0.1  
日期：2026-06-10  
对应需求文档：`docs/PRD.md`

## 1. 架构目标

本项目的技术架构围绕一个核心问题展开：如何把非结构化专利 Markdown 文档，稳定地转化为可检索、可追溯、可推理、可对话的企业级知识系统。

因此，架构不直接从“调用大模型问答”开始，而是分成五层：

1. 数据层：负责文档读取、清洗、结构化解析、数据校验和落盘。
2. 知识层：负责 chunk、向量索引、关键词索引、知识图谱和图谱关系存储。
3. Agent 层：负责意图识别、工具选择、多步推理、引用约束和对话状态。
4. 后端服务层：负责 FastAPI、鉴权预留、异步任务、SSE/WebSocket 流式响应和外部接口契约。
5. 前端应用层：负责用户交互、检索结果展示、对话体验、图谱探索和专利分析工作台。

这样设计的好处是：MVP 可以先打通最短链路，但每个模块天然支持扩展到更复杂的数据源、检索策略和 Agent 工作流。

## 2. 前后端分离形态

本项目应作为前后端分离的完整项目来建设，而不是只有后端脚本或单页 Demo。

```text
Browser
  |
  | HTTP JSON / SSE stream / WebSocket
  v
Frontend Web App
  - patent search workspace
  - patent detail page
  - chat workspace
  - graph explorer
  - idea analysis page
  |
  | REST API contract
  v
Backend API Service
  - FastAPI routes
  - request validation
  - auth placeholder
  - background ingestion tasks
  - Agent orchestration
  |
  +--------------------+
  |                    |
  v                    v
Knowledge Services   Storage
  - retrieval          - processed JSONL
  - graph query        - vector index
  - LLM gateway        - graph data
  - eval/tracing       - logs
```

前后端职责边界：

- 前端不直接访问向量库、图数据库或 LLM API。
- 后端统一封装 RAG、GraphRAG、Agent 工具调用和模型调用。
- 前后端通过稳定 API 契约交互，优先使用 JSON；对长回答和 Agent 执行过程使用 SSE 或 WebSocket。
- 前端负责把证据、引用、图谱关系和 Agent 步骤清晰展示出来，不承担核心推理逻辑。
- 后端负责事实约束、引用生成、数据权限、安全边界和日志追踪。

## 3. 后端知识链路

```text
Markdown patents
    |
    v
Ingestion
  - UTF-8 reading
  - markdown normalization
  - field extraction
  - schema validation
    |
    v
Structured patent JSONL
    |
    +--------------------+
    |                    |
    v                    v
Retrieval Index       Knowledge Graph
  - chunks              - entities
  - embeddings          - relations
  - BM25                - evidence spans
  - reranking
    |                    |
    +---------+----------+
              |
              v
Agent Tools
  - search_patents
  - get_patent_detail
  - query_knowledge_graph
  - summarize_patent
  - compare_patents
  - analyze_idea
              |
              v
API / CLI / UI
```

## 4. 推荐项目结构

```text
Patent_RAG/
  docs/
    PRD.md
    ARCHITECTURE.md
  patant/
    *.md
  configs/
    default.yaml
  data/
    raw/
    processed/
    indexes/
    graph/
  evals/
    datasets/
  logs/
  scripts/
  frontend/
    README.md
    package.json
    index.html
    vite.config.ts
    tsconfig.json
    .env.example
    src/
      App.tsx
      main.tsx
      styles.css
      api/
        client.ts
  src/
    patent_rag/
      __init__.py
      config.py
      observability.py
      api/
        main.py
      agent/
      domain/
        schemas.py
      graph/
      ingestion/
      retrieval/
  tests/
    test_package.py
  .env.example
  .gitignore
  pyproject.toml
  README.md
```

目录设计说明：

- `patant/`：保留当前原始 Markdown 数据，暂不移动，避免破坏你的已有数据路径。
- `data/processed/`：保存解析后的结构化 JSONL。
- `data/indexes/`：保存向量索引、BM25 索引和重排序缓存。
- `data/graph/`：保存图谱节点、边、图数据库导入文件或本地图结构。
- `src/patent_rag/domain/`：定义核心业务对象，所有模块共享同一套 Schema。
- `src/patent_rag/ingestion/`：负责从 Markdown 到结构化 PatentDocument。
- `src/patent_rag/retrieval/`：负责 chunk、embedding、关键词检索、混合检索和 reranking。
- `src/patent_rag/graph/`：负责实体关系抽取、图谱存储和图谱查询。
- `src/patent_rag/agent/`：负责工具封装、工作流编排、对话状态和回答自检。
- `src/patent_rag/api/`：负责 FastAPI 服务。
- `frontend/`：独立前端应用，推荐 React + Vite + TypeScript，负责用户界面和 API 调用。
- `evals/`：保存固定评测集和评测脚本，面试时这是区分 Demo 和工程项目的重要证据。

说明：

- 当前后端 Python 包保留在根目录 `src/patent_rag/`，这是 Python 项目常见布局。
- 前后端分离的关键是运行时边界和 API 契约，而不是必须把后端文件放进 `backend/` 目录。
- 如果后续项目扩大，可以演进为 `apps/backend/` + `apps/frontend/` 的 monorepo 布局；当前阶段先保持后端包路径稳定，减少迁移成本。

## 5. 模块边界

### 5.1 Domain

Domain 层只定义业务对象，不依赖具体模型、数据库或 API。

核心对象：

- `PatentDocument`：一篇完整专利。
- `PatentMetadata`：公开号、申请号、申请人、发明人、IPC 等。
- `PatentSection`：技术领域、背景技术、发明内容等章节。
- `PatentClaim`：权利要求。
- `PatentChunk`：检索用片段。
- `GraphNode` / `GraphEdge`：图谱节点和关系。

工程原则：

- 所有外部数据进入系统前，必须先变成 Domain Schema。
- 后续替换 LLM、向量库、图数据库时，不应该影响 Domain Schema。

### 5.2 Ingestion

Ingestion 层负责“确定性优先”的数据处理。

第一阶段能力：

- 读取 `patant/*.md`，显式使用 UTF-8。
- 清洗 PDF 转 Markdown 带来的异常断行和多余空格。
- 解析头部元数据、摘要、权利要求和说明书章节。
- 生成 `data/processed/patents.jsonl`。

第二阶段能力：

- 使用 LLM 结构化补全难以规则解析的字段。
- 为每个字段记录证据片段和置信度。
- 加入解析质量报告。

这里的关键面试点是：不要把所有抽取都交给 LLM。专利文档格式相对稳定，规则解析更便宜、更可控；LLM 应该用于补全和复杂实体抽取。

### 5.3 Retrieval

Retrieval 层负责“找证据”。

基础检索：

- BM25：适合申请号、IPC、部件名称等精确词。
- Vector Search：适合自然语言语义查询。
- Metadata Filter：按专利类型、IPC、申请人、发明人、日期过滤。

进阶检索：

- Hybrid Search：合并 BM25 和向量召回。
- Reranking：对候选片段二次排序。
- Query Rewriting：把用户问题改写为更适合专利语料的查询。
- Citation Packing：控制最终给 LLM 的上下文质量和引用粒度。

### 5.4 Graph

Graph 层负责“看关系”。

图谱分两步建设：

1. 元数据图谱：Patent、Applicant、Inventor、Agency、IPCClass、Claim。
2. 技术语义图谱：Problem、Solution、Component、Effect、TechnicalField。

为什么分两步：

- 元数据图谱可以通过规则稳定抽取，适合先交付。
- 技术语义图谱依赖 LLM 抽取，需要证据句、置信度和人工抽样校验。

GraphRAG 查询流程：

```text
User query
  -> retrieve seed patents/chunks
  -> expand related graph nodes
  -> collect neighboring evidence
  -> generate grounded answer
```

### 5.5 Agent

Agent 层负责“决定怎么查、怎么答”。

推荐工作流：

1. 判断用户意图：检索、总结、对比、图谱查询、创意分析或报告生成。
2. 选择工具：调用检索、详情、图谱、总结或对比工具。
3. 聚合证据：合并 chunk、结构化字段和图谱关系。
4. 生成答案：输出带引用、带边界声明的回答。
5. 自检：检查是否回答了问题、是否有证据、是否需要说明知识库不足。

企业级 Agent 的重点不是“看起来会聊天”，而是工具调用路径可追踪、结果可复现、错误可定位。

### 5.6 Backend API

API 层对外暴露服务，MVP 保持轻量。

建议接口：

- `GET /health`
- `POST /ingest`
- `GET /patents`
- `GET /patents/{patent_id}`
- `POST /search`
- `POST /chat`
- `POST /idea/analyze`

后端 API 设计原则：

- 路由层只做参数校验和响应组装，不直接写复杂业务逻辑。
- RAG、图谱、Agent 能力放在服务层或工具层，便于 CLI、API、评测脚本复用。
- 长耗时任务采用后台任务或任务队列预留，例如 ingestion、embedding、图谱抽取。
- 对话接口优先支持非流式 JSON，进阶阶段增加 SSE 流式输出 Agent 执行过程。

### 5.7 Frontend Web App

前端应用负责提供完整产品体验，而不是只做 API 调试页面。

核心页面：

- 专利检索页：关键词/语义检索、筛选、结果排序、引用片段。
- 专利详情页：元数据、摘要、权利要求、章节、相关图谱关系。
- Agent 对话页：多轮问答、引用证据、工具调用步骤、流式输出。
- 创意分析页：输入用户想法，展示相似专利、相似点、差异点和可探索方向。
- 知识图谱页：实体搜索、关系展开、专利-申请人-技术实体网络。
- 评测/调试页：面向开发者展示检索 query、召回 chunk、rerank 结果和回答引用。

前端技术建议：

- React + Vite + TypeScript。
- TanStack Query 管理请求缓存和加载状态。
- React Router 管理页面路由。
- 图谱可视化后续可选 React Flow、ECharts Graph 或 Sigma.js。
- UI 风格应偏工作台：信息密度高、筛选明确、引用可追踪，避免做成营销落地页。

前端 API 契约示例：

```text
GET  /health
GET  /patents
GET  /patents/{patent_id}
POST /search
POST /chat
POST /idea/analyze
GET  /graph/neighborhood?node_id=...
```

## 6. 数据落盘约定

建议使用 JSONL 作为早期中间格式：

- 方便增量处理。
- 方便人工查看和 diff。
- 方便后续导入数据库。

规划文件：

- `data/processed/patents.jsonl`：结构化专利。
- `data/processed/chunks.jsonl`：检索片段。
- `data/graph/nodes.jsonl`：图谱节点。
- `data/graph/edges.jsonl`：图谱关系。
- `data/indexes/`：向量库和关键词索引。

## 7. 技术选型原则

MVP 阶段：

- FastAPI 提供 API。
- Pydantic 定义 Schema。
- 本地 JSONL 保存结构化结果。
- 本地向量库或轻量向量索引保存 embedding。
- NetworkX 或 JSONL 保存初版图谱。

进阶阶段：

- 引入 Qdrant、Milvus 或 Chroma 做向量库。
- 引入 Neo4j 做图数据库。
- 引入 LangGraph 或 OpenAI Agents SDK 做 Agent 编排。
- 引入 Ragas、DeepEval 或自研评测脚本做质量回归。
- 引入 tracing 记录检索、工具调用和回答生成链路。

选型原则：

- 先保证数据闭环，再堆框架。
- 先有可测试的规则解析，再引入 LLM 抽取。
- 先做可解释检索，再做多步 Agent。
- 每一层都能独立运行和验证。

## 8. 迭代顺序

### Iteration 1：工程初始化

目标：

- 建立项目目录。
- 建立基础配置和日志。
- 定义 Domain Schema。
- 准备测试框架。

完成标志：

- `python -m compileall src` 可通过。
- `pytest` 至少能跑通基础测试。
- `frontend/` 独立应用骨架存在，前后端 API 契约在文档中明确。

### Iteration 2：专利解析

目标：

- 实现 Markdown reader。
- 实现规则 parser。
- 生成 `patents.jsonl`。
- 为 2 到 3 篇样例编写解析测试。

完成标志：

- 20 篇专利全部解析成功。
- 关键字段抽取结果可人工检查。

### Iteration 3：基础 RAG

目标：

- 实现 chunk。
- 构建关键词和向量检索。
- 实现带引用的问答。

完成标志：

- 对“液氮罐运输固定”“肿瘤内科抢救车”等查询能召回正确专利。

### Iteration 4：前端工作台 MVP

目标：

- 实现专利检索页。
- 实现专利详情页。
- 实现 Agent 对话页基础交互。
- 与后端 `/search`、`/patents`、`/chat` 接口打通。

完成标志：

- 用户可以在浏览器中完成检索、查看详情和基于知识库提问。
- 前端展示引用来源，而不是只展示模型回答。

### Iteration 5：知识图谱

目标：

- 构建元数据图谱。
- 抽取技术实体和关系。
- 实现基础图谱查询。

完成标志：

- 能回答“某专利有哪些关键部件”“某 IPC 下有哪些专利”等问题。

### Iteration 6：Agent 工作流

目标：

- 工具封装。
- 意图路由。
- 多轮状态。
- 创意分析和对比分析。

完成标志：

- Agent 可以根据问题自主调用多个工具，并输出证据链。

## 9. 面试讲解主线

这个项目面试时可以这样讲：

1. 我把项目设计成前后端分离的完整产品，前端是专利分析工作台，后端是 RAG、图谱和 Agent 能力服务。
2. 我没有直接做一个简单聊天框，而是先把专利数据变成可验证的结构化知识。
3. 对专利这种强格式文档，我优先用规则解析保证稳定性，再用 LLM 做技术实体补全。
4. 检索层不是单一向量检索，而是 Hybrid Search，加上元数据过滤和后续 rerank。
5. 图谱不是装饰，而是服务于关系查询和 GraphRAG 的证据扩展。
6. Agent 不是无边界自动推理，而是围绕工具、状态、证据和自检构建。
7. 我设计了评测和 tracing，让系统可以持续优化，而不是凭感觉调 prompt。
