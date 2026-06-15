# 专利知识图谱 Agent 项目需求文档

版本：v0.1  
日期：2026-06-10  
项目名称：Patent KG Agent  
当前数据：`patant/` 目录下 20 篇中文专利 Markdown 文档

## 1. 项目背景

本项目面向“专利数据智能检索、理解、总结与创意辅助分析”场景，基于本地专利 Markdown 文档构建结构化专利知识库、向量索引和知识图谱，并通过 Agent 工作流为用户提供自然语言检索、专利总结、多轮问答、相似专利发现、技术方案对比和创意启发能力。

项目目标不是做一个简单的 RAG Demo，而是按企业级 AI Agent 项目的方式建设：数据管道可复现，知识抽取可追踪，检索结果可解释，Agent 执行可观测，回答质量可评测，后续可以扩展到更多专利数据源。

## 2. 企业级开发路径确认

本项目建议按以下路径推进：

1. 需求文档：明确业务目标、用户场景、功能边界、非功能指标和验收标准。
2. 技术方案：确定数据处理、知识图谱、向量检索、Agent 编排、评测与部署方案。
3. 项目结构：规划工程目录、模块边界、配置管理、数据目录和测试目录。
4. MVP 开发：先打通从文档解析到问答检索的最短闭环。
5. 能力增强：加入图谱检索、混合检索、重排序、多工具 Agent、可观测性和评测体系。
6. 部署与演示：提供 API、前端界面、示例数据、演示脚本和面试讲解材料。

企业项目中 PRD、技术方案和项目结构并不是一次写死，而是随着迭代持续更新。本项目第一阶段先生成 PRD，再进入架构与目录规划。

## 3. 用户与场景

### 3.1 目标用户

- 专利研发人员：快速理解已有专利、查找相关技术路线。
- 产品/技术创新人员：基于已有专利寻找技术空白、生成改进思路。
- 法务/知识产权助理：辅助整理专利摘要、权利要求、申请人、发明人、IPC 分类等信息。
- 面试官视角：关注候选人是否理解 Agentic RAG、GraphRAG、数据工程、评测和工程化落地。

### 3.2 核心使用场景

- 用户输入技术关键词，系统返回相关专利、命中片段、图谱关系和摘要。
- 用户打开某一专利，系统总结其技术领域、背景问题、核心方案、关键部件和技术效果。
- 用户提出一个技术想法，系统检索相关专利并对比已有方案，给出相似点、差异点和可探索方向。
- 用户进行多轮对话，系统能够基于知识库持续引用专利证据，避免空泛回答。
- 用户查看知识图谱，理解专利、申请人、发明人、IPC 分类、技术问题、技术方案和关键组件之间的关系。

## 4. 项目目标

### 4.1 MVP 目标

- 能读取 `patant/` 下的 20 篇专利 Markdown 文件。
- 能解析专利基础字段：公开号/公告号、申请号、申请日、申请人/专利权人、发明人、IPC 分类、专利名称、摘要、权利要求、技术领域、背景技术、发明内容/实用新型内容。
- 能生成结构化 JSON 数据并落盘。
- 能构建向量索引，支持自然语言语义检索。
- 能基于检索结果调用大模型进行带引用的专利问答和总结。
- 能提供基础 API 或命令行入口完成演示。

### 4.2 进阶目标

- 构建专利知识图谱，支持实体和关系查询。
- 实现 Hybrid RAG：关键词检索 + 向量检索 + 图谱扩展 + 重排序。
- 引入 Agent 工作流：根据用户问题自动选择“检索、图谱查询、专利总结、相似度分析、报告生成”等工具。
- 加入对话记忆和任务状态管理，支持多轮复杂任务。
- 建立 RAG 评测集和自动评测指标，验证召回、忠实度、引用正确性和回答质量。
- 加入 tracing、日志、异常恢复、配置管理和可重复构建脚本。

## 5. 功能需求

### 5.1 数据接入与清洗

功能说明：

- 扫描本地 Markdown 专利文件。
- 统一使用 UTF-8 读取，避免 Windows 默认编码导致中文乱码。
- 移除或保留图片引用元数据，但不把图片路径混入正文语义索引。
- 识别 Markdown 标题、列表、段落、权利要求编号和说明书段落编号。
- 为每个专利生成唯一 `patent_id`，建议优先使用公开号/公告号。

验收标准：

- 20 篇 Markdown 文件均可被解析。
- 每篇专利至少能抽取标题、摘要、申请号或公开号、权利要求正文。
- 解析失败时有错误日志，不中断整个批处理。

### 5.2 专利结构化抽取

字段建议：

- `patent_id`：专利唯一标识。
- `publication_number`：公开号或公告号。
- `application_number`：申请号。
- `application_date`：申请日。
- `publication_date`：公开日或公告日。
- `patent_type`：发明专利申请、实用新型专利等。
- `title`：专利名称。
- `abstract`：摘要。
- `applicants`：申请人或专利权人。
- `inventors`：发明人。
- `agency`：代理机构。
- `ipc_classes`：IPC 分类号。
- `claims`：权利要求列表。
- `sections`：技术领域、背景技术、发明内容/实用新型内容、附图说明、具体实施方式等。
- `source_file`：源文件路径。

抽取方式：

- 第一优先级：规则解析，保证确定性和低成本。
- 第二优先级：LLM 结构化抽取，用于补全规则难以解析的字段。
- 第三优先级：人工校验或标注，用于构建小规模黄金数据集。

验收标准：

- 关键字段抽取准确率在抽样 20 篇中达到 90% 以上。
- 结构化结果通过 Pydantic Schema 校验。
- 每个字段保留来源位置或证据片段，便于调试和解释。

### 5.3 文档切分与向量索引

切分策略：

- 按专利结构分层切分，而不是粗暴固定字数切分。
- 权利要求、摘要、背景技术、发明内容、具体实施方式分别进入不同 chunk。
- 每个 chunk 保留 `patent_id`、章节名、权利要求编号、字符范围和源文件信息。

检索能力：

- 支持语义检索。
- 支持关键词检索。
- 支持按专利类型、IPC、申请人、发明人、日期过滤。
- 支持结果重排序。

验收标准：

- 用户输入“液氮罐运输固定”“肿瘤抢救车”“车载医疗设备”等查询时，能召回相关专利。
- 检索结果展示标题、摘要片段、命中章节、相似度和来源文件。

### 5.4 知识图谱构建

实体类型：

- `Patent`：专利。
- `Applicant`：申请人/专利权人。
- `Inventor`：发明人。
- `Agency`：代理机构。
- `IPCClass`：IPC 分类。
- `TechnicalField`：技术领域。
- `Problem`：技术问题。
- `Solution`：技术方案。
- `Component`：关键部件。
- `Effect`：技术效果。
- `Claim`：权利要求。

关系类型：

- `APPLIED_BY`：专利由申请人申请。
- `INVENTED_BY`：专利由发明人发明。
- `CLASSIFIED_AS`：专利属于 IPC 分类。
- `BELONGS_TO_FIELD`：专利属于技术领域。
- `SOLVES`：专利解决技术问题。
- `USES_COMPONENT`：方案使用关键部件。
- `HAS_EFFECT`：方案产生技术效果。
- `HAS_CLAIM`：专利包含权利要求。
- `SIMILAR_TO`：专利之间存在相似关系。

图谱构建方式：

- 专利基础元数据通过规则抽取入图。
- 技术问题、方案、部件、效果通过 LLM 结构化抽取，并要求输出证据句。
- 同义词和别名需要规范化，例如“医疗仓/手术仓/厢体”在不同上下文下可能需要归并或保留差异。

验收标准：

- 每篇专利至少生成 Patent、Applicant、Inventor、IPCClass、Claim 相关节点。
- 每篇专利至少抽取 3 个关键技术实体或技术效果。
- 图谱查询可回答“某申请人有哪些专利”“某 IPC 下有哪些技术问题”“某专利使用了哪些关键部件”等问题。

### 5.5 Agent 对话与工具调用

Agent 应具备的工具：

- `search_patents`：根据自然语言查询检索专利片段。
- `get_patent_detail`：获取某专利结构化详情。
- `query_knowledge_graph`：查询知识图谱实体和关系。
- `summarize_patent`：总结单篇专利。
- `compare_patents`：比较多篇专利的技术问题、方案和效果。
- `analyze_idea`：基于用户想法检索相似专利并给出差异分析。
- `generate_report`：生成 Markdown 分析报告。

Agent 工作流：

- 识别用户意图。
- 选择工具或组合工具。
- 检索证据。
- 必要时进行图谱扩展。
- 生成带引用回答。
- 对回答做自检：是否有证据、是否遗漏用户问题、是否出现无法验证的专利结论。

边界要求：

- 系统不能直接给出法律意义上的“可专利性结论”或“侵权结论”。
- 当证据不足时，应明确说明“当前 20 篇知识库中未检索到充分依据”。
- 回答应优先引用知识库内容，不能把大模型常识伪装成专利事实。

验收标准：

- 用户连续追问时，Agent 能记住当前讨论的专利或技术想法。
- 复杂问题可以拆成多个工具步骤，并输出可解释的中间依据。
- 回答中包含来源专利、章节或片段引用。

### 5.6 前后端分离产品形态

本项目最终应交付为前后端分离的完整应用，而不是只有命令行脚本或单一后端服务。

后端职责：

- 负责专利解析、索引构建、知识图谱构建、检索、Agent 工具调用和大模型访问。
- 通过 FastAPI 提供稳定 API。
- 对耗时任务提供后台任务或任务状态查询。
- 对 Agent 长回答和工具调用过程预留 SSE/WebSocket 流式接口。

前端职责：

- 负责专利检索、专利详情、Agent 对话、创意分析、图谱探索等用户界面。
- 不直接访问 LLM、向量库或图数据库。
- 展示检索证据、引用来源和 Agent 工具调用过程。
- 提供适合面试演示的完整工作台体验。

推荐前端技术栈：

- React + Vite + TypeScript。
- TanStack Query 管理 API 请求。
- React Router 管理页面路由。
- 图谱可视化后续可选 React Flow、ECharts Graph 或 Sigma.js。

### 5.7 后端 API

MVP API：

- `POST /ingest`：触发数据解析和索引构建。
- `GET /patents`：分页查看专利列表。
- `GET /patents/{patent_id}`：查看专利详情。
- `POST /search`：检索专利。
- `POST /chat`：基于知识库问答。
- `POST /idea/analyze`：分析用户技术想法。

前端页面：

- 专利列表页。
- 专利详情页。
- 智能检索页。
- 对话问答页。
- 创意分析页。
- 知识图谱探索页。

MVP 可以先完成 API 和命令行，但项目结构必须从一开始保留独立前端应用；完成基础 RAG 后，应尽快打通前端检索和对话页面。

## 6. 非功能需求

### 6.1 可维护性

- 使用模块化目录，区分数据解析、索引、图谱、Agent、API、配置和测试。
- 所有核心数据结构使用 Pydantic 定义。
- 配置项使用 `.env` 和配置文件管理，避免硬编码模型、数据库和路径。

### 6.2 可观测性

- 记录 ingestion、chunking、embedding、graph extraction、retrieval、agent tool call 的日志。
- 每次回答保存检索 query、召回 chunk、工具调用步骤和最终答案。
- 后续可接入 LangSmith、OpenTelemetry 或模型服务商 tracing。

### 6.3 可评测性

- 建立小型评测集：至少 20 个检索问题、10 个问答问题、5 个创意分析问题。
- 评测维度包括召回准确率、引用正确率、回答忠实度、结构化字段准确率和工具选择合理性。
- 每次改动检索策略或 prompt 后可以重复运行评测。

### 6.4 安全与合规

- 不上传本地专利数据到未知第三方服务。
- API Key 只存放在环境变量或本地配置中。
- 用户上传数据需要做文件类型、大小和内容校验。
- 回答中增加免责声明：系统提供技术信息辅助，不构成法律意见。

### 6.5 性能

- 20 篇专利的本地 MVP 构建时间应控制在 1 分钟内，不包含首次模型调用耗时。
- 检索响应应控制在 2 秒内，不包含大模型生成耗时。
- 设计上支持扩展到 1 万篇专利：异步任务、批量 embedding、增量索引、持久化数据库。

## 7. 推荐技术路线

### 7.1 核心技术栈

- 语言：Python 3.11+
- API：FastAPI
- 数据模型：Pydantic
- 文档解析：Markdown 解析 + 正则规则 + LLM 结构化补全
- 向量库：Chroma、Qdrant 或 Milvus，MVP 可先用 Chroma/Qdrant 本地模式
- 关键词检索：BM25
- 图数据库：Neo4j，MVP 也可先用 NetworkX/SQLite 存储图结构
- Agent 编排：LangGraph 或 OpenAI Agents SDK，优先选择具备状态管理、工具调用和 tracing 能力的方案
- 检索增强：Hybrid RAG、GraphRAG、reranker
- 评测：Ragas、DeepEval 或自定义 eval harness
- 前端：React/Vite 或 Streamlit，MVP 可先用 Streamlit 快速演示

### 7.2 前沿能力映射

- Agentic RAG：Agent 根据问题动态决定检索、图谱查询、总结、对比和报告生成。
- GraphRAG：先从知识图谱中扩展相关实体，再结合向量片段生成回答。
- Hybrid Search：BM25 解决精确词匹配，向量检索解决语义相似，重排序提升最终上下文质量。
- Structured Extraction：使用 JSON Schema/Pydantic 约束 LLM 输出，减少不可控文本。
- Tool Calling：把检索、图谱、详情查询、报告生成封装为工具。
- Memory/State：保留多轮对话中的当前专利、用户想法、已比较对象和任务状态。
- Evaluation Loop：每次优化 prompt、chunk 或检索策略时，用固定评测集回归。
- Observability：记录工具调用链路和检索证据，面试时可以展示系统如何“思考和查证”。
- MCP 扩展：后续可以把专利检索、图谱查询作为 MCP 工具暴露给其他 Agent 客户端。

## 8. 里程碑规划

### Phase 0：项目初始化

- 初始化 Git 仓库。
- 规划目录结构。
- 建立 Python 环境和依赖管理。
- 新建配置、日志、测试基础设施。

交付物：

- `README.md`
- `pyproject.toml`
- `.env.example`
- 基础目录结构

### Phase 1：数据解析与结构化

- 读取 `patant/` 下 Markdown。
- 实现专利字段规则抽取。
- 输出结构化 JSON。
- 增加解析单元测试。

交付物：

- `data/processed/patents.jsonl`
- `src/patent_rag/ingestion/`
- `tests/test_patent_parser.py`

### Phase 2：基础 RAG

- 实现结构化 chunk。
- 构建 embedding 和向量索引。
- 实现关键词检索和向量检索。
- 实现带引用问答。

交付物：

- `src/patent_rag/retrieval/`
- `src/patent_rag/llm/`
- `POST /search`
- `POST /chat`

### Phase 3：前端工作台 MVP

- 创建独立前端应用。
- 实现专利检索页。
- 实现专利详情页。
- 实现对话问答页。
- 接入后端 `/patents`、`/search`、`/chat` 接口。

交付物：

- `frontend/`
- 检索页面
- 专利详情页面
- Agent 对话页面

### Phase 4：知识图谱

- 定义图谱 Schema。
- 抽取实体和关系。
- 存储到图数据库或本地图结构。
- 实现图谱查询工具。

交付物：

- `src/patent_rag/graph/`
- `data/processed/graph_nodes.jsonl`
- `data/processed/graph_edges.jsonl`
- `query_knowledge_graph` 工具

### Phase 5：Agent 工作流

- 构建意图识别和工具路由。
- 实现专利总结、对比和创意分析工具。
- 加入多轮状态管理。
- 增加 tracing 和工具调用日志。

交付物：

- `src/patent_rag/agent/`
- `POST /idea/analyze`
- Agent 运行轨迹样例

### Phase 6：评测与演示

- 建立评测集。
- 运行自动评测。
- 完善前端图谱探索页、创意分析页和调试页。
- 完成面试项目讲解文档。

交付物：

- `evals/`
- `docs/ARCHITECTURE.md`
- `docs/INTERVIEW_GUIDE.md`
- 可运行 Demo

## 9. 验收指标

MVP 验收：

- 20 篇专利全部解析成功。
- 每篇专利生成结构化 JSON。
- 至少支持 5 类自然语言检索问题。
- 问答结果包含引用来源。
- 单篇专利总结覆盖技术领域、背景问题、核心方案、关键部件和技术效果。

进阶验收：

- 知识图谱中至少包含 10 类实体和 8 类关系。
- Agent 至少能稳定调用 5 个工具。
- 创意分析可以返回相似专利、相似点、差异点和建议探索方向。
- 评测集可一键运行，并输出指标报告。
- 系统具备本地启动说明和演示脚本。

## 10. 当前数据观察

- 当前根目录只有 `patant/` 数据目录，尚未形成工程项目结构。
- `patant/` 中共有 20 篇 Markdown 专利文档。
- 文档为中文 UTF-8 内容，Windows PowerShell 默认读取时可能出现乱码，后续代码必须显式使用 UTF-8。
- 文档中包含图片引用，例如 `_page_0_Picture_1.jpeg`，MVP 阶段先不处理图片 OCR，只保留图片路径元数据。
- 专利文本具有明显结构：头部元数据、摘要、权利要求、技术领域、背景技术、发明/实用新型内容等，适合规则解析优先的策略。

## 11. 风险与应对

- 风险：Markdown 由 PDF/OCR 转换而来，存在断行、空格、字段缺失。
  - 应对：解析前做文本规范化，抽取时保留置信度和原始证据。
- 风险：LLM 抽取实体可能幻觉。
  - 应对：要求每个实体关系带证据句，并用 Schema 校验。
- 风险：小数据集上演示效果好，但难以证明可扩展。
  - 应对：架构上保留批处理、增量索引和数据库持久化设计。
- 风险：Agent 回答过度自信。
  - 应对：增加证据不足时的拒答策略和法律意见边界。
- 风险：图谱构建复杂度较高。
  - 应对：先做元数据图谱，再做技术实体图谱，最后做 GraphRAG。

## 12. 参考资料

- LangGraph 官方文档：https://docs.langchain.com/oss/python/langgraph/overview
- Microsoft GraphRAG 官方文档：https://microsoft.github.io/graphrag/
- LlamaIndex Property Graph Index 文档：https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/
- Model Context Protocol 官方规范：https://modelcontextprotocol.io/specification/
- OpenAI Agents SDK 文档：https://openai.github.io/openai-agents-python/
- OpenAI Retrieval/File Search 文档：https://platform.openai.com/docs/guides/retrieval
