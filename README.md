# Patent KG Agent

面向中文专利文档的知识图谱与 Agentic RAG 项目。

当前目标是把 `patant/` 目录下的 20 篇专利 Markdown 文档，逐步建设成一个可检索、可总结、可对话、可图谱推理的企业级 AI Agent 实战项目。

## 当前阶段

- 已完成：项目需求文档 `docs/PRD.md`
- 已完成：技术架构文档 `docs/ARCHITECTURE.md`
- 已完成：Markdown 专利解析、质量报告、chunk 构建
- 已完成：BM25、Chroma 向量检索、Hybrid RAG 问答
- 已完成：FastAPI 后端与 React 前端工作台
- 已完成：检索评测、规则 postprocess/rerank
- 已完成：规则版知识图谱构建、图谱 API 和前端图谱探索
- 已完成：GraphRAG 证据融合
- 已完成：Agent 工具调用链、规则 Planner 和前端 Agent 工作台
- 下一步：Agent 任务评测、流式输出和更细粒度技术关系抽取

## 项目结构

```text
docs/                 项目文档
patant/               当前原始专利 Markdown 文件
configs/              配置模板
data/                 中间数据、索引和图谱产物
evals/                评测集和评测脚本
frontend/             独立前端应用
logs/                 本地运行日志
scripts/              一次性脚本和运维脚本
src/patent_rag/       项目源码
tests/                自动化测试
```

## 前后端分离

本项目采用前后端分离形态：

- 后端：FastAPI + 专利解析 + RAG + 知识图谱 + Agent 工具编排。
- 前端：React + Vite + TypeScript，负责检索、详情、对话、创意分析和图谱探索界面。
- 通信：REST JSON 为主，后续对 Agent 流式输出预留 SSE/WebSocket。

## 计划能力

- 专利 Markdown 解析
- 专利结构化字段抽取
- 基础语义检索和关键词检索
- Hybrid RAG
- 专利知识图谱
- GraphRAG
- 多工具 Agent 工作流
- 专利总结、对比和创意分析
- 自动化评测和 tracing

## 本地开发

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

如果你没有激活虚拟环境，可以直接使用：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

启动后端 API：

```powershell
uvicorn patent_rag.api.main:app --reload --app-dir src
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

## 数据解析

将 `patant/` 下的专利 Markdown 解析为结构化 JSONL：

```powershell
python scripts\ingest_patents.py
```

默认输出：

```text
data/processed/patents.jsonl
```

检查结构化数据质量：

```powershell
python scripts\inspect_processed_patents.py
```

构建检索 chunk：

```powershell
python scripts\build_chunks.py
```

默认输出：

```text
data/processed/chunks.jsonl
```

命令行检索验证：

```powershell
python scripts\search_patents.py "液氮罐运输固定" --top-k 3
```

安装 Chroma 向量检索依赖：

```powershell
pip install -e ".[vector]"
```

构建 Chroma 向量索引：

```powershell
python scripts\build_vector_index.py
```

默认输出：

```text
data/indexes/chroma
```

命令行向量检索验证：

```powershell
python scripts\vector_search_patents.py "液氮罐运输固定" --top-k 3
```

使用 Qwen/DashScope embedding 时，先在 `.env` 中配置：

```dotenv
PATENT_RAG_EMBEDDING_PROVIDER=dashscope
PATENT_RAG_EMBEDDING_MODEL=text-embedding-v4
PATENT_RAG_DASHSCOPE_API_KEY=your-dashscope-api-key
PATENT_RAG_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

然后重新构建向量索引：

```powershell
python scripts\build_vector_index.py --embedding-provider dashscope --embedding-model text-embedding-v4
```

再使用同一个 embedding provider 查询：

```powershell
python scripts\vector_search_patents.py "低温样本运输时如何避免容器碰撞" --embedding-provider dashscope --embedding-model text-embedding-v4 --top-k 3
```

后端 `/search` 支持三种检索模式：

```json
{"query": "液氮罐运输固定", "top_k": 5, "mode": "keyword"}
{"query": "液氮罐运输固定", "top_k": 5, "mode": "vector"}
{"query": "液氮罐运输固定", "top_k": 5, "mode": "hybrid"}
```

命令行 RAG 问答：

```powershell
python scripts\ask_patent_rag.py "低温样本运输时如何避免容器碰撞？" --retrieval-mode hybrid --embedding-provider dashscope --embedding-model text-embedding-v4 --llm-provider dashscope --llm-model qwen-plus --top-k 5
```

命令行 GraphRAG 问答：

```powershell
python scripts\ask_patent_rag.py "低温样本运输时如何避免容器碰撞？" --use-graph --retrieval-mode hybrid --embedding-provider dashscope --embedding-model text-embedding-v4 --llm-provider dashscope --llm-model qwen-plus --top-k 5
```

RAG 问答流程：

```text
用户问题 -> hybrid search 检索证据 -> 构造带引用的 prompt -> Qwen 生成回答 -> 返回 answer + sources
```

GraphRAG 问答流程：

```text
用户问题
-> hybrid search 检索原文 chunk
-> 根据命中的专利和问题关键词查询知识图谱邻域
-> 构造 [S] 原文证据 + [G] 图谱证据 prompt
-> Qwen 生成回答
-> 返回 answer + sources + graph_sources
```

命令行 Agent 任务：

```powershell
python scripts\run_patent_agent.py "我想设计一个低温样本运输装置，现有专利有哪些方案？还能怎么改进？" --retrieval-mode hybrid --embedding-provider dashscope --embedding-model text-embedding-v4 --llm-provider dashscope --llm-model qwen-plus --top-k 5
```

Agent API：

```text
POST /agent/run
```

示例请求：

```json
{
  "query": "我想设计一个低温样本运输装置，现有专利有哪些方案？还能怎么改进？",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "use_graph": true,
  "graph_top_k": 3
}
```

Agent 执行流程：

```text
用户目标
-> 规则 Planner 判断意图
-> 生成工具调用计划
-> search_patents / graph_search / summarize_patent / graph_rag_answer
-> 记录每一步 tool trace
-> 返回 answer + plan + steps + sources + graph_sources
```

前端 Agent 工作台会展示：

```text
intent 识别结果
plan 工具调用计划
tool trace 执行过程
最终回答
原文证据和图谱证据
```

检索质量评测：

```powershell
python scripts\evaluate_retrieval.py --modes keyword --top-k-values 1,3,5
```

使用 Qwen embedding 对向量检索和混合检索评测：

```powershell
python scripts\evaluate_retrieval.py --modes vector,hybrid --embedding-provider dashscope --embedding-model text-embedding-v4 --top-k-values 1,3,5
```

开启规则 postprocess/rerank 后评测：

```powershell
python scripts\evaluate_retrieval.py --modes hybrid --embedding-provider dashscope --embedding-model text-embedding-v4 --postprocess --top-k-values 1,3,5
```

构建规则版专利知识图谱：

```powershell
python scripts\build_graph.py
```

默认输出：

```text
data/graph/patent_graph.json
```

查看图谱统计和关键词关联专利：

```powershell
python scripts\inspect_graph.py --keyword 液氮罐 --limit 5
```

后端图谱接口：

```text
GET /graph/stats
GET /graph/search?keyword=液氮罐&limit=5
```

前端工作台会调用这些接口展示图谱节点、关系统计，以及关键词关联的专利结果。

前端开发命令将在依赖安装后使用：

```powershell
cd frontend
npm install
npm run dev
```

## 文档

- `docs/PRD.md`：需求文档
- `docs/ARCHITECTURE.md`：技术架构和项目结构
