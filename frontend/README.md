# Patent KG Agent Frontend

独立前端应用，负责专利检索、专利详情、Agent 对话、创意分析和知识图谱探索。

## 技术栈

- React
- Vite
- TypeScript

## 开发命令

```powershell
npm install
npm run dev
```

默认后端地址通过 `.env` 配置：

```text
VITE_API_BASE_URL=http://localhost:8000
```

## 页面规划

- `/`：项目工作台首页
- `/search`：专利检索
- `/patents/:id`：专利详情
- `/chat`：Agent 对话
- `/idea`：创意分析
- `/graph`：知识图谱探索

