# Frontend GraphRAG Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GraphRAG evidence visible in the React frontend and fix corrupted Chinese UI labels.

**Architecture:** Keep the existing single-page Vite React structure. Extend the API response type for graph evidence, then add reusable rendering helpers in `App.tsx` so RAG and Agent share one graph evidence card layout.

**Tech Stack:** React 18, TypeScript, Vite, plain CSS, existing Patent RAG FastAPI backend.

---

## File Structure

- Modify `frontend/src/api/client.ts`
  - Extend `RagGraphSource` with backend graph evidence fields.
- Modify `frontend/src/App.tsx`
  - Replace corrupted Chinese text.
  - Add helper functions for source labels, chip groups, and graph evidence cards.
  - Reuse graph evidence cards in RAG and Agent results.
- Modify `frontend/src/styles.css`
  - Add graph evidence section, group, and chip styles.
  - Keep the existing operational dashboard layout.
- Verify with `npm run build` from `frontend`.

---

### Task 1: Extend Graph Source Types

**Files:**
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Add technical graph evidence fields to `RagGraphSource`**

Update the type to include these properties:

```ts
export type RagGraphSource = {
  source_id: string;
  patent_id: string;
  title: string;
  score: number;
  matched_terms: string[];
  supporting_chunk_ids: string[];
  keywords: string[];
  applicants: string[];
  inventors: string[];
  ipc_classes: string[];
  section_names: string[];
  claim_numbers: number[];
  technical_fields: string[];
  problems: string[];
  components: string[];
  solutions: string[];
  effects: string[];
  relation_summary: string;
};
```

- [ ] **Step 2: Run frontend type check through build**

Run:

```bash
cd frontend
npm run build
```

Expected before the UI code is updated: TypeScript should still compile if only the type is extended, because adding fields to an object type used for API responses does not require new render code.

---

### Task 2: Restore Readable Chinese UI Text

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Replace corrupted constants and labels**

Use readable Chinese text for sample queries, default questions, intent labels, headings, buttons, placeholders, notices, and aria labels. Keep the same state variables and functions.

Use these values:

```ts
const sampleQueries = ["液氮罐运输固定", "心脏支架材料 生物降解", "小儿智能雾化器"];
const intentLabels: Record<AgentRunResult["intent"], string> = {
  patent_qa: "专利问答",
  patent_summary: "专利总结",
  idea_analysis: "创意分析"
};
```

The RAG default question should be:

```ts
"低温样本运输时如何避免容器磕碰？"
```

The graph search default keyword should be:

```ts
"液氮罐"
```

The Agent default task should be:

```ts
"我想设计一个低温样本运输装置，现有专利有哪些方案？还能怎么改进？"
```

- [ ] **Step 2: Keep behavior unchanged**

Do not rename state variables, API functions, or response types in this task. This task only restores readable UI text.

---

### Task 3: Add Reusable Graph Evidence Rendering

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Import the graph source type**

Add `RagGraphSource` to the existing import list from `./api/client`.

- [ ] **Step 2: Add helper functions above `export default function App()`**

Add these helpers:

```tsx
type EvidenceGroup = {
  label: string;
  values: string[];
};

function formatScore(score: number): string {
  return Number.isFinite(score) ? score.toFixed(2) : "0.00";
}

function renderChipGroup(label: string, values: string[]) {
  if (values.length === 0) {
    return null;
  }

  return (
    <div className="evidence-group" key={label}>
      <span className="evidence-group-label">{label}</span>
      <div className="keyword-row evidence-chip-row">
        {values.map((value) => (
          <span key={value}>{value}</span>
        ))}
      </div>
    </div>
  );
}

function renderGraphEvidenceCard(source: RagGraphSource) {
  const groups: EvidenceGroup[] = [
    { label: "技术领域", values: source.technical_fields },
    { label: "技术问题", values: source.problems },
    { label: "关键组件", values: source.components },
    { label: "技术方案", values: source.solutions },
    { label: "技术效果", values: source.effects }
  ];

  return (
    <article className="source-item graph-source-item" key={source.source_id}>
      <div className="result-title-row">
        <h3>
          [{source.source_id}] {source.title}
        </h3>
        <span>{formatScore(source.score)}</span>
      </div>
      <div className="result-meta">
        <span>{source.patent_id}</span>
        <span>{source.keywords.length} 个关键词</span>
        <span>{source.supporting_chunk_ids.length} 个支撑片段</span>
        <span>{source.claim_numbers.length} 项权利要求</span>
      </div>
      <p>{source.relation_summary}</p>
      <div className="evidence-groups">{groups.map((group) => renderChipGroup(group.label, group.values))}</div>
      {source.matched_terms.length > 0 ? renderChipGroup("命中词", source.matched_terms) : null}
      {source.supporting_chunk_ids.length > 0 ? (
        <div className="supporting-chunks">
          <span>支撑片段</span>
          <code>{source.supporting_chunk_ids.join(" / ")}</code>
        </div>
      ) : null}
    </article>
  );
}
```

- [ ] **Step 3: Replace Agent graph source rendering**

Replace the Agent graph source `map` body with:

```tsx
{agentResult.graph_sources.slice(0, 3).map((source) => renderGraphEvidenceCard(source))}
```

- [ ] **Step 4: Replace RAG graph source rendering**

Replace the RAG graph source `map` body with:

```tsx
{ragAnswer.graph_sources.map((source) => renderGraphEvidenceCard(source))}
```

---

### Task 4: Add Evidence Styles

**Files:**
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Add graph evidence spacing and labels**

Add these styles near the existing `.keyword-row` and `.graph-source-item` styles:

```css
.graph-rag-source-list,
.agent-graph-source-list {
  padding-top: 0;
}

.evidence-groups {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}

.evidence-group {
  min-width: 0;
}

.evidence-group-label {
  display: block;
  color: #2f4968;
  font-size: 13px;
  font-weight: 700;
}

.evidence-chip-row {
  margin-top: 8px;
}

.supporting-chunks {
  display: grid;
  gap: 6px;
  margin-top: 12px;
  color: #526070;
  font-size: 13px;
}

.supporting-chunks code {
  display: block;
  max-width: 100%;
  padding: 8px;
  border: 1px solid #d8e0ec;
  border-radius: 8px;
  background: #ffffff;
  color: #334155;
  overflow-wrap: anywhere;
}
```

- [ ] **Step 2: Ensure buttons disable consistently**

If button disabled styles remain split across `.search-form`, `.qa-form`, `.agent-form`, and `.graph-form`, optionally consolidate them without changing visual behavior:

```css
.search-form button:disabled,
.graph-form button:disabled,
.qa-form button:disabled,
.agent-form button:disabled {
  cursor: wait;
  opacity: 0.65;
}
```

---

### Task 5: Verify Frontend Build And Smoke Behavior

**Files:**
- Verify only

- [ ] **Step 1: Run production build**

Run:

```bash
cd frontend
npm run build
```

Expected:

```text
✓ built in
```

- [ ] **Step 2: Start frontend dev server**

Run:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Expected:

```text
Local:   http://127.0.0.1:5173/
```

- [ ] **Step 3: Manual smoke test**

With the backend running, ask:

```text
哪些专利使用清洗箱、喷嘴或雾化器？它们分别解决了什么技术问题？
```

Expected:

- The answer area renders readable Chinese.
- Text sources render as source cards.
- Graph sources render as graph evidence cards.
- Graph evidence cards show grouped technical evidence when the backend returns it.
- Long chip text and supporting chunk ids wrap rather than overflowing.

---

## Self-Review

- Spec coverage: The plan covers type updates, Chinese text repair, reusable graph evidence rendering, CSS support, build verification, and smoke testing.
- Placeholder scan: No TODO/TBD placeholders remain.
- Type consistency: `RagGraphSource` fields match the backend evidence fields named in the design spec.
