# Frontend GraphRAG Evidence Design

## Goal

Make the current GraphRAG behavior visible in the frontend. After a user asks a question through RAG or Agent mode, the page should show not only the generated answer and text chunks, but also the graph evidence that influenced the answer.

## Current Context

The frontend already has four major areas:

- Agent workspace
- Knowledge graph search
- Patent RAG question answering
- Patent search and patent list

The backend now returns richer graph evidence through `graph_sources`, including technical fields, problems, components, solutions, effects, matched terms, supporting chunks, and relation summaries. The current frontend only renders a small part of that graph evidence.

`frontend/src/App.tsx` also contains visible Chinese text corruption. This must be fixed as part of the same frontend pass because unreadable labels make the GraphRAG workflow hard to test.

## Selected Approach

Use an evidence-enhanced frontend rather than building a graph visualization in this step.

This keeps the current page structure and adds clearer evidence panels where the existing RAG and Agent responses already appear. It gives immediate testing value with a small implementation surface.

## User Experience

When a user runs RAG or Agent:

- The answer remains at the top of the result area.
- Source chunks are shown as text evidence with `[S]` source ids.
- Graph evidence is shown as separate graph evidence cards with `[G]` source ids.
- Each graph card shows patent id, title, score, matched terms, supporting chunk ids, and relation summary.
- Each graph card groups technical evidence into these sections:
  - Technical fields
  - Technical problems
  - Key components
  - Technical solutions
  - Technical effects

If a graph evidence field is empty, that group is omitted rather than rendered as an empty section.

## Components And Data Flow

`frontend/src/api/client.ts`

- Extend `RagGraphSource` with the technical evidence arrays now returned by the backend:
  - `technical_fields`
  - `problems`
  - `components`
  - `solutions`
  - `effects`

`frontend/src/App.tsx`

- Fix corrupted Chinese display text.
- Add small reusable render helpers for graph evidence chips and graph evidence cards.
- Reuse the same graph evidence card rendering in both RAG and Agent results.
- Keep the existing API calls and data flow unchanged.

`frontend/src/styles.css`

- Add focused styles for evidence groups and chips.
- Keep the operational dashboard look: dense, readable, and not decorative.
- Preserve responsive behavior on narrow screens.

## Error Handling

No new API error mode is introduced.

If the backend returns no `graph_sources`, the graph evidence section is not rendered. Existing loading and error states remain in place for API, RAG, Agent, and graph search failures.

## Testing

Verification should include:

- TypeScript/Vite production build with `npm run build`.
- A frontend smoke test using a GraphRAG-style question such as:
  - `哪些专利使用清洗箱、喷嘴或雾化器？它们分别解决了什么技术问题？`
- Confirm the UI shows answer text, text sources, graph sources, and grouped technical evidence without obvious overflow on desktop and mobile widths.

## Out Of Scope

This step will not implement a node-edge graph visualization. That can be a later feature after we add a graph visualization-friendly API response.
