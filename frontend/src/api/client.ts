const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type PatentListItem = {
  patent_id: string;
  title: string | null;
  patent_type: string;
  publication_number: string | null;
  application_number: string | null;
  application_date: string | null;
  publication_date: string | null;
  applicants: string[];
  inventors: string[];
  ipc_classes: string[];
  claim_count: number;
  section_count: number;
};

export type SearchHit = {
  chunk_id: string;
  patent_id: string;
  title: string;
  section: string;
  claim_number: number | null;
  score: number;
  snippet: string;
  source_file: string;
};

export type SearchResponse = {
  query: string;
  top_k: number;
  mode: "keyword" | "vector" | "hybrid";
  hits: SearchHit[];
};

export type RagSource = {
  source_id: string;
  chunk_id: string;
  patent_id: string;
  title: string;
  section: string;
  claim_number: number | null;
  score: number;
  snippet: string;
  source_file: string;
};

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
  relation_summary: string;
};

export type RagAnswer = {
  question: string;
  answer: string;
  retrieval_mode: "keyword" | "vector" | "hybrid";
  top_k: number;
  use_graph: boolean;
  sources: RagSource[];
  graph_sources: RagGraphSource[];
};

export type AgentIntent = "patent_qa" | "patent_summary" | "idea_analysis";

export type AgentToolName =
  | "search_patents"
  | "summarize_patent"
  | "graph_search"
  | "graph_rag_answer";

export type AgentPlannedStep = {
  tool_name: AgentToolName;
  reason: string;
};

export type AgentPlan = {
  query: string;
  intent: AgentIntent;
  rationale: string;
  steps: AgentPlannedStep[];
};

export type AgentToolStep = {
  tool_name: string;
  tool_input: Record<string, unknown>;
  status: "success" | "skipped" | "error";
  observation: string;
  output: Record<string, unknown>;
};

export type AgentRunResult = {
  query: string;
  intent: AgentIntent;
  answer: string;
  retrieval_mode: "keyword" | "vector" | "hybrid";
  top_k: number;
  use_graph: boolean;
  plan: AgentPlan;
  steps: AgentToolStep[];
  sources: RagSource[];
  graph_sources: RagGraphSource[];
};

export type GraphStats = {
  node_count: number;
  edge_count: number;
  node_labels: Record<string, number>;
  edge_relations: Record<string, number>;
};

export type GraphKeywordMatch = {
  patent_id: string;
  title: string;
  keywords: string[];
};

export type GraphSearchResponse = {
  keyword: string;
  matches: GraphKeywordMatch[];
};

export async function getHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  return response.json() as Promise<{ status: string }>;
}

export async function listPatents(): Promise<PatentListItem[]> {
  const response = await fetch(`${API_BASE_URL}/patents`);

  if (!response.ok) {
    throw new Error(`List patents failed: ${response.status}`);
  }

  return response.json() as Promise<PatentListItem[]>;
}

export async function searchPatents(query: string, topK = 8): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      top_k: topK
    })
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.status}`);
  }

  return response.json() as Promise<SearchResponse>;
}

export async function askPatentRag(
  question: string,
  topK = 5,
  useGraph = true
): Promise<RagAnswer> {
  const response = await fetch(`${API_BASE_URL}/rag/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      question,
      top_k: topK,
      retrieval_mode: "hybrid",
      use_graph: useGraph,
      graph_top_k: 3
    })
  });

  if (!response.ok) {
    throw new Error(`RAG ask failed: ${response.status}`);
  }

  return response.json() as Promise<RagAnswer>;
}

export async function runPatentAgent(
  query: string,
  topK = 5,
  useGraph = true
): Promise<AgentRunResult> {
  const response = await fetch(`${API_BASE_URL}/agent/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      top_k: topK,
      retrieval_mode: "hybrid",
      use_graph: useGraph,
      graph_top_k: 3
    })
  });

  if (!response.ok) {
    throw new Error(`Agent run failed: ${response.status}`);
  }

  return response.json() as Promise<AgentRunResult>;
}

export async function getGraphStats(): Promise<GraphStats> {
  const response = await fetch(`${API_BASE_URL}/graph/stats`);

  if (!response.ok) {
    throw new Error(`Graph stats failed: ${response.status}`);
  }

  return response.json() as Promise<GraphStats>;
}

export async function searchGraph(keyword: string, limit = 8): Promise<GraphSearchResponse> {
  const params = new URLSearchParams({
    keyword,
    limit: String(limit)
  });
  const response = await fetch(`${API_BASE_URL}/graph/search?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Graph search failed: ${response.status}`);
  }

  return response.json() as Promise<GraphSearchResponse>;
}
