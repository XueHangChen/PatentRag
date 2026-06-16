"""FastAPI application entrypoint."""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from patent_rag.agent import run_patent_agent
from patent_rag.domain import PatentDocument
from patent_rag.graph import find_patents_by_keyword, read_graph_json, summarize_graph
from patent_rag.ingestion.jsonl import read_jsonl
from patent_rag.rag import answer_patent_question
from patent_rag.retrieval import search_chunks, search_hybrid_chunks, search_vector_chunks


def _resolve_graph_path(graph_dir: Path) -> Path:
    enhanced_graph_path = graph_dir / "patent_graph_llm_full20_fixed.json"
    if enhanced_graph_path.exists():
        return enhanced_graph_path
    return graph_dir / "patent_graph.json"


PROCESSED_PATENTS_PATH = Path("data/processed/patents.jsonl")
PROCESSED_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CHROMA_INDEX_PATH = Path("data/indexes/chroma")
GRAPH_PATH = _resolve_graph_path(Path("data/graph"))


class PatentListItem(BaseModel):
    """Patent summary returned by the list endpoint."""

    patent_id: str
    title: str | None = None
    patent_type: str
    publication_number: str | None = None
    application_number: str | None = None
    application_date: str | None = None
    publication_date: str | None = None
    applicants: list[str]
    inventors: list[str]
    ipc_classes: list[str]
    claim_count: int
    section_count: int


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    mode: Literal["keyword", "vector", "hybrid"] = "keyword"


class RagAskRequest(BaseModel):
    """RAG question-answering request payload."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)
    retrieval_mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    use_graph: bool = False
    graph_top_k: int = Field(default=3, ge=0, le=8)


class AgentRunRequest(BaseModel):
    """Agent task request payload."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)
    retrieval_mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    use_graph: bool = True
    graph_top_k: int = Field(default=3, ge=0, le=8)


def create_app() -> FastAPI:
    """Create the API application."""

    app = FastAPI(
        title="Patent KG Agent",
        version="0.1.0",
        description="Patent knowledge graph and Agentic RAG service.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/patents")
    def list_patents() -> list[PatentListItem]:
        documents = _load_patent_documents()
        return [_to_patent_list_item(document) for document in documents]

    @app.get("/patents/{patent_id}")
    def get_patent(patent_id: str) -> dict:
        for document in _load_patent_documents():
            if document.metadata.patent_id == patent_id:
                return document.model_dump(mode="json", exclude_none=True)
        raise HTTPException(status_code=404, detail=f"Patent not found: {patent_id}")

    @app.post("/search")
    def search(request: SearchRequest) -> dict:
        hits = _search_patent_chunks(request)
        return {
            "query": request.query,
            "top_k": request.top_k,
            "mode": request.mode,
            "hits": [hit.model_dump(mode="json") for hit in hits],
        }

    @app.post("/rag/ask")
    def ask_rag(request: RagAskRequest) -> dict:
        _ensure_rag_dependencies_exist(request)
        answer = answer_patent_question(
            request.question,
            chunks_path=PROCESSED_CHUNKS_PATH,
            index_path=CHROMA_INDEX_PATH,
            graph_path=GRAPH_PATH,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_graph=request.use_graph,
            graph_top_k=request.graph_top_k,
        )
        return answer.model_dump(mode="json")

    @app.post("/agent/run")
    def run_agent(request: AgentRunRequest) -> dict:
        _ensure_agent_dependencies_exist(request)
        result = run_patent_agent(
            request.query,
            patents_path=PROCESSED_PATENTS_PATH,
            chunks_path=PROCESSED_CHUNKS_PATH,
            index_path=CHROMA_INDEX_PATH,
            graph_path=GRAPH_PATH,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_graph=request.use_graph,
            graph_top_k=request.graph_top_k,
        )
        return result.model_dump(mode="json")

    @app.get("/graph/stats")
    def graph_stats() -> dict:
        graph = _load_graph()
        return summarize_graph(graph).model_dump(mode="json")

    @app.get("/graph/search")
    def graph_search(
        keyword: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        graph = _load_graph()
        return {
            "keyword": keyword,
            "matches": find_patents_by_keyword(graph, keyword, limit=limit),
        }

    return app


app = create_app()


def _load_patent_documents() -> list[PatentDocument]:
    if not PROCESSED_PATENTS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Processed patents are missing. Run `python scripts\\ingest_patents.py` first.",
        )
    return read_jsonl(PROCESSED_PATENTS_PATH, PatentDocument)


def _search_patent_chunks(request: SearchRequest):
    if request.mode == "keyword":
        _ensure_chunks_exist()
        return search_chunks(PROCESSED_CHUNKS_PATH, request.query, top_k=request.top_k)

    if request.mode == "vector":
        _ensure_vector_index_exists()
        return search_vector_chunks(CHROMA_INDEX_PATH, request.query, top_k=request.top_k)

    _ensure_chunks_exist()
    _ensure_vector_index_exists()
    return search_hybrid_chunks(
        PROCESSED_CHUNKS_PATH,
        CHROMA_INDEX_PATH,
        request.query,
        top_k=request.top_k,
    )


def _ensure_chunks_exist() -> None:
    if not PROCESSED_CHUNKS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Retrieval chunks are missing. Run `python scripts\\build_chunks.py` first.",
        )


def _ensure_vector_index_exists() -> None:
    if not CHROMA_INDEX_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Chroma index is missing. Run `python scripts\\build_vector_index.py` first.",
        )


def _load_graph():
    _ensure_graph_exists()
    return read_graph_json(GRAPH_PATH)


def _ensure_graph_exists() -> None:
    if not GRAPH_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph is missing. Run `python scripts\\build_graph.py` first.",
        )


def _ensure_rag_dependencies_exist(request: RagAskRequest) -> None:
    _ensure_chunks_exist()
    if request.retrieval_mode in {"vector", "hybrid"}:
        _ensure_vector_index_exists()
    if request.use_graph:
        _ensure_graph_exists()


def _ensure_agent_dependencies_exist(request: AgentRunRequest) -> None:
    _ensure_chunks_exist()
    if request.retrieval_mode in {"vector", "hybrid"}:
        _ensure_vector_index_exists()
    if request.use_graph:
        _ensure_graph_exists()


def _to_patent_list_item(document: PatentDocument) -> PatentListItem:
    metadata = document.metadata
    return PatentListItem(
        patent_id=metadata.patent_id,
        title=metadata.title,
        patent_type=metadata.patent_type.value,
        publication_number=metadata.publication_number,
        application_number=metadata.application_number,
        application_date=metadata.application_date,
        publication_date=metadata.publication_date,
        applicants=metadata.applicants,
        inventors=metadata.inventors,
        ipc_classes=metadata.ipc_classes,
        claim_count=len(document.claims),
        section_count=len(document.sections),
    )
