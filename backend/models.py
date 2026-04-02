"""Pydantic request/response schemas for the FastAPI backend."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


# ── Documents ──────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: int
    filename: str
    total_pages: int
    file_size_bytes: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UploadResult(BaseModel):
    status: str          # "success" | "skipped" | "error"
    filename: str
    document_id: Optional[int] = None
    total_pages: Optional[int] = None
    message: Optional[str] = None


# ── Queries ────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    document_ids: Optional[List[int]] = None
    filenames: Optional[List[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None


class ChunkRef(BaseModel):
    chunk_id: str
    document_id: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = {}


class AskResponse(BaseModel):
    answer: str
    chunks_used: List[ChunkRef]


class BackfillResponse(BaseModel):
    embedded_count: int


# ── Agent ──────────────────────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    goal: str
    top_k: int = 3
    document_ids: Optional[List[int]] = None
    filenames: Optional[List[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None


class SubQueryResult(BaseModel):
    question: str
    answer: str
    chunks_used: List[ChunkRef]


class AgentResult(BaseModel):
    goal: str
    sub_queries: List[SubQueryResult]
    synthesis: str


# ── Canvas ─────────────────────────────────────────────────────────────────

class CanvasMeta(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CanvasCreateRequest(BaseModel):
    name: str = "Untitled Canvas"


class CanvasSaveRequest(BaseModel):
    name: Optional[str] = None
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


class CanvasDetail(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
