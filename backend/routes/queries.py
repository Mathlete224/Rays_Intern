"""RAG query routes."""
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from dependencies import get_rag
from rag_gemini import GeminiRAGPipeline, RetrievalFilters
from models import AskRequest, AskResponse, BackfillResponse, ChunkRef

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    req: AskRequest,
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    filters = RetrievalFilters(
        document_ids=req.document_ids,
        filenames=req.filenames,
        page_min=req.page_min,
        page_max=req.page_max,
        sender_names=req.sender_names,
        sender_companies=req.sender_companies,
        written_date_from=req.written_date_from,
        written_date_to=req.written_date_to,
    )
    result = await run_in_threadpool(
        rag.answer_question, req.question, top_k=req.top_k, filters=filters
    )
    return AskResponse(
        answer=result["answer"],
        chunks_used=[ChunkRef(**c) for c in result["chunks_used"]],
    )


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_embeddings(
    batch_size: int = 64,
    max_batches: Optional[int] = None,
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    count = await run_in_threadpool(
        rag.backfill_embeddings, batch_size=batch_size, max_batches=max_batches
    )
    return BackfillResponse(embedded_count=count)
