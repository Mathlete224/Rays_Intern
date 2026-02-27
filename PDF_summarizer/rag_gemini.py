"""
RAG pipeline: search on verbalized_summary, answer from raw_content.

Golden rule: Store verbalization for searching, keep raw content for answering.
"""
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from dotenv import load_dotenv
import google.generativeai as genai

from database import DatabaseManager, PDFChunk

# Use model that outputs 3072 dims (schema expects Vector(3072))
# models/embedding-001 or models/text-embedding-005; gemini-embedding-001 defaults to 3072
EMBEDDING_MODEL = "models/text-embedding-003-small"
GENERATION_MODEL = "gemini-1.5-pro"

load_dotenv()


def _configure_gemini(api_key: Optional[str] = None) -> None:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)


def embed_text(text: str) -> List[float]:
    if not text.strip():
        return []
    _configure_gemini()
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return result["embedding"]


@dataclass
class RetrievalFilters:
    document_ids: Optional[Sequence[int]] = None
    filenames: Optional[Sequence[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None


class GeminiRAGPipeline:
    """RAG over verbalized pages (text + chart descriptions)."""

    def __init__(self, database_url: str):
        _configure_gemini()
        self.db = DatabaseManager(database_url)
        self.model = genai.GenerativeModel(GENERATION_MODEL)

    def backfill_embeddings(
        self,
        batch_size: int = 64,
        max_batches: Optional[int] = None,
    ) -> int:
        """Generate embeddings from verbalized_summary for chunks that don't have them."""
        total = 0
        batches = 0

        while True:
            if max_batches is not None and batches >= max_batches:
                break

            chunks = self.db.get_chunks_without_embedding(limit=batch_size)
            if not chunks:
                break
            for chunk in chunks:
                text = (
                    (chunk.raw_content or "") +
                    "\n\n" +
                    (chunk.verbalized_summary or "")
                ).strip()
                if not text:
                    continue
                emb = embed_text(text)
                if emb:
                    self.db.upsert_chunk_embedding(chunk.id, emb)
                    total += 1

            batches += 1

        return total

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[PDFChunk]:
        """Vector search over verbalized_summary embeddings."""
        if filters is None:
            filters = RetrievalFilters()

        query_emb = embed_text(query)
        return self.db.semantic_search_chunks(
            query_embedding=query_emb,
            limit=top_k,
            document_ids=filters.document_ids,
            filenames=filters.filenames,
            page_min=filters.page_min,
            page_max=filters.page_max,
        )

    def _build_context(self, chunks: Iterable[PDFChunk]) -> str:
        """Use raw_content for answering (original data), not verbalized summary."""
        parts = []
        for c in chunks:
            meta = c.metadata_ or {}
            pg = meta.get("page_number", c.page_number) or "?"
            header = f"[doc_id={c.document_id}, page={pg}]"
            parts.append(f"{header}\n{c.raw_content}\n")
        return "\n\n".join(parts)

    def answer_question(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[RetrievalFilters] = None,
    ) -> dict:
        """Search on verbalized_summary, answer from raw_content."""
        chunks = self.retrieve_relevant_chunks(question, top_k=top_k, filters=filters)
        context = self._build_context(chunks)

        system = (
            "You are a financial analysis assistant. Use ONLY the provided context "
            "(original document content from reports) to answer. If not derivable, say you don't know."
        )
        prompt = (
            f"{system}\n\nContext:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer clearly. Reference document/page when relevant."
        )

        response = self.model.generate_content(prompt)
        answer = response.text if hasattr(response, "text") else str(response)

        return {
            "answer": answer,
            "chunks_used": [
                {"chunk_id": str(c.id), "document_id": c.document_id, "page_number": c.page_number}
                for c in chunks
            ],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gemini RAG over verbalized PDF pages")
    parser.add_argument(
        "--db-url",
        default=os.getenv("PDF_SUMMARIZER_DB_URL", "postgresql+psycopg://user:password@localhost/pdf_summarizer"),
        help="Database URL",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser("backfill", help="Backfill embeddings for pages")
    backfill.add_argument("--batch-size", type=int, default=64)
    backfill.add_argument("--max-batches", type=int, default=None)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question", help="Question")
    ask.add_argument("--top-k", type=int, default=10)
    ask.add_argument("--filename", action="append", help="Filter by filename")
    ask.add_argument("--doc-id", type=int, action="append", help="Filter by doc id")
    ask.add_argument("--page-min", type=int, default=None)
    ask.add_argument("--page-max", type=int, default=None)

    args = parser.parse_args()
    pipeline = GeminiRAGPipeline(database_url=args.db_url)

    if args.command == "backfill":
        n = pipeline.backfill_embeddings(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        print(f"Embedded {n} chunk(s).")
    elif args.command == "ask":
        filters = RetrievalFilters(
            document_ids=args.doc_id,
            filenames=args.filename,
            page_min=args.page_min,
            page_max=args.page_max,
        )
        result = pipeline.answer_question(args.question, top_k=args.top_k, filters=filters)
        print("\n=== Answer ===\n")
        print(result["answer"])
        print("\n=== Chunks used ===")
        for m in result["chunks_used"]:
            print(m)


if __name__ == "__main__":
    main()
