"""
RAG pipeline: search on verbalized_summary, answer from raw_content.

Flow: Return top 3 most relevant chunks; for each, include metadata + summary of the chunk,
its parent, and its siblings, then repeat for the other two (~9 chunks in context).
"""
import os
import time
import uuid as uuid_lib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import google.api_core.exceptions
from google import genai
from google.genai import types

from database import DatabaseManager, PDFChunk

# Use model that outputs 3072 dims (schema expects Vector(3072))
# models/embedding-001 or models/text-embedding-005; gemini-embedding-001 defaults to 3072
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-2.5-flash"

load_dotenv()


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


EMBEDDING_DIMS = 768  # must match Vector(768) in database.py


def embed_text(text: str) -> List[float]:
    if not text.strip():
        return []
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMS),
    )
    return list(result.embeddings[0].values)


@dataclass
class RetrievalFilters:
    document_ids: Optional[Sequence[int]] = None
    filenames: Optional[Sequence[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    sender_names: Optional[Sequence[str]] = None
    sender_companies: Optional[Sequence[str]] = None
    written_date_from: Optional[str] = None
    written_date_to: Optional[str] = None


class GeminiRAGPipeline:
    """RAG over verbalized pages (text + chart descriptions)."""

    def __init__(self, database_url: str = None, db: DatabaseManager = None):
        self.client = _get_client()
        if db is not None:
            self.db = db
        elif database_url is not None:
            self.db = DatabaseManager(database_url)
        else:
            raise ValueError("Either database_url or db must be provided")

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

    # Minimum cosine similarity for a chunk to be considered relevant.
    # Cosine similarity ranges 0–1; chunks below this are discarded rather than
    # passed as context, preventing the model from hallucinating from unrelated content.
    SIMILARITY_THRESHOLD = 0.40

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[RetrievalFilters] = None,
    ) -> List[PDFChunk]:
        """
        Vector search over verbalized_summary embeddings.

        Returns the top_k most relevant chunks (default 3) that exceed
        SIMILARITY_THRESHOLD. Returns an empty list if nothing is relevant enough.
        """
        if filters is None:
            filters = RetrievalFilters()

        query_emb = embed_text(query)
        initial_limit = max(top_k * 3, top_k + 3)
        raw_chunks = self.db.semantic_search_chunks(
            query_embedding=query_emb,
            limit=initial_limit,
            document_ids=filters.document_ids,
            filenames=filters.filenames,
            page_min=filters.page_min,
            page_max=filters.page_max,
            sender_names=filters.sender_names,
            sender_companies=filters.sender_companies,
            written_date_from=filters.written_date_from,
            written_date_to=filters.written_date_to,
            similarity_threshold=self.SIMILARITY_THRESHOLD,
        )

        if not raw_chunks:
            return []

        return self._diversify_chunks(raw_chunks, top_k=top_k)

    def _get_chunk_family(
        self, chunk: PDFChunk
    ) -> Tuple[Optional[PDFChunk], Optional[PDFChunk], Optional[PDFChunk]]:
        """Return (parent, prev_sibling, next_sibling) for a chunk using metadata IDs."""
        meta = chunk.metadata_ or {}
        parent, prev_sib, next_sib = None, None, None
        try:
            pid = meta.get("parent_chunk_id")
            if pid:
                parent = self.db.get_chunk_by_id(uuid_lib.UUID(pid))
        except (ValueError, TypeError):
            pass
        try:
            pid = meta.get("prev_sibling_chunk_id")
            if pid:
                prev_sib = self.db.get_chunk_by_id(uuid_lib.UUID(pid))
        except (ValueError, TypeError):
            pass
        try:
            pid = meta.get("next_sibling_chunk_id")
            if pid:
                next_sib = self.db.get_chunk_by_id(uuid_lib.UUID(pid))
        except (ValueError, TypeError):
            pass
        return parent, prev_sib, next_sib

    def _build_context(self, top_chunks: List[PDFChunk]) -> str:
        """
        Build context from top 3 chunks: for each chunk include its metadata + summary,
        then the same for its parent and sibling chunks (~9 chunks total).
        """
        parts: List[str] = []
        for n, chunk in enumerate(top_chunks, 1):
            parent, prev_sib, next_sib = self._get_chunk_family(chunk)
            block = self._format_chunk_block(
                chunk, parent, prev_sib, next_sib, label=f"Retrieved chunk {n}"
            )
            parts.append(block)
        return "\n\n".join(parts)

    def _format_chunk_block(
        self,
        chunk: PDFChunk,
        parent: Optional[PDFChunk],
        prev_sibling: Optional[PDFChunk],
        next_sibling: Optional[PDFChunk],
        label: str = "Chunk",
    ) -> str:
        """Format one retrieved chunk plus its parent and siblings (metadata + summary + content)."""
        lines: List[str] = [f"=== {label} ==="]

        def append_chunk(c: PDFChunk, role: str) -> None:
            meta = c.metadata_ or {}
            summary = (c.verbalized_summary or "").strip()
            content = (c.raw_content or "").strip()
            filename = c.document.filename if c.document else f"document_id={c.document_id}"
            page = f" page {c.page_number}" if c.page_number is not None else ""
            lines.append(f"  {role} source: {filename}{page}")
            lines.append(f"  {role} metadata: {meta}")
            lines.append(f"  {role} summary: {summary[:1500]}{'...' if len(summary) > 1500 else ''}")
            lines.append(f"  {role} content: {content[:4000]}{'...' if len(content) > 4000 else ''}")

        append_chunk(chunk, "Chunk")
        if parent:
            append_chunk(parent, "Parent")
        else:
            lines.append("  Parent: (none)")
        if prev_sibling:
            append_chunk(prev_sibling, "Previous sibling")
        else:
            lines.append("  Previous sibling: (none)")
        if next_sibling:
            append_chunk(next_sibling, "Next sibling")
        else:
            lines.append("  Next sibling: (none)")

        return "\n".join(lines)

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        filters: Optional[RetrievalFilters] = None,
    ) -> dict:
        """Return top_k chunks (default 3), expand each with parent + siblings, feed ~9 chunks to context."""
        chunks = self.retrieve_relevant_chunks(question, top_k=top_k, filters=filters)

        if not chunks:
            return {
                "answer": (
                    "No sufficiently relevant content was found in the uploaded documents "
                    "to answer this question. Try rephrasing, or check that the relevant "
                    "document has been uploaded and its embeddings backfilled."
                ),
                "chunks_used": [],
            }

        context = self._build_context(chunks)

        system = (
            "You are a financial analysis assistant. Use ONLY the provided context "
            "(original document content from reports) to answer. If the answer is not "
            "derivable from the context, say you don't know.\n"
            "Each chunk in the context has a 'source:' line indicating the filename and page. "
            "When citing information, reference ONLY those exact filenames and page numbers — "
            "do not invent or guess document names."
        )
        prompt = (
            f"{system}\n\nContext:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer clearly, citing the filename and page from the 'source:' field when relevant."
        )

        for attempt in range(4):
            try:
                response = self.client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
                break
            except google.api_core.exceptions.ResourceExhausted:
                if attempt == 3:
                    raise
                wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
                print(f"[WARNING] Gemini rate limited, retrying in {wait}s (attempt {attempt + 1}/4)")
                time.sleep(wait)
        answer = response.text if hasattr(response, "text") else str(response)

        return {
            "answer": answer,
            "chunks_used": [
                {
                    "chunk_id": str(c.id),
                    "document_id": c.document_id,
                    "page_number": c.page_number,
                    "metadata": c.metadata_ or {},
                }
                for c in chunks
            ],
        }

    def _diversify_chunks(self, chunks: List[PDFChunk], top_k: int) -> List[PDFChunk]:
        """
        Promote diversity across sections and hierarchy levels.

        Strategy:
            - Prefer document-level and section-level chunks.
            - Spread page-level chunks across different sections.
        """
        # Keep original ranking index as tie-breaker
        ranked = list(enumerate(chunks))

        doc_level: List[Tuple[int, PDFChunk]] = []
        section_level: Dict[str, List[Tuple[int, PDFChunk]]] = {}
        page_level: Dict[Tuple[int, Optional[str]], List[Tuple[int, PDFChunk]]] = {}

        for idx, c in ranked:
            meta = c.metadata_ or {}
            level = meta.get("level")
            section_id = meta.get("section_id")

            if level == "document":
                doc_level.append((idx, c))
            elif level == "section":
                section_level.setdefault(section_id or f"sec-{idx}", []).append((idx, c))
            else:
                key = (c.document_id, section_id)
                page_level.setdefault(key, []).append((idx, c))

        selected: List[PDFChunk] = []

        # 1) At most one document-level chunk per document.
        for _, c in sorted(doc_level, key=lambda t: t[0]):
            if len(selected) >= top_k:
                break
            if c not in selected:
                selected.append(c)

        if len(selected) >= top_k:
            return selected[:top_k]

        # 2) One section-level chunk per section in order.
        for sec_id, items in sorted(section_level.items(), key=lambda kv: kv[0] or ""):
            if len(selected) >= top_k:
                break
            items_sorted = sorted(items, key=lambda t: t[0])
            _, c = items_sorted[0]
            if c not in selected:
                selected.append(c)

        if len(selected) >= top_k:
            return selected[:top_k]

        # 3) Round-robin across sections for page-level chunks.
        # Convert dict values to queues.
        queues: List[List[Tuple[int, PDFChunk]]] = [
            sorted(v, key=lambda t: t[0]) for _, v in sorted(page_level.items(), key=lambda kv: kv[0])
        ]

        exhausted = False
        while len(selected) < top_k and not exhausted:
            exhausted = True
            for q in queues:
                if not q:
                    continue
                exhausted = False
                _, c = q.pop(0)
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= top_k:
                        break

        return selected[:top_k]


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
    ask.add_argument("--top-k", type=int, default=3)
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
