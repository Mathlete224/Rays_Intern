"""
RAG pipeline: search on verbalized_summary, answer from raw_content.

Flow: Return top 3 most relevant chunks; for each, include metadata + summary of the chunk,
its parent, and its siblings, then repeat for the other two (~9 chunks in context).
"""
import json
import os
import time
import uuid as uuid_lib
from dataclasses import dataclass
from datetime import date as date_cls
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import google.api_core.exceptions
from google import genai
from google.genai import types

from database import DatabaseManager, PDFChunk, PDFDocument

# Use model that outputs 3072 dims (schema expects Vector(3072))
# models/embedding-001 or models/text-embedding-005; gemini-embedding-001 defaults to 3072
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-2.5-flash"

# Conversation history window: last 14 messages = 7 full Q&A pairs.
# Enough for extended analysis sessions without bloating context.
HISTORY_WINDOW = 14

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
    # Extended metadata filters
    tickers: Optional[Sequence[str]] = None
    report_type: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    coverage_period_from: Optional[str] = None
    coverage_period_to: Optional[str] = None


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

    # For conversational follow-ups (e.g. "expand on that"), skip the similarity
    # threshold entirely. The vector search still runs and returns the top-k closest
    # chunks — we just don't discard any of them. The model has the conversation
    # history to understand it should elaborate, not introduce new topics.
    FOLLOWUP_SIMILARITY_THRESHOLD = 0.0

    def _analyze_query(
        self,
        question: str,
        history: Optional[List[dict]] = None,
    ) -> dict:
        """Single Gemini Flash call that simultaneously:
          1. Extracts hard filters (company, date range, pages) from the question + history.
          2. Rewrites the question into a self-contained standalone_query for vector search
             (strips filter noise, resolves history references like "their" / "that").
          3. Classifies whether this is a conversational follow-up (expand, clarify, continue)
             vs a genuinely new information search.

        Returns:
            {
                "hard_filters": {
                    "sender_companies": [...] | null,
                    "written_date_from": "YYYY-MM-DD" | null,
                    "written_date_to":   "YYYY-MM-DD" | null,
                    "page_min": int | null,
                    "page_max": int | null,
                },
                "standalone_query": str,  # used for embedding / vector search
                "is_followup": bool       # true → relax similarity threshold
            }

        Falls back to {"hard_filters": {}, "standalone_query": question, "is_followup": false}
        on any error so the pipeline degrades gracefully.
        """
        # Fetch known companies and filenames to ground the model's extraction.
        session = self.db.get_session()
        try:
            known_companies = [
                r[0] for r in session.query(PDFDocument.sender_company).distinct().all()
                if r[0]
            ]
            known_filenames = [
                r[0] for r in session.query(PDFDocument.filename).distinct().all()
                if r[0]
            ]
        finally:
            session.close()

        # Format history as readable lines (oldest first).
        history_text = "(none)"
        if history:
            lines = []
            for msg in history[-HISTORY_WINDOW:]:
                label = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{label}: {msg['content']}")
            history_text = "\n".join(lines)

        today = date_cls.today().isoformat()

        prompt = f"""You are a query analysis assistant for a financial document RAG system.
Today's date: {today}
Known companies in the database: {json.dumps(known_companies)}
Known filenames: {json.dumps(known_filenames)}

Given the conversation history and current question, return a JSON object with exactly three fields:

1. "hard_filters": deterministic constraints to narrow the document search.
   Extract only what is explicitly stated or clearly implied:
   - "sender_companies": list of company/firm names matching a known company above, or null
   - "written_date_from": ISO date (YYYY-MM-DD) for when the document was PUBLISHED, or null
   - "written_date_to":   ISO date (YYYY-MM-DD) for when the document was PUBLISHED, or null
   - "page_min": integer page lower bound if explicitly mentioned, or null
   - "page_max": integer page upper bound if explicitly mentioned, or null
   - "tickers": list of ticker symbols mentioned (e.g. ["BTC", "AAPL"]), or null.
     Use standard exchange symbols, not full names.
   - "report_type": type of analysis if specified — one of: equity_research,
     technical_analysis, macro, crypto, sector_note, strategy, other — or null
   - "sector": GICS sector if mentioned (e.g. "Technology", "Energy"), or null
   - "asset_class": asset class if specified — one of: equity, crypto, fixed_income,
     commodity, fx, mixed — or null
   - "coverage_period_from": ISO date for the START of the period being ANALYSED
     (not when published). Use for queries about earnings, quarterly performance, etc.
     e.g. "Q3 2024 performance" → 2024-07-01. Or null.
   - "coverage_period_to": ISO date for the END of the period being ANALYSED. Or null.

   Quarter mapping: Q1=Jan 1–Mar 31, Q2=Apr 1–Jun 30, Q3=Jul 1–Sep 30, Q4=Oct 1–Dec 31.
   If a quarter is mentioned without a year, infer from context or use today's year.
   Distinguish written_date (when published) from coverage_period (what period analysed):
   a Q3 earnings report published in November uses written_date=Nov, coverage_period=Jul–Sep.

2. "standalone_query": rewrite the question as a self-contained semantic search string.
   - Resolve all pronouns / references using history (e.g. "their" → company name)
   - Remove information already captured in hard_filters (ticker, date, company, report type)
   - Keep only the semantic/reasoning intent (what the user actually wants to know)
   - If the question is already self-contained, keep it as-is

3. "is_followup": true if the question asks the assistant to elaborate, clarify, continue,
   or reason further about something already discussed (e.g. "can you expand on that",
   "tell me more", "what do you mean by X", "go deeper on the second point", "why is that").
   Set to false if the question requests genuinely new information from the documents,
   even if it references prior context (e.g. "what about their bond holdings?").

Conversation history (oldest first):
{history_text}

Current question: {question}

Return ONLY a valid JSON object. No markdown, no explanation."""

        try:
            response = self.client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            text = (response.text or "").strip()
            # Strip markdown code fences if present despite response_mime_type
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            result = json.loads(text)
            if "hard_filters" not in result or "standalone_query" not in result:
                raise ValueError("Missing required keys in analysis response")
            # Ensure is_followup is always a bool
            result.setdefault("is_followup", False)
            result["is_followup"] = bool(result["is_followup"])
            return result
        except Exception as exc:
            print(f"[WARNING] _analyze_query failed ({exc}); falling back to original question")
            return {"hard_filters": {}, "standalone_query": question, "is_followup": False}

    @staticmethod
    def _parse_date(value) -> Optional[date_cls]:
        """Convert an ISO date string from the LLM ('YYYY-MM-DD') to a datetime.date.
        Returns None if the value is missing or unparseable.
        """
        if value is None:
            return None
        if isinstance(value, date_cls):
            return value
        try:
            return date_cls.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    def _merge_filters(
        self, explicit: RetrievalFilters, analysis: dict
    ) -> RetrievalFilters:
        """Merge LLM-inferred hard filters with explicit caller-supplied filters.
        Explicit (sidebar) values always win — inferred values only fill in None fields.
        Date strings from the LLM are converted to datetime.date so SQLAlchemy
        can compare them against the DATE column without a type error.
        """
        hf = analysis.get("hard_filters") or {}
        return RetrievalFilters(
            document_ids=explicit.document_ids,
            filenames=explicit.filenames,
            page_min=explicit.page_min if explicit.page_min is not None else hf.get("page_min"),
            page_max=explicit.page_max if explicit.page_max is not None else hf.get("page_max"),
            sender_names=explicit.sender_names,
            sender_companies=explicit.sender_companies or hf.get("sender_companies") or None,
            written_date_from=explicit.written_date_from or self._parse_date(hf.get("written_date_from")),
            written_date_to=explicit.written_date_to or self._parse_date(hf.get("written_date_to")),
            tickers=explicit.tickers or hf.get("tickers") or None,
            report_type=explicit.report_type or hf.get("report_type") or None,
            sector=explicit.sector or hf.get("sector") or None,
            asset_class=explicit.asset_class or hf.get("asset_class") or None,
            coverage_period_from=explicit.coverage_period_from or self._parse_date(hf.get("coverage_period_from")),
            coverage_period_to=explicit.coverage_period_to or self._parse_date(hf.get("coverage_period_to")),
        )

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[RetrievalFilters] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[PDFChunk]:
        """Vector search over verbalized_summary embeddings.

        Expects `query` to already be the standalone/cleaned query from _analyze_query
        and `filters` to already be the merged result of _merge_filters.
        Returns the top_k most relevant chunks that exceed the threshold,
        or an empty list if nothing is relevant enough.

        Args:
            similarity_threshold: Override the default SIMILARITY_THRESHOLD.
                Pass FOLLOWUP_SIMILARITY_THRESHOLD for conversational follow-ups.
        """
        if filters is None:
            filters = RetrievalFilters()
        if similarity_threshold is None:
            similarity_threshold = self.SIMILARITY_THRESHOLD

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
            tickers=filters.tickers,
            report_type=filters.report_type,
            sector=filters.sector,
            asset_class=filters.asset_class,
            coverage_period_from=filters.coverage_period_from,
            coverage_period_to=filters.coverage_period_to,
            similarity_threshold=similarity_threshold,
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
        history: Optional[List[dict]] = None,
    ) -> dict:
        """Answer a question using RAG with optional conversation history.

        Pipeline:
          1. _analyze_query  → hard_filters + standalone_query (one Gemini Flash call)
          2. _merge_filters  → combine inferred filters with explicit caller filters
          3. retrieve_relevant_chunks(standalone_query, merged_filters)
          4. _build_context  → format retrieved chunks
          5. Multi-turn Gemini generation with history interlaced as user/model turns

        Args:
            question: The user's current question (original, not rewritten).
            top_k: Number of chunks to retrieve.
            filters: Explicit filters set by the user (always override inferred).
            history: Recent chat messages [{"role": "user"|"assistant", "content": str}, ...].
                     The last HISTORY_WINDOW entries are used.
        """
        if filters is None:
            filters = RetrievalFilters()

        # Step 1 & 2: Analyze query, extract hard filters + standalone search query.
        analysis = self._analyze_query(question, history=history)
        standalone_query = analysis.get("standalone_query") or question
        is_followup = analysis.get("is_followup", False)
        trimmed_history = (history or [])[-HISTORY_WINDOW:]

        print(f"[DEBUG] standalone_query: {standalone_query!r}")
        print(f"[DEBUG] is_followup: {is_followup}")
        print(f"[DEBUG] inferred hard_filters: {analysis.get('hard_filters')}")

        citation_instruction = (
            "Answer clearly. For every specific fact cite (filename, page). "
            "For synthesized conclusions, note which documents they are drawn from."
        )

        # ── Follow-up path: skip RAG entirely ────────────────────────────────
        # For "expand on that", "clarify X", "go deeper" etc. the model already
        # has the full answer in history. Sending new chunks is noise and risks
        # pulling the model away from what it was discussing.
        # Fall back to normal RAG if there is no history to elaborate from.
        if is_followup and trimmed_history:
            followup_system = (
                "You are a financial analysis assistant continuing a conversation. "
                "The user is asking you to elaborate, clarify, or go deeper on your "
                "previous answer. Stay strictly grounded in what has already been "
                "discussed and cited in this conversation — do not introduce new facts, "
                "figures, or claims that were not present in your prior answers. "
                "If the user asks about something not covered in the conversation, "
                "say so explicitly rather than speculating."
            )
            contents: List = []
            first_question = trimmed_history[0]["content"]
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"{followup_system}\n\nQuestion: {first_question}")],
            ))
            for msg in trimmed_history[1:]:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                ))
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"Question: {question}\n\n{citation_instruction}")],
            ))

            for attempt in range(4):
                try:
                    response = self.client.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=contents,
                        config={"temperature": 0},
                    )
                    break
                except google.api_core.exceptions.ResourceExhausted:
                    if attempt == 3:
                        raise
                    wait = 15 * (2 ** attempt)
                    print(f"[WARNING] Gemini rate limited, retrying in {wait}s (attempt {attempt + 1}/4)")
                    time.sleep(wait)

            answer = response.text if hasattr(response, "text") else str(response)
            return {
                "answer": answer,
                "chunks_used": [],
                "inferred_filters": {},
            }

        # ── Normal RAG path ───────────────────────────────────────────────────
        merged_filters = self._merge_filters(filters, analysis)

        # Step 3: Retrieve chunks using the focused standalone query.
        chunks = self.retrieve_relevant_chunks(
            standalone_query, top_k=top_k, filters=merged_filters,
        )

        if not chunks:
            return {
                "answer": (
                    "No sufficiently relevant content was found in the uploaded documents "
                    "to answer this question. Try rephrasing, or check that the relevant "
                    "document has been uploaded and its embeddings backfilled."
                ),
                "chunks_used": [],
                "inferred_filters": analysis.get("hard_filters") or {},
            }

        # Step 4: Build context string from retrieved chunks.
        context = self._build_context(chunks)

        system = (
            "You are a financial analysis assistant. Your answers must be grounded exclusively "
            "in the provided document context below — never in your training knowledge.\n\n"
            "Rules:\n"
            "1. Every specific claim (numbers, dates, company actions, prices, forecasts) "
            "must be directly supported by the context. Cite the source filename and page "
            "for each such claim.\n"
            "2. You MAY reason, synthesize, and identify trends across the provided chunks — "
            "but only from what the documents actually say. Drawing a conclusion like "
            "'revenue has trended upward across these reports' is allowed if the chunks support it.\n"
            "3. You may NOT use general industry knowledge, assumptions, or facts from your "
            "training data to fill gaps. If the context does not contain enough information "
            "to support a claim, say so explicitly: 'The provided documents do not state...'\n"
            "4. If the question cannot be answered at all from the context, say: "
            "'The uploaded documents do not contain sufficient information to answer this question.'\n"
            "5. Do not invent or guess document names, page numbers, or figures.\n\n"
            "Each chunk has a 'source:' line with the filename and page — use these for citations."
        )
        system_and_context = f"{system}\n\nContext:\n{context}"

        # Step 5: Build contents for Gemini.
        # With history: interleave prior turns so the model has conversation context.
        # Without history: single-turn (identical to original behaviour).
        if trimmed_history:
            contents = []
            # Inject system prompt + retrieved context into the very first user turn
            # so all subsequent turns are grounded in the same document context.
            first_question = trimmed_history[0]["content"]
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"{system_and_context}\n\nQuestion: {first_question}")],
            ))
            for msg in trimmed_history[1:]:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                ))
            # Final turn: the current question (original, not standalone_query)
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"Question: {question}\n\n{citation_instruction}")],
            ))
        else:
            # Stateless single-turn path
            contents = (
                f"{system_and_context}\n\n"
                f"Question: {question}\n\n"
                f"{citation_instruction}"
            )

        for attempt in range(4):
            try:
                response = self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=contents,
                    config={"temperature": 0},
                )
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
            "inferred_filters": analysis.get("hard_filters") or {},
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
