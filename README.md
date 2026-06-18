# Financial PDF Research Platform

A full-stack research tool for hedge funds. Upload financial PDFs, run semantic search and RAG queries, and explore findings on a visual node-based canvas.

## Architecture

```
PDF_summarizer/   Python RAG pipeline (Docling + Gemini + PostgreSQL/pgvector)
backend/          FastAPI REST API wrapping the pipeline
frontend/         React + Vite visual research canvas (ReactFlow)
```

## Prerequisites

- Python 3.9+
- Node.js 18+
- PostgreSQL 13+ with pgvector extension
- Gemini API key

### PostgreSQL setup (one-time)

```sql
CREATE DATABASE finance_rag;
\c finance_rag
CREATE EXTENSION IF NOT EXISTS vector;
```

### Environment variables

Create a `.env` file in the repo root:

```
PDF_SUMMARIZER_DB_URL=postgresql+psycopg://user@localhost:5432/finance_rag
GEMINI_API_KEY=your_key_here
```

## Setup

```bash
# Install Python dependencies (from repo root)
pip install -r PDF_summarizer/requirements.txt
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

## Running

Open **two terminals**:

```bash
# Terminal 1 — backend (port 8000)
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend (port 5173)
cd frontend
npm run dev
```

Open **http://localhost:5173**

The backend API docs are available at **http://localhost:8000/docs**

## Using the Canvas

1. **Upload PDFs** — click "+ Upload" in the left sidebar
2. **Filter documents** — expand the Filters panel to narrow by company, author, date, ticker, report type, asset class, or sector
3. **Add nodes** — right-click anywhere on the canvas:
   - **Query Node** — type a question, optionally filter by document, click Ask → generates a linked Answer Node
   - **Note** — free-text sticky note for annotations
   - **Research Agent** — enter a high-level goal (e.g. "Analyze company X's financial health") → automatically decomposes into 3–5 sub-queries, runs each against the RAG pipeline, and synthesizes a final report
4. **Connect nodes** — drag from any node's right handle to another node's left handle
5. **Save canvases** — canvases auto-save every 2 seconds; use the toolbar to create named canvases or load previous ones

## CLI Usage (without the web UI)

```bash
# Ingest a single PDF
python PDF_summarizer/pipeline.py report.pdf --db-url $PDF_SUMMARIZER_DB_URL

# Ingest a directory of PDFs (parallelized, default 4 workers)
python PDF_summarizer/pipeline.py research_pdfs/ --db-url $PDF_SUMMARIZER_DB_URL --max-workers 8

# Backfill embeddings
python PDF_summarizer/rag_gemini.py backfill --db-url $PDF_SUMMARIZER_DB_URL

# Backfill extended metadata (tickers, sector, etc.) for existing documents — no re-ingestion needed
python PDF_summarizer/pipeline.py --backfill-metadata --db-url $PDF_SUMMARIZER_DB_URL

# Re-extract metadata for all documents (use after improving the extraction prompt)
python PDF_summarizer/pipeline.py --backfill-metadata --force --db-url $PDF_SUMMARIZER_DB_URL

# Ask a question
python PDF_summarizer/rag_gemini.py ask "What were the main revenue drivers?" --db-url $PDF_SUMMARIZER_DB_URL
```

## How it works

### PDF ingestion (3-level hierarchy)
1. **Docling** parses the PDF — extracts text, tables, and page images
2. **Gemini 2.0 Flash** verbalizes every chart/table into plain text; runs in parallel across sections and images for 3–5× speedup
3. Three chunk levels are stored: document summary → section summaries → per-page content
4. Parent/sibling relationships are stored in JSONB metadata for context expansion
5. **Extended metadata** is extracted in a single Gemini call: tickers, report type, sector, asset class, and coverage period

### RAG query
1. Question is embedded via `text-embedding-003-small` (3072 dims)
2. pgvector cosine search retrieves top-k chunks, filtered by any active hard filters (company, date range, sector, etc.)
3. Each chunk is expanded with its parent + prev/next sibling (~9 chunks total context)
4. **Gemini 1.5 Pro** generates the final answer with citations

### Agentic research
1. **Gemini 2.0 Flash** decomposes the goal into 3–5 sub-questions
2. Each sub-question runs through the full RAG pipeline independently
3. **Gemini 2.0 Flash** synthesizes all findings into a final report

## Document metadata

Each ingested PDF stores the following metadata, used for filtering:

| Field | Description |
|-------|-------------|
| `sender_name` | Author of the report |
| `sender_company` | Company that published the report |
| `sent_date` | Date the report was published |
| `tickers` | List of stock/asset tickers covered (e.g. `["AAPL", "9958 TT"]`) |
| `report_type` | e.g. `equity_research`, `technical_analysis`, `macro_outlook` |
| `sector` | GICS sector classification (e.g. `Energy`, `Technology`) |
| `asset_class` | e.g. `equity`, `crypto`, `fixed_income` |
| `coverage_period_from/to` | Date range the report *analyzes* (distinct from publish date) |

Missing metadata can be backfilled at any time without re-ingesting PDFs — see CLI usage above.

## Database schema

| Table | Description |
|-------|-------------|
| `pdf_documents` | File-level metadata (filename, hash, page count, all metadata fields above) |
| `pdf_chunks` | Chunks with `Vector(1024)` embedding, raw markdown, verbalized summary, JSONB metadata |
| `canvases` | Saved research canvas names and timestamps |
| `canvas_state` | ReactFlow nodes/edges stored as JSONB per canvas |
