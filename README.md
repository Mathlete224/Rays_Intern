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
2. **Add nodes** — right-click anywhere on the canvas:
   - **Query Node** — type a question, optionally filter by document, click Ask → generates a linked Answer Node
   - **Note** — free-text sticky note for annotations
   - **Research Agent** — enter a high-level goal (e.g. "Analyze company X's financial health") → automatically decomposes into 3–5 sub-queries, runs each against the RAG pipeline, and synthesizes a final report
3. **Connect nodes** — drag from any node's right handle to another node's left handle
4. **Save canvases** — canvases auto-save every 2 seconds; use the toolbar to create named canvases or load previous ones

## CLI Usage (without the web UI)

```bash
# Ingest PDFs
python PDF_summarizer/pipeline.py test_pdfs/long_example.pdf --db-url $PDF_SUMMARIZER_DB_URL

# Backfill embeddings
python PDF_summarizer/rag_gemini.py backfill --db-url $PDF_SUMMARIZER_DB_URL

# Ask a question
python PDF_summarizer/rag_gemini.py ask "What were the main revenue drivers?" --db-url $PDF_SUMMARIZER_DB_URL
```

## How it works

### PDF ingestion (3-level hierarchy)
1. **Docling** parses the PDF — extracts text, tables, and page images
2. **Gemini 2.0 Flash** verbalizes every chart/table into plain text
3. Three chunk levels are stored: document summary → section summaries → per-page content
4. Parent/sibling relationships are stored in JSONB metadata for context expansion

### RAG query
1. Question is embedded via `text-embedding-003-small` (3072 dims)
2. pgvector cosine search retrieves top-k chunks
3. Each chunk is expanded with its parent + prev/next sibling (~9 chunks total context)
4. **Gemini 1.5 Pro** generates the final answer

### Agentic research
1. **Gemini 2.0 Flash** decomposes the goal into 3–5 sub-questions
2. Each sub-question runs through the full RAG pipeline independently
3. **Gemini 2.0 Flash** synthesizes all findings into a final report

## Database schema

| Table | Description |
|-------|-------------|
| `pdf_documents` | File-level metadata (filename, hash, page count) |
| `pdf_chunks` | Chunks with `Vector(1024)` embedding, raw markdown, verbalized summary, JSONB metadata |
| `canvases` | Saved research canvas names and timestamps |
| `canvas_state` | ReactFlow nodes/edges stored as JSONB per canvas |
