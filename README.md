# PDF Summarizer Pipeline

Docling + Gemini pipeline for image-heavy financial PDFs. One verbalized row per page for RAG.

## Workflow

1. **Parse** – Docling extracts page layout and page images
2. **Verbalize** – Gemini 2.0 Flash describes every chart/graph/table on each page
3. **Store** – One row per page in Postgres: Text + Chart Summary
4. **Query** – pgvector semantic search over verbalized pages

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key"
```

**Note:** Postgres + pgvector is required (SQLite is not supported for vector search).

```sql
CREATE DATABASE pdf_summarizer;
\c pdf_summarizer
CREATE EXTENSION IF NOT EXISTS vector;
```

## Usage

### 1. Process PDFs (Parse + Verbalize + Store)

```bash
python pipeline.py research_pdfs/ --db-url "postgresql+psycopg://user:pass@localhost/pdf_summarizer"
```

### 2. Backfill embeddings

```bash
python rag_gemini.py backfill --db-url "postgresql+psycopg://user:pass@localhost/pdf_summarizer"
```

### 3. Ask questions (RAG)

```bash
python rag_gemini.py ask "What were the main revenue drivers?" --db-url "postgresql+psycopg://user:pass@localhost/pdf_summarizer"
```

With filters:

```bash
python rag_gemini.py ask "Summarize risk factors" --filename report.pdf --page-min 5 --page-max 20
```

## Schema (Golden Schema)

- **pdf_documents** – File-level metadata
- **pdf_chunks** – One row per page/section:
  - `id` – UUID primary key
  - `embedding` – Vector(3072), from verbalized_summary (search against this)
  - `raw_content` – Original Docling markdown (use for answering)
  - `verbalized_summary` – Gemini chart description (embed & search)
  - `metadata` – JSONB: page_number, company_ticker, report_type, file_path
  - `image_blob` – Optional BYTEA for high-res chart crop

## Environment

- `GEMINI_API_KEY` – Required for verbalization and RAG
- `PDF_SUMMARIZER_DB_URL` – Optional default DB URL
