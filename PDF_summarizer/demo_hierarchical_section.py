"""
Small demo script to showcase hierarchical section summarization.

For a single PDF, it:
  1) Runs the main PDFSummarizerPipeline (Docling + Gemini).
  2) Picks one section-level chunk.
  3) Prints:
       a) the exact text that would be embedded,
       b) the metadata stored for that chunk,
       c) the section summary (verbalized_summary).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from database import DatabaseManager, PDFChunk, PDFDocument
from pipeline import PDFSummarizerPipeline
from rag_gemini import embed_text


def main() -> None:
    load_dotenv()

    db_url = os.getenv(
        "PDF_SUMMARIZER_DB_URL",
        "postgresql+psycopg://user:password@localhost/pdf_summarizer",
    )

    pdf_path = os.getenv("PDF_SUMMARIZER_DEMO_PDF")
    if not pdf_path:
        raise RuntimeError(
            "Set PDF_SUMMARIZER_DEMO_PDF to point to a sample PDF file for this demo."
        )

    pdf_path = str(Path(pdf_path).expanduser())

    print(f"Using database: {db_url}")
    print(f"Processing PDF: {pdf_path}")

    pipeline = PDFSummarizerPipeline(database_url=db_url)
    result = pipeline.process_single_pdf(pdf_path, skip_existing=True)
    if result.get("status") not in {"success", "skipped"}:
        raise RuntimeError(f"Pipeline failed: {result}")

    doc_id = result["document_id"]
    db = DatabaseManager(database_url=db_url)

    # Fetch all chunks for this document and identify one section-level chunk.
    chunks = db.get_chunks_by_document(doc_id)
    section_chunk: PDFChunk | None = None
    for c in chunks:
        meta = c.metadata_ or {}
        if meta.get("level") == "section":
            section_chunk = c
            break

    if section_chunk is None:
        raise RuntimeError("No section-level chunk found for this document.")

    text_for_embedding = (
        (section_chunk.raw_content or "") + "\n\n" + (section_chunk.verbalized_summary or "")
    ).strip()

    print("\n=== DEMO: Hierarchical Section Chunk ===\n")

    # a) Show what is embedded (raw text)
    print("A) Text that will be embedded (first 1000 chars):\n")
    print(text_for_embedding[:1000])
    if len(text_for_embedding) > 1000:
        print("\n...[truncated]...\n")

    # Actually compute the embedding once, to demonstrate shape/length.
    emb = embed_text(text_for_embedding)
    print(f"Embedding length: {len(emb)}\n")

    # b) Show metadata
    print("B) Chunk metadata:\n")
    print(section_chunk.metadata_)

    # c) Show the summary text (Gemini section summary)
    print("\nC) Section summary (verbalized_summary):\n")
    print((section_chunk.verbalized_summary or "")[:2000])


if __name__ == "__main__":
    main()

