"""
Main pipeline: Docling parse → Gemini verbalize → Store in Postgres.

Process PDFs from research_pdfs (or any directory) and save one verbalized row per page.
Chunk metadata includes parent_chunk_id and prev/next_sibling_chunk_id (UUIDs as strings)
for easy DB lookup in the RAG pipeline.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
import uuid as uuid_lib

from database import DatabaseManager
from pdf_processor import DoclingProcessor
from rag_gemini import GeminiRAGPipeline
from utils import get_file_hash

class PDFSummarizerPipeline:
    """Parse PDFs with Docling, verbalize charts with Gemini, store in Postgres."""

    def __init__(self, database_url: str = "sqlite:///pdf_summarizer.db"):
        self.db_manager = DatabaseManager(database_url)
        self.processor = DoclingProcessor()
        self.rag = GeminiRAGPipeline(db=self.db_manager)

    def process_single_pdf(
        self,
        pdf_path: str,
        skip_existing: bool = True,
    ) -> dict:
        """
        Process a single PDF: Docling parse → Gemini verbalize → store pages.
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        file_hash = get_file_hash(pdf_path)
        if skip_existing:
            existing = self.db_manager.get_document_by_hash(file_hash)
            if existing:
                print(f" Already processed: {filename}")
                return {
                    "status": "skipped",
                    "filename": filename,
                    "document_id": existing.id,
                    "message": "File already in database",
                }

        try:
            print(f"📄 Processing: {filename}")

            chunks, total_pages, file_size_bytes, sender_info = self.processor.process_pdf(str(pdf_path))

            doc = self.db_manager.add_document(
                filename=filename,
                file_path=str(pdf_path.absolute()),
                total_pages=total_pages,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                sender_name=sender_info.get("sender_name"),
                sender_company=sender_info.get("sender_company"),
                sent_date=sender_info.get("sent_date"),
            )

            chunk_ids = self.db_manager.add_chunks(doc.id, chunks)
            self._set_parent_sibling_metadata(chunk_ids, chunks)

            print(f"   Generating embeddings...")
            self.rag.backfill_embeddings()

            print(f"✅ Done: {filename}")
            print(f"   Pages: {total_pages}")
            print(f"   Document ID: {doc.id}")

            return {
                "status": "success",
                "filename": filename,
                "document_id": doc.id,
                "total_pages": total_pages,
                "file_size_bytes": file_size_bytes,
            }

        except Exception as e:
            print(f"❌ Error: {filename}: {e}")
            return {"status": "error", "filename": filename, "error": str(e)}

    def process_directory(
        self,
        directory_path: str,
        skip_existing: bool = True,
    ) -> dict:
        """Process all PDFs in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDFs in {directory_path}")
            return {"status": "no_files", "total_files": 0}

        print(f"📁 Found {len(pdf_files)} PDF(s) in {directory_path}\n")

        results = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "results": [],
        }

        for pdf_file in pdf_files:
            r = self.process_single_pdf(str(pdf_file), skip_existing=skip_existing)
            results["results"].append(r)
            if r["status"] == "success":
                results["successful"] += 1
            elif r["status"] == "error":
                results["failed"] += 1
            elif r["status"] == "skipped":
                results["skipped"] += 1

        print(f"\n📊 Summary: {results['successful']} ok, {results['failed']} failed, {results['skipped']} skipped")
        return results

    def _set_parent_sibling_metadata(
        self, chunk_ids: List[uuid_lib.UUID], chunks: List[dict]
    ) -> None:
        """
        Set parent_chunk_id, prev_sibling_chunk_id, next_sibling_chunk_id in each chunk's
        metadata for easy DB lookup in RAG. chunk_ids and chunks are in the same order.
        Uses only chunk dicts (no detached ORM objects) to avoid Session errors.
        """
        doc_idx: int | None = None
        section_id_to_idx: dict = {}

        for i, c_dict in enumerate(chunks):
            meta = c_dict.get("metadata") or {}
            level = meta.get("level")
            if level == "document":
                doc_idx = i
            elif level == "section":
                sid = meta.get("section_id")
                if sid is not None:
                    section_id_to_idx[sid] = i

        for i, c_dict in enumerate(chunks):
            meta = dict(c_dict.get("metadata") or {})
            level = meta.get("level")
            parent_id: str | None = None
            prev_sib_id: str | None = None
            next_sib_id: str | None = None

            if level == "section" and doc_idx is not None:
                parent_id = str(chunk_ids[doc_idx])
                if i > doc_idx + 1 and (chunks[i - 1].get("metadata") or {}).get("level") == "section":
                    prev_sib_id = str(chunk_ids[i - 1])
                if i < len(chunks) - 1 and (chunks[i + 1].get("metadata") or {}).get("level") == "section":
                    next_sib_id = str(chunk_ids[i + 1])
            elif level == "image":
                sid = meta.get("section_id")
                if sid and sid in section_id_to_idx:
                    parent_id = str(chunk_ids[section_id_to_idx[sid]])

            meta["parent_chunk_id"] = parent_id
            meta["prev_sibling_chunk_id"] = prev_sib_id
            meta["next_sibling_chunk_id"] = next_sib_id
            self.db_manager.update_chunk_metadata(chunk_ids[i], meta)

    def get_document_info(self, document_id: int) -> Optional[dict]:
        """Get info about a processed document."""
        session = self.db_manager.get_session()
        try:
            from database import PDFDocument

            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if not doc:
                return None

            chunks = self.db_manager.get_chunks_by_document(document_id)
            return {
                "id": doc.id,
                "filename": doc.filename,
                "file_path": doc.file_path,
                "total_pages": doc.total_pages,
                "total_chunks_stored": len(chunks),
                "uploaded_at": doc.uploaded_at.isoformat(),
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "chunks": [
                    {
                        "id": str(c.id),
                        "page_number": c.page_number,
                        "raw_preview": (c.raw_content or "")[:200] + "...",
                    }
                    for c in chunks[:10]
                ],
            }
        finally:
            session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF pipeline: Docling + Gemini verbalization")
    parser.add_argument("input", help="PDF file or directory (e.g. research_pdfs/)")
    parser.add_argument(
        "--db-url",
        default=os.getenv("PDF_SUMMARIZER_DB_URL", "postgresql+psycopg://user:password@localhost/pdf_summarizer"),
        help="Database URL (Postgres + pgvector required)",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess existing files")

    args = parser.parse_args()
    pipeline = PDFSummarizerPipeline(database_url=args.db_url)

    path = Path(args.input)
    if path.is_file() and path.suffix.lower() == ".pdf":
        pipeline.process_single_pdf(str(path), skip_existing=not args.no_skip_existing)
    elif path.is_dir():
        pipeline.process_directory(str(path), skip_existing=not args.no_skip_existing)
    else:
        print(f"❌ Invalid input: {args.input} (must be PDF file or directory)")
        sys.exit(1)


if __name__ == "__main__":
    main()
