"""
Database module: Golden Schema for financial RAG.

Store verbalization for searching, raw content for answering.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import joinedload, relationship, sessionmaker
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class PDFDocument(Base):
    """File-level metadata."""

    __tablename__ = "pdf_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False, unique=True)
    file_path = Column(String(1000), nullable=False)
    total_pages = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    sender_name = Column(String(500), nullable=True)
    sender_company = Column(String(500), nullable=True)
    sent_date = Column(Date, nullable=True)
    written_date = Column(Date, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    chunks = relationship(
        "PDFChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class PDFChunk(Base):
    """
    Golden Schema: one row per page/section.

    - embedding: Verbalization vector (search against this)
    - raw_content: Original Docling markdown (use for answering)
    - verbalized_summary: Gemini chart description (search uses this)
    - metadata: page_number, company_ticker, report_type, file_path
    - image_blob: Optional high-res chart crop for final LLM check
    """

    __tablename__ = "pdf_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(Integer, ForeignKey("pdf_documents.id"), nullable=False)

    # Verbalization vector — embed verbalized_summary, search against this
    embedding = Column(Vector(768), nullable=True)

    # Original Docling markdown (text + reconstructed tables) — use for answering
    raw_content = Column(Text, nullable=False)

    # Gemini plain-text description of charts — used for embedding/search
    verbalized_summary = Column(Text, nullable=True)

    # Denormalized for filtering; also in metadata
    page_number = Column(Integer, nullable=True)

    # page_number, company_ticker, report_type, file_path, etc.
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Optional: high-res chart crop for final LLM check
    image_blob = Column(LargeBinary, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("PDFDocument", back_populates="chunks")


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, database_url: str = "sqlite:///pdf_summarizer.db"):
        """
        Args:
            database_url: postgresql+psycopg://user:pass@localhost/pdf_summarizer
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        if database_url.startswith("postgresql"):
            with self.engine.connect() as conn:
                # Schema migrations — always safe
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sender_name VARCHAR(500)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sender_company VARCHAR(500)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sent_date DATE"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS written_date DATE"))
                conn.commit()

            # pgvector index — requires the vector extension; skip gracefully if not installed
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS pdf_chunks_embedding_idx
                        ON pdf_chunks
                        USING hnsw (embedding vector_cosine_ops);
                    """))
                    conn.commit()
            except Exception as e:
                print(f"[WARNING] Could not create pgvector index (is the 'vector' extension installed?): {e}")

    def get_session(self):
        return self.SessionLocal()

    # -------- Document & Chunk CRUD --------

    def add_document(
        self,
        filename: str,
        file_path: str,
        total_pages: int,
        file_size_bytes: int,
        file_hash: str,
        sender_name: Optional[str] = None,
        sender_company: Optional[str] = None,
        sent_date=None,
    ) -> PDFDocument:
        session = self.get_session()
        try:
            doc = PDFDocument(
                filename=filename,
                file_path=file_path,
                total_pages=total_pages,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                sender_name=sender_name,
                sender_company=sender_company,
                sent_date=sent_date,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_document_by_hash(self, file_hash: str) -> Optional[PDFDocument]:
        session = self.get_session()
        try:
            return session.query(PDFDocument).filter_by(file_hash=file_hash).first()
        finally:
            session.close()
    def add_chunks(self, document_id: int, chunks: List[dict]) -> List[uuid.UUID]:
        """
        Add chunks for a document. Returns list of chunk IDs (in same order as chunks).
        Each chunk dict: raw_content, verbalized_summary, metadata, image_blob (optional)
        """
        session = self.get_session()
        try:
            chunk_objects: List[PDFChunk] = []
            for c in chunks:
                obj = PDFChunk(
                    document_id=document_id,
                    raw_content=c["raw_content"],
                    verbalized_summary=c.get("verbalized_summary"),
                    page_number=c.get("metadata", {}).get("page_number"),
                    metadata_=c.get("metadata"),
                    image_blob=c.get("image_blob"),
                )
                chunk_objects.append(obj)
                session.add(obj)

            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if doc:
                doc.processed_at = datetime.utcnow()

            session.commit()
            # Return IDs while session is still open so callers don't touch detached objects
            return [obj.id for obj in chunk_objects]
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_document_by_filename(self, filename: str) -> Optional[PDFDocument]:
        session = self.get_session()
        try:
            return session.query(PDFDocument).filter_by(filename=filename).first()
        finally:
            session.close()

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and all its chunks (cascade). Returns True if deleted, False if not found.
        """
        session = self.get_session()
        try:
            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if not doc:
                return False
            session.delete(doc)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_all_documents(self) -> int:
        """Delete all documents and their chunks (cascade). Returns number of documents deleted."""
        session = self.get_session()
        try:
            docs = session.query(PDFDocument).all()
            n = len(docs)
            for doc in docs:
                session.delete(doc)
            session.commit()
            return n
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_chunks_by_document(self, document_id: int) -> List[PDFChunk]:
        session = self.get_session()
        try:
            return (
                session.query(PDFChunk)
                .filter_by(document_id=document_id)
                .order_by(PDFChunk.created_at)
                .all()
            )
        finally:
            session.close()

    def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Optional[PDFChunk]:
        session = self.get_session()
        try:
            return (
                session.query(PDFChunk)
                .options(joinedload(PDFChunk.document))
                .filter_by(id=chunk_id)
                .first()
            )
        finally:
            session.close()

    def update_chunk_metadata(self, chunk_id: uuid.UUID, metadata: Dict[str, Any]) -> None:
        """Update the metadata JSONB for a chunk (merge with existing)."""
        session = self.get_session()
        try:
            chunk = session.query(PDFChunk).filter_by(id=chunk_id).first()
            if chunk:
                existing = dict(chunk.metadata_ or {})
                existing.update(metadata)
                chunk.metadata_ = existing
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------- Embeddings & Vector Search --------

    def upsert_chunk_embedding(self, chunk_id: uuid.UUID, embedding: Sequence[float]) -> None:
        session = self.get_session()
        try:
            chunk = session.query(PDFChunk).filter_by(id=chunk_id).first()
            if chunk:
                chunk.embedding = list(embedding)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_chunks_without_embedding(self, limit: int = 100) -> List[PDFChunk]:
        session = self.get_session()
        try:
            return (
                session.query(PDFChunk)
                .filter(PDFChunk.embedding.is_(None))
                .order_by(PDFChunk.created_at)
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def semantic_search_chunks(
        self,
        query_embedding: Sequence[float],
        limit: int = 20,
        document_ids: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None,
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        sender_names: Optional[Sequence[str]] = None,
        sender_companies: Optional[Sequence[str]] = None,
        written_date_from=None,
        written_date_to=None,
        similarity_threshold: float = 0.0,
    ) -> List[PDFChunk]:
        """Vector search over verbalized_summary embeddings.

        Args:
            similarity_threshold: Minimum cosine similarity (0–1) a chunk must score
                to be returned. Cosine distance = 1 − similarity, so a threshold of
                0.4 discards anything with distance > 0.6. Defaults to 0.0 (no filter).
        """
        session = self.get_session()
        try:
            embedding_list = list(query_embedding)
            distance_col = PDFChunk.embedding.cosine_distance(embedding_list)

            q = session.query(PDFChunk).join(PDFDocument).options(joinedload(PDFChunk.document))

            if document_ids:
                q = q.filter(PDFChunk.document_id.in_(document_ids))
            if filenames:
                q = q.filter(PDFDocument.filename.in_(filenames))
            if page_min is not None:
                q = q.filter(PDFChunk.page_number >= page_min)
            if page_max is not None:
                q = q.filter(PDFChunk.page_number <= page_max)
            if sender_names:
                q = q.filter(PDFDocument.sender_name.in_(sender_names))
            if sender_companies:
                q = q.filter(PDFDocument.sender_company.in_(sender_companies))
            if written_date_from is not None:
                q = q.filter(PDFDocument.written_date >= written_date_from)
            if written_date_to is not None:
                q = q.filter(PDFDocument.written_date <= written_date_to)
            if similarity_threshold > 0.0:
                q = q.filter(distance_col <= (1.0 - similarity_threshold))

            q = q.order_by(distance_col)
            return q.limit(limit).all()
        finally:
            session.close()
