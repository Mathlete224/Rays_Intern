"""
PDF processing: Docling for layout parsing + Gemini for chart verbalization.

Workflow:
  1. Parse: Docling extracts page layout and page images
  2. Verbalize: Gemini describes every chart/graph/table on each page
  3. Store: One row per page with Text + Chart Summary
"""
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import google.generativeai as genai

VERBALIZE_MODEL = "gemini-2.0-flash"
TEXT_SUMMARY_MODEL = "gemini-2.0-flash"
IMAGE_RESOLUTION_SCALE = 2.0


def _configure_gemini(api_key: Optional[str] = None) -> None:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=key)


def _verbalize_page_image(pil_image, model_name: str = VERBALIZE_MODEL) -> str:
    """
    Send page image to Gemini; get plain-text description of charts/graphs/tables.
    (Used for embedding/search. Raw text comes from Docling.)
    """
    # return "GEMINI_DISABLED: This is a placeholder for testing database ingestion."
    _configure_gemini()
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are a financial analyst. Summarize this single page of a larger financial report. "
        "For every chart and table, extract the key data points, trends, and legends "
        "into a Markdown table format. Ensure that the visual insights "
        "(e.g., 'Revenue spiked in Q3') are explicitly written as text. "
        "Output the result in clean Markdown."
    )

    # Convert PIL to bytes for Gemini
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": buf.getvalue()}]
    )
    return response.text if hasattr(response, "text") else str(response)


def _summarize_text_block(text: str, purpose: str) -> str:
    """
    Summarize a (potentially long) text block with Gemini.

    purpose: short description used in the prompt (e.g. 'whole document', 'section')
    """
    if not text.strip():
        return ""

    _configure_gemini()
    model = genai.GenerativeModel(TEXT_SUMMARY_MODEL)

    prompt = (
        "You are a senior equity research analyst.\n"
        f"Summarize the following {purpose} from a financial report.\n"
        "- Capture company, report type, period, and key topics.\n"
        "- Use 3–8 bullet points.\n"
        "- Be factual and avoid speculation.\n"
    )

    # Truncate very long inputs to keep latency and token usage reasonable.
    if len(text) > 20000:
        text = text[:20000]

    response = model.generate_content([prompt, text])
    return response.text if hasattr(response, "text") else str(response)


@dataclass
class SectionInfo:
    section_id: str
    title: str
    level: int
    start_page: int
    end_page: int


class DoclingProcessor:
    """
    Parse PDFs with Docling and verbalize charts with Gemini.
    Produces one verbalized page per row.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_pdf(self, pdf_path: str) -> tuple:
        """
        Parse PDF with Docling and verbalize each page with Gemini.

        Returns:
            Tuple of (chunks, total_pages, file_size_bytes)
            Each chunk dict: raw_content (Docling markdown), verbalized_summary (Gemini), metadata

        Hierarchical strategy:
            - One document-level chunk with overall summary.
            - One chunk per major section (based on Docling headings).
            - One chunk per page, enriched with section context and sibling info.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_size_bytes = pdf_path.stat().st_size

        # Parse with Docling
        conv_res = self.converter.convert(str(pdf_path))
        doc = conv_res.document
        total_doc_pages = len(doc.pages)

        # Build a map of major sections from Docling's hierarchical structure.
        sections, page_to_section = self._build_section_map(doc)

        chunks: List[dict] = []

        # ----- Document-level chunk -----
        try:
            full_markdown = doc.export_to_markdown()
        except Exception:
            full_markdown = ""

        doc_summary = _summarize_text_block(full_markdown, purpose="whole document")
        doc_metadata = {
            "level": "document",
            "section_id": None,
            "section_title": None,
            "page_number": None,
            "page_span": [1, total_doc_pages] if total_doc_pages else None,
            "file_path": str(pdf_path.absolute()),
        }
        if full_markdown.strip() or doc_summary.strip():
            chunks.append(
                {
                    "raw_content": full_markdown or doc_summary or "[Document summary]",
                    "verbalized_summary": doc_summary or None,
                    "metadata": doc_metadata,
                }
            )

        # ----- Section-level chunks -----
        for sec in sections:
            pages = list(range(sec.start_page, sec.end_page + 1))
            section_text = self._get_pages_text(doc, pages)
            section_summary = _summarize_text_block(
                section_text, purpose=f"section '{sec.title}'"
            )
            metadata = {
                "level": "section",
                "section_id": sec.section_id,
                "section_title": sec.title,
                "section_level": sec.level,
                "page_span": [sec.start_page, sec.end_page],
                "file_path": str(pdf_path.absolute()),
            }
            content_for_storage = section_text if section_text.strip() else section_summary
            if not content_for_storage.strip():
                # Skip empty sections entirely
                continue
            chunks.append(
                {
                    "raw_content": content_for_storage,
                    "verbalized_summary": section_summary or None,
                    "metadata": metadata,
                }
            )

        # ----- Page-level chunks (with section + sibling context) -----
        for page_no, page in doc.pages.items():
            # raw_content: Docling markdown (text + reconstructed tables)
            raw_content = self._get_page_text(doc, page_no) or ""

            # verbalized_summary: Gemini description of charts (for search embedding)
            verbalized_summary = ""
            if page.image is not None and page.image.pil_image is not None:
                verbalized_summary = _verbalize_page_image(page.image.pil_image)

            # enrich with section context
            sec = page_to_section.get(page_no)
            section_id = sec.section_id if sec else None
            section_title = sec.title if sec else None
            section_level = sec.level if sec else None
            section_span: Optional[Tuple[int, int]] = (
                (sec.start_page, sec.end_page) if sec else None
            )

            prev_page_in_section = None
            next_page_in_section = None
            if section_span:
                start, end = section_span
                if page_no > start:
                    prev_page_in_section = page_no - 1
                if page_no < end:
                    next_page_in_section = page_no + 1

            metadata = {
                "level": "page",
                "page_number": page_no,
                "file_path": str(pdf_path.absolute()),
                "section_id": section_id,
                "section_title": section_title,
                "section_level": section_level,
                "section_page_span": list(section_span) if section_span else None,
                "prev_page_in_section": prev_page_in_section,
                "next_page_in_section": next_page_in_section,
            }

            # raw_content must be non-empty; use verbalized_summary if Docling returned nothing
            content_for_storage = raw_content if raw_content.strip() else verbalized_summary
            if not content_for_storage.strip():
                content_for_storage = f"[Page {page_no}]"

            chunks.append(
                {
                    "raw_content": content_for_storage,
                    "verbalized_summary": verbalized_summary or None,
                    "metadata": metadata,
                }
            )

        return chunks, total_doc_pages, file_size_bytes

    def _get_page_text(self, doc, page_no: int) -> str:
        """Extract text for a single page from Docling document, if available."""
        try:
            if hasattr(doc, "filter"):
                filtered = doc.filter(pages=[page_no])
                return filtered.export_to_markdown()
        except Exception:
            pass
        return ""

    def _get_pages_text(self, doc, pages: List[int]) -> str:
        """Extract text for a list of pages from Docling document."""
        if not pages:
            return ""
        try:
            if hasattr(doc, "filter"):
                filtered = doc.filter(pages=pages)
                return filtered.export_to_markdown()
        except Exception:
            pass
        return ""

    def _build_section_map(self, doc) -> Tuple[List[SectionInfo], Dict[int, SectionInfo]]:
        """
        Use Docling's iterate_items() to identify major sections and map pages to sections.

        Heuristic:
            - Treat heading-like items (based on their type/category) as section starts.
            - Use the item's page_span to infer which pages belong to each section.
            - Prefer lower 'level' values as higher-level sections.
        """
        sections: List[SectionInfo] = []

        # Discover section-like headings.
        for item, level in doc.iterate_items():
            title = getattr(item, "title", None) or getattr(item, "text", None)
            if not title:
                continue

            item_type = getattr(item, "category", None) or getattr(item, "kind", None)
            if item_type is None:
                type_name = ""
            else:
                type_name = str(item_type).lower()

            # Heuristic: keep items whose type mentions "heading" or "section".
            if "heading" not in type_name and "section" not in type_name:
                continue

            page_span = getattr(item, "page_span", None)
            if page_span is None:
                continue

            # Different Docling versions may expose page span slightly differently.
            start = getattr(page_span, "start", None) or getattr(page_span, "first", None)
            end = getattr(page_span, "end", None) or getattr(page_span, "last", None)
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start

            try:
                start_page = int(start)
                end_page = int(end)
            except (TypeError, ValueError):
                continue

            section_id = f"sec_{len(sections) + 1}"
            sections.append(
                SectionInfo(
                    section_id=section_id,
                    title=title.strip(),
                    level=level,
                    start_page=start_page,
                    end_page=end_page,
                )
            )

        # Sort sections by (start_page, level) so that higher-level (smaller level) sections
        # for the same page range come first.
        sections.sort(key=lambda s: (s.start_page, s.level))

        # Map each page to the most specific section (highest level number) that contains it.
        page_to_section: Dict[int, SectionInfo] = {}
        for sec in sections:
            for p in range(sec.start_page, sec.end_page + 1):
                current = page_to_section.get(p)
                if current is None or sec.level > current.level:
                    page_to_section[p] = sec

        return sections, page_to_section
