"""
PDF processing: Docling for layout parsing + Gemini for chart verbalization.

Workflow:
  1. Parse: Docling extracts page layout and page images
  2. Verbalize: Gemini describes every chart/graph/table on each page
  3. Store: One row per page with Text + Chart Summary
"""
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import PictureItem
from google import genai
from google.genai import types

VERBALIZE_MODEL = "models/gemini-2.5-flash"
TEXT_SUMMARY_MODEL = "models/gemini-2.5-flash"
IMAGE_RESOLUTION_SCALE = 2.0


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=key)


def _generate_with_retry(client: genai.Client, model: str, contents, max_attempts: int = 5):
    """Call generate_content with exponential backoff on 429 errors."""
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == max_attempts - 1:
                    raise
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                print(f"   [WARNING] Gemini still rate limited despite throttling, retrying in {wait}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            else:
                raise


def _verbalize_page_image(pil_image, model_name: str = VERBALIZE_MODEL) -> str:
    """
    Send page image to Gemini; get plain-text description of charts/graphs/tables.
    (Used for embedding/search. Raw text comes from Docling.)
    """
    # return "GEMINI_DISABLED: This is a placeholder for testing database ingestion."
    client = _get_client()

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

    response = _generate_with_retry(
        client, model_name,
        [prompt, types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")],
    )
    return response.text if hasattr(response, "text") else str(response)


def _summarize_text_block(text: str, purpose: str) -> str:
    """
    Summarize a (potentially long) text block with Gemini.

    purpose: short description used in the prompt (e.g. 'whole document', 'section')
    """
    if not text.strip():
        return ""

    client = _get_client()

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

    response = _generate_with_retry(client, TEXT_SUMMARY_MODEL, [prompt, text])
    return response.text if hasattr(response, "text") else str(response)


def _extract_sender_info(text: str) -> Dict[str, Optional[str]]:
    """Extract sender name, company, and sent date from document text using Gemini.

    Tries the first 2000 chars first, then the last 2000 chars for any fields
    still missing (sender/date sometimes appear in email footers at the bottom).
    """
    if not text.strip():
        return {"sender_name": None, "sender_company": None, "sent_date": None}

    client = _get_client()

    def _run_extraction(excerpt: str) -> Dict[str, Optional[str]]:
        prompt = (
            "Extract the sender/author name, their company, and the date this document was sent/published "
            "from this financial document excerpt. "
            "Return ONLY valid JSON with keys \"sender_name\", \"sender_company\", and \"sent_date\". "
            "Use null if not found. Format sent_date as YYYY-MM-DD.\n"
            "Example: {\"sender_name\": \"John Smith\", \"sender_company\": \"Goldman Sachs\", \"sent_date\": \"2024-03-15\"}\n\n"
            f"Document text:\n{excerpt}"
        )
        try:
            response = _generate_with_retry(client, TEXT_SUMMARY_MODEL, prompt)
            raw = response.text.strip() if hasattr(response, "text") else ""
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return {
                "sender_name": data.get("sender_name"),
                "sender_company": data.get("sender_company"),
                "sent_date": data.get("sent_date"),
            }
        except Exception:
            return {"sender_name": None, "sender_company": None, "sent_date": None}

    result = _run_extraction(text[:2000])

    # If any fields are still missing, try the end of the document
    if not all(result.values()):
        tail = text[-2000:] if len(text) > 2000 else ""
        if tail:
            tail_result = _run_extraction(tail)
            for key in ("sender_name", "sender_company", "sent_date"):
                if not result[key] and tail_result[key]:
                    result[key] = tail_result[key]

    return result


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
        pipeline_options.generate_page_images = False
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

        sender_info = _extract_sender_info(full_markdown)
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

        # ----- Image-level chunks -----
        image_idx = 0
        for item, _ in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue
            if item.image is None or item.image.pil_image is None:
                continue

            # Determine which page this image is on
            page_no = None
            if item.prov:
                page_no = item.prov[0].page_no

            # Look up parent section via page number
            sec = page_to_section.get(page_no) if page_no else None
            section_id = sec.section_id if sec else None
            section_title = sec.title if sec else None

            image_idx += 1
            verbalized = _verbalize_page_image(item.image.pil_image)
            if not verbalized.strip():
                continue
            metadata = {
                "level": "image",
                "image_index": image_idx,
                "page_number": page_no,
                "section_id": section_id,
                "section_title": section_title,
                "file_path": str(pdf_path.absolute()),
            }
            chunks.append(
                {
                    "raw_content": verbalized,
                    "verbalized_summary": verbalized,
                    "metadata": metadata,
                }
            )

        return chunks, total_doc_pages, file_size_bytes, sender_info

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
        Identify sections from Docling headings and map each page to its section.

        Strategy:
          - Walk all items; keep those whose type contains 'heading' or 'section'.
          - Use prov[0].page_no (Docling's provenance) to get the heading's page.
          - Each section starts at its heading page and ends just before the next heading.
        """
        sections: List[SectionInfo] = []

        # Collect (page_no, level, title) for every heading item.
        headings: List[Tuple[int, int, str]] = []

        for item, level in doc.iterate_items():
            title = getattr(item, "text", None) or getattr(item, "title", None)
            if not title:
                continue

            item_type = getattr(item, "label", None) or getattr(item, "category", None) or getattr(item, "kind", None)
            type_name = str(item_type).lower() if item_type is not None else ""

            if "heading" not in type_name and "section" not in type_name:
                continue

            # Use provenance to get the page this heading appears on.
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            try:
                page_no = int(prov[0].page_no)
            except (IndexError, AttributeError, TypeError, ValueError):
                continue

            headings.append((page_no, level, title.strip()))

        if not headings:
            return [], {}

        total_pages = len(doc.pages)

        # Each section runs from its heading page to the page before the next heading.
        for i, (start_page, level, title) in enumerate(headings):
            if i + 1 < len(headings):
                end_page = headings[i + 1][0] - 1
            else:
                end_page = total_pages
            end_page = max(end_page, start_page)

            section_id = f"sec_{i + 1}"
            sections.append(
                SectionInfo(
                    section_id=section_id,
                    title=title,
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
