"""
PDF processing: Docling for layout parsing + Gemini for chart verbalization.

Workflow:
  1. Parse: Docling extracts page layout and page images
  2. Verbalize: Gemini describes every chart/graph/table on each page
  3. Store: One row per page with Text + Chart Summary
"""
import io
import os
from pathlib import Path
from typing import List, Optional
#this is for the time delay to ensure the image is loaded
import time
import random

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import google.generativeai as genai

VERBALIZE_MODEL = "gemini-2.0-flash"
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
    #adding sleep to ensure the image is loaded
    sleep_time = random.uniform(15, 20)
    time.sleep(sleep_time)
    return response.text if hasattr(response, "text") else str(response)


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
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_size_bytes = pdf_path.stat().st_size

        # Parse with Docling
        conv_res = self.converter.convert(str(pdf_path))
        doc = conv_res.document

        chunks: List[dict] = []

        for page_no, page in doc.pages.items():
            # raw_content: Docling markdown (text + reconstructed tables)
            raw_content = self._get_page_text(doc, page_no) or ""

            # verbalized_summary: Gemini description of charts (for search embedding)
            verbalized_summary = ""
            if page.image is not None and page.image.pil_image is not None:
                verbalized_summary = _verbalize_page_image(page.image.pil_image)

            # metadata: page_number, file_path, etc.
            metadata = {
                "page_number": page_no,
                "file_path": str(pdf_path.absolute()),
            }

            # raw_content must be non-empty; use verbalized_summary if Docling returned nothing
            content_for_storage = raw_content if raw_content.strip() else verbalized_summary
            if not content_for_storage.strip():
                content_for_storage = f"[Page {page_no}]"

            chunks.append({
                "raw_content": content_for_storage,
                "verbalized_summary": verbalized_summary or None,
                "metadata": metadata,
            })

        total_pages = len(chunks)
        return chunks, total_pages, file_size_bytes

    def _get_page_text(self, doc, page_no: int) -> str:
        """Extract text for a single page from Docling document, if available."""
        try:
            if hasattr(doc, "filter"):
                filtered = doc.filter(pages=[page_no])
                return filtered.export_to_markdown()
        except Exception:
            pass
        return ""
