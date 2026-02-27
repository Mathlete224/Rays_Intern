from pipeline import PDFSummarizerPipeline
import os
from dotenv import load_dotenv
load_dotenv()  


DB_URL = os.getenv("PDF_SUMMARIZER_DB_URL")
pipeline = PDFSummarizerPipeline(database_url=DB_URL)

pdf_file = "/Users/davidfu/Desktop/Rays_Intern/PDF_summarizer/research_pdfs/1_91APP (6741) - HK Roadshow - Nov 7-8th, 2024_91App HK Road Nov 7-8.pdf"

result = pipeline.process_single_pdf(pdf_file)
print(result)