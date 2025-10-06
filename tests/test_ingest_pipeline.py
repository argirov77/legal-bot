import io
from pathlib import Path

import pytest
from docx import Document
from fpdf import FPDF

from src.app.ingest import IngestPipeline, IngestPipelineConfig
from src.app.ingest.format_detection import DocumentFormat, DocumentFormatDetector


@pytest.fixture()
def pipeline() -> IngestPipeline:
    config = IngestPipelineConfig(chunk_chars=120, overlap_chars=20)
    return IngestPipeline(config=config)


def test_document_format_detection_by_suffix():
    assert DocumentFormatDetector.detect("sample.pdf") is DocumentFormat.PDF
    assert DocumentFormatDetector.detect("sample.docx") is DocumentFormat.DOCX
    assert DocumentFormatDetector.detect("notes.txt") is DocumentFormat.TXT


def test_text_ingestion_chunking(pipeline: IngestPipeline):
    text = (
        "Първи абзац с правни твърдения. Той съдържа важно основание.\n\n"
        "Втори абзац, който продължава с аргументация и добавя примери.\n\n"
        "Трети абзац завършва документа."
    )
    chunks = pipeline.ingest(text.encode("utf-8"), "memo.txt")
    assert len(chunks) >= 2
    previous_end = -1
    for index, chunk in enumerate(chunks):
        assert chunk.metadata.file_name == "memo.txt"
        assert chunk.metadata.chunk_index == index
        assert chunk.metadata.char_start > previous_end
        assert chunk.metadata.char_end > chunk.metadata.char_start
        previous_end = chunk.metadata.char_end
        assert "  " not in chunk.content


def _build_docx_bytes(text: str) -> bytes:
    document = Document()
    for paragraph in text.split("\n\n"):
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _build_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.exists():
        pytest.skip("DejaVu font not available for PDF generation")
    pdf.add_font("DejaVu", "", fname=str(font_path), uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest="S").encode("latin1")


def test_docx_ingestion_detects_language(pipeline: IngestPipeline):
    text = "Здравей свят. Това е документ на български език."
    docx_bytes = _build_docx_bytes(text)
    chunks = pipeline.ingest(docx_bytes, "contract.docx")
    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata.language == "bg"
    assert "Здравей" in chunks[0].content


def test_pdf_ingestion_uses_text_layer(pipeline: IngestPipeline):
    text = "Съдържание на PDF документа с правни бележки."
    pdf_bytes = _build_pdf_bytes(text)
    chunks = pipeline.ingest(pdf_bytes, "evidence.pdf")
    assert chunks
    assert chunks[0].metadata.page == 1
    assert "PDF документа" in chunks[0].content
