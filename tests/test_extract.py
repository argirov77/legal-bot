from pathlib import Path

import pytest

from app import extract


@pytest.fixture
def pdf_bytes() -> bytes:
    # Minimal PDF document with extractable text "Hello PDF"
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 53 >>\nstream\nBT /F1 12 Tf 72 120 Td (Hello PDF) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n0000000110 00000 n \n"
        b"0000000276 00000 n \n0000000393 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n452\n%%EOF\n"
    )


def _make_txt(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text("Hello from TXT!", encoding="utf-8")
    return path


def _make_docx(tmp_path: Path) -> Path:
    docx_mod = pytest.importorskip("docx", reason="python-docx is required for DOCX tests")
    document = docx_mod.Document()
    document.add_paragraph("Hello from DOCX!")
    path = tmp_path / "sample.docx"
    document.save(path)
    return path


def _make_pdf(tmp_path: Path, pdf_bytes: bytes) -> Path:
    pytest.importorskip(
        "pdfminer.high_level", reason="pdfminer.six is required for PDF extraction tests"
    )
    path = tmp_path / "sample.pdf"
    path.write_bytes(pdf_bytes)
    return path


@pytest.mark.parametrize(
    "maker",
    (_make_txt, _make_docx, _make_pdf),
)
def test_extract_text_returns_content(tmp_path: Path, maker, pdf_bytes: bytes) -> None:
    path = maker(tmp_path, pdf_bytes) if maker is _make_pdf else maker(tmp_path)
    text = extract.extract_text(path)
    assert isinstance(text, str)
    assert text.strip()


def test_extract_pdf_triggers_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "needs_ocr.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    calls = {"count": 0}

    def fake_extract_pdf(_: Path) -> str:
        calls["count"] += 1
        return "" if calls["count"] == 1 else "OCR TEXT"

    monkeypatch.setattr(extract, "_extract_pdf", fake_extract_pdf)

    recorded_command: dict[str, list[str]] = {}

    def fake_run(cmd, check, stdout, stderr):  # noqa: ANN001
        recorded_command["cmd"] = cmd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(extract.subprocess, "run", fake_run)

    text = extract.extract_text(pdf_path, use_ocr=True)

    assert text == "OCR TEXT"
    assert recorded_command["cmd"][0] == "ocrmypdf"
    assert recorded_command["cmd"][1] == "--skip-text"
    assert recorded_command["cmd"][2] == str(pdf_path)
