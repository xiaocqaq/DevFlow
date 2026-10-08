from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path


MAX_DOCUMENT_BYTES = 12 * 1024 * 1024
MAX_EXTRACTED_CHARS = 4_000_000
MAX_PDF_PAGES = 500
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


def _ensure_text_limit(text: str) -> str:
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ValueError(f"extracted text exceeds the {MAX_EXTRACTED_CHARS} character limit")
    return text


def decode_text(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise ValueError("binary content cannot be indexed as text")
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return _ensure_text_limit(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return ""


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES} page limit")
    pages: list[str] = []
    extracted_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            content = f"<!-- page: {page_number} -->\n{text}"
            extracted_chars += len(content) + (2 if pages else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"PDF text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            pages.append(content)
    return _ensure_text_limit("\n\n".join(pages))


def _paragraph_as_markdown(paragraph) -> str:
    text = (paragraph.text or "").strip()
    if not text:
        return ""
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
    heading = re.match(r"heading\s+(\d+)", style_name, re.IGNORECASE)
    if heading:
        level = min(6, max(1, int(heading.group(1))))
        return f"{'#' * level} {text}"
    lowered = style_name.lower()
    if "list bullet" in lowered:
        return f"- {text}"
    if "list number" in lowered:
        return f"1. {text}"
    return text


def _table_as_markdown(table) -> str:
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    return "\n".join([render(normalized[0]), render(["---"] * width), *(render(row) for row in normalized[1:])])


def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise ValueError(f"DOCX exceeds the {MAX_DOCX_ENTRIES} entry limit")
            uncompressed_size = sum(info.file_size for info in infos)
            if uncompressed_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"DOCX expanded size exceeds the {MAX_DOCX_UNCOMPRESSED_BYTES} byte limit"
                )
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("encrypted DOCX files are not supported")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid DOCX archive") from exc

    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(BytesIO(data))
    blocks: list[str] = []
    extracted_chars = 0
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            content = _paragraph_as_markdown(Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            content = _table_as_markdown(Table(child, document))
        else:
            content = ""
        if content:
            extracted_chars += len(content) + (2 if blocks else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"DOCX text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            blocks.append(content)
    return _ensure_text_limit("\n\n".join(blocks))


def extract_document_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    return decode_text(data)
