import io
import zipfile

import pytest

from app.services.rag import chunking, document_extraction
from app.services.rag.chunking import chunk_code_text, chunk_document, chunk_text


def test_chunk_text_keeps_paragraphs_and_respects_size_limit() -> None:
    text = "第一段说明登录流程。" * 8 + "\n\n" + "第二段说明刷新令牌。" * 8 + "\n\n" + "第三段说明退出流程。" * 8

    chunks = chunk_text(text, max_chars=190, overlap=20)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 190 for chunk in chunks)
    assert "\n\n" in chunks[0]


def test_chunk_text_rejects_invalid_overlap() -> None:
    try:
        chunk_text("content", max_chars=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_markdown_chunking_preserves_heading_context_and_code_fences() -> None:
    text = """# Authentication

The service verifies access tokens before every request.

## Refresh tokens

Refresh tokens are rotated after a successful exchange.

```python
def rotate_token(token: str) -> str:
    return token + "-rotated"
```
"""

    chunks = chunk_document(text, "auth.md", max_chars=180, overlap=20)

    assert chunks
    assert all(len(chunk["content"]) <= 180 for chunk in chunks)
    assert any(chunk["section_path"] == ["Authentication", "Refresh tokens"] for chunk in chunks)
    code_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "markdown_code")
    assert code_chunk["content"].count("```") == 2
    assert code_chunk["section_title"] == "Refresh tokens"
    assert code_chunk["start_line"] <= code_chunk["end_line"]


def test_markdown_overlap_starts_at_a_readable_boundary() -> None:
    text = "# A\n\nalpha bravo charlie delta echo foxtrot\n\nsecond evidence"

    chunks = chunk_document(text, "overlap.md", max_chars=55, overlap=20)

    assert len(chunks) == 2
    overlap_body = chunks[1]["content"].split("\n\n", 1)[1]
    assert overlap_body.split()[0] in {"charlie", "delta"}


def test_markdown_chunk_ids_are_stable_for_the_same_content() -> None:
    text = "# Deploy\n\nRun the migration before restarting the API."

    first = chunk_document(text, "runbook.md", max_chars=200, overlap=20)
    second = chunk_document(text, "runbook.md", max_chars=200, overlap=20)

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert [chunk["parent_id"] for chunk in first] == [chunk["parent_id"] for chunk in second]


def test_python_chunking_uses_ast_symbol_boundaries() -> None:
    source = """import os

def load_settings() -> dict:
    return {"environment": os.getenv("ENV", "dev")}

class TokenService:
    def issue(self, user_id: str) -> str:
        return f"token:{user_id}"
"""

    chunks = chunk_code_text(source, "app/token_service.py", max_chars=220)

    assert {chunk["symbol"] for chunk in chunks} >= {"module", "load_settings", "TokenService"}
    function = next(chunk for chunk in chunks if chunk["symbol"] == "load_settings")
    assert function["symbol_kind"] == "function"
    assert function["start_line"] == 3
    assert function["header"].startswith("File: app/token_service.py")


def test_typescript_chunking_recognizes_exported_functions_and_arrow_functions() -> None:
    source = """import { api } from './api';

export async function loadUser(id: string) {
  return api.get(`/users/${id}`);
}

export const saveUser = async (id: string) => {
  return api.post(`/users/${id}`);
};
"""

    chunks = chunk_code_text(source, "src/users.ts", max_chars=220)

    assert {chunk["symbol"] for chunk in chunks} >= {"module", "loadUser", "saveUser"}
    assert all(chunk["language"] == "ts" for chunk in chunks)


def test_invalid_python_falls_back_without_losing_content() -> None:
    source = "def broken(:\n    return 1\n"

    chunks = chunk_code_text(source, "broken.py", max_chars=80)

    assert chunks
    assert "def broken" in "\n".join(chunk["content"] for chunk in chunks)
    assert chunks[0]["symbol"] == "module"


def test_csv_chunking_repeats_header_and_tracks_row_ranges() -> None:
    text = "id,name\n1,Alice\n2,Bob\n3,Carol\n4,David"

    chunks = chunk_document(text, "users.csv", max_chars=25, overlap=5)

    assert len(chunks) >= 2
    assert all(chunk["content"].splitlines()[0] == "id,name" for chunk in chunks)
    assert all(chunk["chunk_type"] == "csv_rows" for chunk in chunks)
    assert all(chunk["row_start"] <= chunk["row_end"] for chunk in chunks)


def test_small_json_manifest_stays_in_one_chunk() -> None:
    chunks = chunk_document('{"name":"demo","version":"1.0.0"}', "package.json", max_chars=200, overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["json_path"] == "$"
    assert '"name": "demo"' in chunks[0]["content"]


def test_plain_text_overlap_is_present_and_line_numbers_keep_leading_blanks() -> None:
    text = "\n\n" + "".join(f"{index:03d}" for index in range(100))

    chunks = chunk_document(text, "notes.txt", max_chars=60, overlap=12)

    assert len(chunks) > 2
    assert chunks[0]["start_line"] == 3
    assert all(previous["content"][-12:] == current["content"][:12] for previous, current in zip(chunks, chunks[1:]))


def test_csv_long_rows_are_split_without_data_loss() -> None:
    text = "id,data\n1," + "x" * 80 + "\n2,ok"

    chunks = chunk_document(text, "long.csv", max_chars=20, overlap=2)

    assert all(len(chunk["content"]) <= 20 for chunk in chunks)
    assert sum(chunk["content"].count("x") for chunk in chunks) == 80
    assert {chunk["row_start"] for chunk in chunks if chunk.get("row_part")} == {2}


def test_csv_fields_larger_than_the_stdlib_default_remain_structured() -> None:
    rendered_row = "1," + "x" * 140_000

    chunks = chunk_document(f"id,data\n{rendered_row}", "large-field.csv", max_chars=5000, overlap=0)

    reconstructed = "".join(chunk["content"].split("\n", 1)[1] for chunk in chunks)
    assert reconstructed == rendered_row
    assert all(chunk["chunk_type"] == "csv_rows" for chunk in chunks)


def test_csv_oversized_header_is_lossless_and_size_limited() -> None:
    header = "column_" * 8

    chunks = chunk_document(f"{header}\nvalue", "header.csv", max_chars=20, overlap=2)

    header_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "csv_header_part"]
    assert "".join(chunk["content"] for chunk in header_chunks) == header
    assert any("value" in chunk["content"] for chunk in chunks)
    assert all(len(chunk["content"]) <= 20 for chunk in chunks)


def test_repeated_and_empty_markdown_sections_remain_distinct() -> None:
    text = "# API\n\n## Auth\n\nfirst\n\n## Auth\n\nsecond\n\n### Empty"

    chunks = chunk_document(text, "api.md", max_chars=100, overlap=10)

    auth_chunks = [chunk for chunk in chunks if chunk["section_title"] == "Auth"]
    assert len(auth_chunks) == 2
    assert auth_chunks[0]["parent_id"] != auth_chunks[1]["parent_id"]
    assert any(chunk["section_title"] == "Empty" for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_long_markdown_prefix_and_fence_obey_limit_with_exact_body_lines() -> None:
    fence = "`" * 3
    text = "\n".join(
        [
            "# " + "H" * 200,
            "",
            fence + "python-with-a-very-long-info-string",
            "line1=1111",
            "line2=2222",
            "line3=3333",
            fence,
        ]
    )

    chunks = chunk_document(text, "fence.md", max_chars=30, overlap=2)

    assert all(len(chunk["content"]) <= 30 for chunk in chunks)
    line2_chunk = next(chunk for chunk in chunks if "line2=2222" in chunk["content"])
    assert line2_chunk["start_line"] == 5
    assert line2_chunk["end_line"] == 5
    assert line2_chunk["fence_start_line"] == 3


def test_tiny_markdown_budget_falls_back_without_zero_width_fence_split() -> None:
    text = "# A\n\n```\nvalue\n```"

    chunks = chunk_document(text, "tiny.md", max_chars=8, overlap=1)

    assert chunks
    assert all(len(chunk["content"]) <= 8 for chunk in chunks)


def test_duplicate_code_symbols_receive_unique_stable_ids() -> None:
    source = "def same():\n    pass\n\ndef same():\n    pass"

    first = chunk_code_text(source, "same.py", max_chars=100)
    second = chunk_code_text(source, "same.py", max_chars=100)
    same_chunks = [chunk for chunk in first if chunk["symbol"] == "same"]

    assert len(same_chunks) == 2
    assert same_chunks[0]["chunk_id"] != same_chunks[1]["chunk_id"]
    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]


def test_large_json_does_not_claim_invented_source_lines() -> None:
    text = '{\n  "first": "' + "a" * 40 + '",\n  "second": "' + "b" * 40 + '"\n}'

    chunks = chunk_document(text, "data.json", max_chars=30, overlap=3)

    assert len(chunks) > 1
    assert all(chunk["start_line"] is None and chunk["end_line"] is None for chunk in chunks)


def test_text_extraction_rejects_binary_and_excessive_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(document_extraction, "MAX_EXTRACTED_CHARS", 8)

    with pytest.raises(ValueError, match="character limit"):
        document_extraction.decode_text(b"more than eight characters")
    with pytest.raises(ValueError, match="binary"):
        document_extraction.decode_text(b"abc\x00def")


def test_docx_archive_expansion_is_checked_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 32)
    monkeypatch.setattr(document_extraction, "MAX_DOCX_UNCOMPRESSED_BYTES", 16)

    with pytest.raises(ValueError, match="expanded size"):
        document_extraction.extract_document_text("bomb.docx", buffer.getvalue())


def test_pdf_page_limit_is_checked_before_page_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    from pypdf import PdfWriter

    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.write(buffer)
    monkeypatch.setattr(document_extraction, "MAX_PDF_PAGES", 1)

    with pytest.raises(ValueError, match="page limit"):
        document_extraction.extract_document_text("many-pages.pdf", buffer.getvalue())


def test_document_chunk_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunking, "MAX_CHUNKS_PER_DOCUMENT", 2)

    with pytest.raises(ValueError, match="chunk limit"):
        chunking.chunk_document("abcdef", "tiny.txt", max_chars=2, overlap=0)
