from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".pdf", ".docx"}
STRUCTURED_DATA_SUFFIXES = {".json", ".csv"}


@dataclass
class _Block:
    text: str
    start_line: int | None
    end_line: int | None
    kind: str = "paragraph"
    section_path: tuple[str, ...] = ()
    section_instance: tuple[tuple[str, int], ...] = ()
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


MAX_CHUNKS_PER_DOCUMENT = 5000


def _validate_limits(max_chars: int, overlap: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between zero and max_chars")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _line_for_offset(text: str, offset: int, first_line: int | None) -> int | None:
    if first_line is None:
        return None
    return first_line + text.count("\n", 0, max(0, offset))


def _split_long_text(block: _Block, max_chars: int) -> list[_Block]:
    if len(block.text) <= max_chars:
        return [block]

    units = [
        unit
        for unit in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+|\n+", block.text)
        if unit and unit.strip()
    ]
    if len(units) <= 1:
        units = [block.text[index : index + max_chars] for index in range(0, len(block.text), max_chars)]

    pieces: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(unit[index : index + max_chars] for index in range(0, len(unit), max_chars))
            continue
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)

    output: list[_Block] = []
    cursor = 0
    for piece in pieces:
        offset = block.text.find(piece, cursor)
        if offset < 0:
            offset = cursor
        end_offset = min(len(block.text), offset + len(piece))
        output.append(
            _Block(
                text=piece,
                start_line=_line_for_offset(block.text, offset, block.start_line),
                end_line=_line_for_offset(block.text, max(offset, end_offset - 1), block.start_line),
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        )
        cursor = end_offset
    return output


def _split_long_code_fence(block: _Block, max_chars: int) -> list[_Block]:
    if len(block.text) <= max_chars:
        return [block]
    lines = block.text.splitlines()
    if not lines:
        return []

    if max_chars <= 8:
        plain = _Block(
            text=block.text,
            start_line=block.start_line,
            end_line=block.end_line,
            kind="code_fragment",
            section_path=block.section_path,
            section_instance=block.section_instance,
            page=block.page,
            metadata=dict(block.metadata),
        )
        return _split_long_text(plain, max_chars)

    original_opener = lines[0] if lines[0].lstrip().startswith(("```", "~~~")) else "```"
    marker = original_opener.lstrip()[:3]
    opener = marker
    has_closer = len(lines) > 1 and lines[-1].lstrip().startswith(marker)
    body = lines[1:-1] if has_closer else lines[1:]
    closing = marker
    wrapper_size = len(opener) + len(closing) + 2
    body_limit = max_chars - wrapper_size

    if not body:
        return [
            _Block(
                text=f"{opener}\n{closing}",
                start_line=block.start_line,
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        ]

    output: list[_Block] = []
    current: list[str] = []
    current_size = 0
    first_body_line = (block.start_line + 1) if block.start_line is not None else None
    current_start = first_body_line

    def flush(end_line: int | None) -> None:
        nonlocal current, current_size, current_start
        if not current:
            return
        body_text = "\n".join(current)
        output.append(
            _Block(
                text=f"{opener}\n{body_text}\n{closing}",
                start_line=current_start,
                end_line=end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={
                    **block.metadata,
                    "fence_start_line": block.start_line,
                    "fence_end_line": block.end_line if has_closer else None,
                },
            )
        )
        current = []
        current_size = 0

    for body_index, line in enumerate(body):
        index = (first_body_line + body_index) if first_body_line is not None else None
        if len(line) > body_limit:
            flush((index - 1) if index is not None else None)
            for offset in range(0, len(line), body_limit):
                piece = line[offset : offset + body_limit]
                output.append(
                    _Block(
                        text=f"{opener}\n{piece}\n{closing}",
                        start_line=index,
                        end_line=index,
                        kind=block.kind,
                        section_path=block.section_path,
                        section_instance=block.section_instance,
                        page=block.page,
                        metadata={
                            **block.metadata,
                            "fence_start_line": block.start_line,
                            "fence_end_line": block.end_line if has_closer else None,
                        },
                    )
                )
            current_start = (index + 1) if index is not None else None
            continue
        line_size = len(line) + (1 if current else 0)
        if current and current_size + line_size > body_limit:
            flush((index - 1) if index is not None else None)
            current_start = index
        if not current:
            current_start = index
        current.append(line)
        current_size += line_size
    last_body_line = None
    if block.end_line is not None:
        last_body_line = block.end_line - (1 if has_closer else 0)
    flush(last_body_line)
    return output


def _expand_block(block: _Block, max_chars: int) -> list[_Block]:
    if block.kind == "code_fence":
        return _split_long_code_fence(block, max_chars)
    return _split_long_text(block, max_chars)


def _overlap_blocks(blocks: list[_Block], overlap: int) -> list[_Block]:
    if overlap <= 0 or not blocks:
        return []
    selected: list[_Block] = []
    remaining = overlap
    for block in reversed(blocks):
        if not block.text or block.kind == "code_fence":
            continue
        separator = 2 if selected else 0
        allowance = remaining - separator
        if allowance <= 0:
            break
        if len(block.text) <= allowance:
            selected.append(block)
            remaining -= len(block.text) + separator
            continue
        offset = len(block.text) - allowance
        lookback_start = max(0, offset - min(64, max(16, allowance * 2)))
        lookback = block.text[lookback_start:offset]
        sentence_boundaries = list(re.finditer(r"[。！？.!?；;]\s*", lookback))
        word_boundaries = list(re.finditer(r"\s+", lookback))
        if sentence_boundaries:
            fragment_offset = lookback_start + sentence_boundaries[-1].end()
        elif word_boundaries:
            fragment_offset = lookback_start + word_boundaries[-1].end()
        else:
            boundary_window = block.text[offset : offset + min(48, max(8, allowance // 3))]
            boundary = re.search(r"\s+|[。！？.!?；;]\s*", boundary_window)
            fragment_offset = offset + boundary.end() if boundary else offset
        selected.append(
            _Block(
                text=block.text[fragment_offset:],
                start_line=_line_for_offset(block.text, fragment_offset, block.start_line),
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={**block.metadata, "overlap_fragment": True},
            )
        )
        remaining = 0
        break
    return list(reversed(selected))


def _section_prefix(section_path: tuple[str, ...], page: int | None, limit: int | None = None) -> str:
    parts = [f"{'#' * min(index + 1, 6)} {title}" for index, title in enumerate(section_path)]
    if page is not None:
        parts.append(f"Page {page}")
    value = "\n".join(parts)
    if limit is None or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _prefix_for_chunk(
    section_path: tuple[str, ...],
    page: int | None,
    max_chars: int,
    *,
    has_body: bool,
) -> str:
    if not section_path and page is None:
        return ""
    if not has_body:
        limit = max_chars
    elif max_chars <= 3:
        limit = 0
    else:
        limit = min(512, max_chars // 3, max_chars - 3)
    return _section_prefix(section_path, page, limit=limit) if limit > 0 else ""


def _join_block_text(blocks: list[_Block]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _emit_chunk(
    blocks: list[_Block],
    *,
    source_path: str,
    chunk_type: str,
    child_index: int,
    max_chars: int,
) -> dict[str, Any]:
    first = blocks[0]
    body = _join_block_text(blocks)
    prefix = _prefix_for_chunk(first.section_path, first.page, max_chars, has_body=bool(body))
    content = f"{prefix}\n\n{body}" if prefix and body else (prefix or body)
    parent_id = _stable_id(
        "parent",
        source_path,
        first.section_instance or tuple((title, 0) for title in first.section_path),
        first.page,
    )
    effective_type = "markdown_code" if any(block.kind == "code_fence" for block in blocks) else chunk_type
    metadata: dict[str, Any] = {}
    for block in blocks:
        metadata.update(block.metadata)
    return {
        "content": content,
        "chunk_id": _stable_id("chunk", parent_id, child_index, content),
        "parent_id": parent_id,
        "child_index": child_index,
        "chunk_type": effective_type,
        "section_path": list(first.section_path),
        "section_title": first.section_path[-1] if first.section_path else None,
        "page": first.page,
        "start_line": min((block.start_line for block in blocks if block.start_line is not None), default=None),
        "end_line": max((block.end_line for block in blocks if block.end_line is not None), default=None),
        **metadata,
    }


def _pack_blocks(
    blocks: list[_Block],
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    chunk_type: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[_Block] = []
    current_key: tuple[tuple[tuple[str, int], ...], int | None] | None = None
    child_counts: dict[tuple[tuple[tuple[str, int], ...], int | None], int] = {}

    def flush() -> None:
        nonlocal current
        if not current or current_key is None:
            return
        child_index = child_counts.get(current_key, 0)
        chunks.append(
            _emit_chunk(
                current,
                source_path=source_path,
                chunk_type=chunk_type,
                child_index=child_index,
                max_chars=max_chars,
            )
        )
        child_counts[current_key] = child_index + 1

    for raw_block in blocks:
        key = (raw_block.section_instance, raw_block.page)
        prefix = _prefix_for_chunk(
            raw_block.section_path,
            raw_block.page,
            max_chars,
            has_body=bool(raw_block.text),
        )
        prefix_size = len(prefix)
        body_limit = max(1, max_chars - prefix_size - (2 if prefix_size else 0))
        for block in _expand_block(raw_block, body_limit):
            key = (block.section_instance, block.page)
            if current and key != current_key:
                flush()
                current = []
            current_key = key
            candidate_body = _join_block_text([*current, block])
            prefix = _prefix_for_chunk(
                block.section_path,
                block.page,
                max_chars,
                has_body=bool(candidate_body),
            )
            candidate_size = len(candidate_body) + len(prefix) + (2 if prefix else 0)
            if current and candidate_size > max_chars:
                previous = list(current)
                flush()
                current = _overlap_blocks(previous, overlap)
                while current:
                    candidate_body = _join_block_text([*current, block])
                    if len(candidate_body) + len(prefix) + (2 if prefix else 0) <= max_chars:
                        break
                    current.pop(0)
            current.append(block)
    flush()
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("chunk content exceeded max_chars")
    return chunks


def _plain_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(_Block(content, start_line, max(start_line, end_line)))
        current = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            flush(line_number - 1)
            continue
        if not current:
            start_line = line_number
        current.append(line)
    flush(len(lines))
    return blocks


def _plain_chunks(
    text: str,
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "plain")
    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        end = min(text_length, cursor + max_chars)
        if end < text_length:
            search_start = cursor + max(1, max_chars // 2)
            candidates = [
                text.rfind("\n\n", search_start, end),
                text.rfind("\n", search_start, end),
            ]
            boundary = max(candidates)
            if boundary > cursor:
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1)

        raw = text[cursor:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        content = raw[left_trim:right_trim]
        if content:
            start_offset = cursor + left_trim
            end_offset = cursor + right_trim
            child_index = len(chunks)
            chunks.append(
                {
                    "content": content,
                    "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "chunk_type": "text",
                    "section_path": [],
                    "section_title": None,
                    "page": None,
                    "start_line": _line_for_offset(text, start_offset, 1),
                    "end_line": _line_for_offset(text, max(start_offset, end_offset - 1), 1),
                }
            )
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
        if end >= text_length:
            break
        cursor = max(cursor + 1, end - overlap)
    return chunks


def _markdown_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    headings: list[tuple[int, str, int]] = []
    heading_occurrences: dict[tuple[tuple[tuple[str, int], ...], int, str], int] = {}
    current_page: int | None = None
    current: list[str] = []
    current_start = 1
    current_kind = "paragraph"
    fence_marker: str | None = None
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    page_re = re.compile(r"^<!--\s*page\s*:\s*(\d+)\s*-->$", re.IGNORECASE)

    def section_path() -> tuple[str, ...]:
        return tuple(title for _, title, _ in headings)

    def section_instance() -> tuple[tuple[str, int], ...]:
        return tuple((title, occurrence) for _, title, occurrence in headings)

    def flush(end_line: int) -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(
                _Block(
                    content,
                    current_start,
                    max(current_start, end_line),
                    kind=current_kind,
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                )
            )
        current = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if fence_marker is not None:
            current.append(line)
            if stripped.startswith(fence_marker):
                flush(line_number)
                fence_marker = None
                current_kind = "paragraph"
            continue

        page_match = page_re.match(stripped)
        if page_match:
            flush(line_number - 1)
            current_page = int(page_match.group(1))
            continue

        heading_match = heading_re.match(line)
        if heading_match:
            flush(line_number - 1)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            parent_instance = section_instance()
            occurrence_key = (parent_instance, level, title)
            occurrence = heading_occurrences.get(occurrence_key, 0)
            heading_occurrences[occurrence_key] = occurrence + 1
            headings.append((level, title, occurrence))
            blocks.append(
                _Block(
                    "",
                    line_number,
                    line_number,
                    kind="heading",
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                    metadata={"heading_line": line_number},
                )
            )
            continue

        if stripped.startswith(("```", "~~~")):
            flush(line_number - 1)
            current_start = line_number
            current_kind = "code_fence"
            fence_marker = stripped[:3]
            current = [line]
            continue

        if not stripped:
            flush(line_number - 1)
            continue
        if not current:
            current_start = line_number
        current.append(line)
    flush(len(lines))
    return blocks


def _json_blocks(text: str, max_chars: int) -> list[_Block]:
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _plain_blocks(text)

    formatted = json.dumps(value, ensure_ascii=False, indent=2)
    if len(formatted) <= max_chars:
        return [
            _Block(
                formatted,
                1,
                max(1, len(text.splitlines())),
                kind="json_value",
                metadata={"json_path": "$"},
            )
        ]

    blocks: list[_Block] = []
    if isinstance(value, dict):
        items = [(f"$.{key}", {key: item}) for key, item in value.items()]
    elif isinstance(value, list):
        items = [(f"$[{index}]", item) for index, item in enumerate(value)]
    else:
        items = [("$", value)]
    for json_path, item in items:
        blocks.append(
            _Block(
                json.dumps(item, ensure_ascii=False, indent=2),
                None,
                None,
                kind="json_value",
                section_path=(json_path,),
                section_instance=((json_path, 0),),
                metadata={"json_path": json_path},
            )
        )
    return blocks


def _csv_chunks(text: str, source_path: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    try:
        if len(text) > csv.field_size_limit():
            csv.field_size_limit(len(text))
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return _pack_blocks(
            _plain_blocks(text),
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            chunk_type="text",
        )
    if not rows:
        return []

    def render(row: list[str]) -> str:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow(row)
        return buffer.getvalue()

    header = render(rows[0])
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "csv")

    def append_chunk(
        content: str,
        *,
        row_start: int,
        row_end: int,
        chunk_type: str = "csv_rows",
        row_part: int | None = None,
        row_parts: int | None = None,
        precise_lines: bool = True,
    ) -> None:
        child_index = len(chunks)
        chunks.append(
            {
                "content": content,
                "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                "parent_id": parent_id,
                "child_index": child_index,
                "chunk_type": chunk_type,
                "section_path": [],
                "section_title": None,
                "page": None,
                "start_line": row_start if precise_lines else None,
                "end_line": row_end if precise_lines else None,
                "row_start": row_start,
                "row_end": row_end,
                **({"row_part": row_part, "row_parts": row_parts} if row_part is not None else {}),
            }
        )
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")

    if header and len(header) + 1 >= max_chars:
        header_parts = [header[index : index + max_chars] for index in range(0, len(header), max_chars)] or [""]
        for part_index, part in enumerate(header_parts):
            append_chunk(
                part,
                row_start=1,
                row_end=1,
                chunk_type="csv_header_part",
                row_part=part_index + 1,
                row_parts=len(header_parts),
                precise_lines="\n" not in part,
            )
        header_prefix = ""
    else:
        header_prefix = header

    if not rows[1:]:
        if not chunks:
            append_chunk(header, row_start=1, row_end=1, precise_lines="\n" not in header)
        return chunks

    prefix_size = len(header_prefix) + (1 if header_prefix else 0)
    row_budget = max_chars - prefix_size
    records: list[tuple[str, int, int | None, int | None, bool]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        rendered = render(row)
        parts = [rendered[index : index + row_budget] for index in range(0, len(rendered), row_budget)] or [""]
        for part_index, part in enumerate(parts):
            records.append((part, row_number, part_index + 1 if len(parts) > 1 else None, len(parts) if len(parts) > 1 else None, "\n" not in rendered))

    start = 0
    while start < len(records):
        selected: list[tuple[str, int, int | None, int | None, bool]] = []
        index = start
        while index < len(records):
            candidate_rows = [item[0] for item in selected]
            candidate_rows.append(records[index][0])
            candidate = "\n".join([header_prefix, *candidate_rows]) if header_prefix else "\n".join(candidate_rows)
            if selected and len(candidate) > max_chars:
                break
            selected.append(records[index])
            index += 1
        selected_text = [item[0] for item in selected]
        content = "\n".join([header_prefix, *selected_text]) if header_prefix else "\n".join(selected_text)
        row_start = min(item[1] for item in selected)
        row_end = max(item[1] for item in selected)
        row_part = selected[0][2] if len(selected) == 1 else None
        row_parts = selected[0][3] if len(selected) == 1 else None
        append_chunk(
            content,
            row_start=row_start,
            row_end=row_end,
            row_part=row_part,
            row_parts=row_parts,
            precise_lines=all(item[4] for item in selected),
        )
        if index >= len(records):
            break
        if overlap <= 0:
            start = index
            continue
        size = 0
        overlap_records = 0
        for record in reversed(selected):
            record_size = len(record[0])
            if overlap_records and size + record_size > overlap:
                break
            size += record_size
            overlap_records += 1
            if size >= overlap:
                break
        start = max(start + 1, index - overlap_records)
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("CSV chunk content exceeded max_chars")
    return chunks


def chunk_document(
    text: str,
    source_path: str = "document.txt",
    max_chars: int = 1600,
    overlap: int = 160,
) -> list[dict[str, Any]]:
    """Split a document on structural boundaries and return retrieval metadata."""

    _validate_limits(max_chars, overlap)
    if not text:
        return []
    normalized = _normalize_text(text)
    if not normalized.strip():
        return []

    suffix = Path(source_path).suffix.lower()
    if suffix == ".csv":
        return _csv_chunks(normalized, source_path, max_chars, overlap)
    if suffix in MARKDOWN_SUFFIXES:
        blocks = _markdown_blocks(normalized)
        chunk_type = "markdown_section"
    elif suffix == ".json":
        blocks = _json_blocks(normalized, max_chars)
        chunk_type = "json_value"
    else:
        return _plain_chunks(
            normalized,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
        )
    chunks = _pack_blocks(
        blocks,
        source_path=source_path,
        max_chars=max_chars,
        overlap=overlap,
        chunk_type=chunk_type,
    )
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
    return chunks


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    """Backward-compatible plain-text API used by older indexing paths."""

    return [
        chunk["content"]
        for chunk in chunk_document(text, "document.txt", max_chars=max_chars, overlap=overlap)
    ]


def _line_windows(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
    overlap_lines: int = 6,
) -> list[tuple[str, int, int]]:
    output: list[tuple[str, int, int]] = []
    cursor = max(0, start_line - 1)
    stop = min(len(lines), end_line)
    while cursor < stop:
        if len(lines[cursor]) > max_chars:
            for offset in range(0, len(lines[cursor]), max_chars):
                output.append((lines[cursor][offset : offset + max_chars], cursor + 1, cursor + 1))
            cursor += 1
            continue
        size = 0
        end = cursor
        while end < stop:
            line_size = len(lines[end]) + (1 if end > cursor else 0)
            if end > cursor and size + line_size > max_chars:
                break
            size += line_size
            end += 1
        if end > cursor:
            content = "\n".join(lines[cursor:end]).strip("\n")
            if content:
                output.append((content, cursor + 1, end))
        if end >= stop:
            break
        cursor = max(cursor + 1, end - overlap_lines)
    return output


def _python_regions(text: str, max_chars: int) -> list[tuple[int, int, str, str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    lines = text.splitlines()
    regions: list[tuple[int, int, str, str]] = []
    cursor = 1

    def node_start(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", []) or []
        positions = [getattr(node, "lineno", 1), *(getattr(item, "lineno", 1) for item in decorators)]
        return min(positions)

    def append_node(node: ast.AST, parent: str | None = None) -> None:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        name = str(getattr(node, "name", "module"))
        symbol = f"{parent}.{name}" if parent else name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if isinstance(node, ast.ClassDef) and len("\n".join(lines[start - 1 : end])) > max_chars:
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                method_cursor = start
                first_method_start = node_start(methods[0])
                if first_method_start > method_cursor:
                    regions.append((method_cursor, first_method_start - 1, symbol, "class_header"))
                for index, method in enumerate(methods):
                    method_start = node_start(method)
                    next_start = node_start(methods[index + 1]) if index + 1 < len(methods) else end + 1
                    regions.append((method_start, next_start - 1, f"{symbol}.{method.name}", "method"))
                return
        regions.append((start, end, symbol, kind))

    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in top_level:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        if start > cursor:
            regions.append((cursor, start - 1, "module", "module"))
        append_node(node)
        cursor = max(cursor, end + 1)
    if cursor <= len(lines):
        regions.append((cursor, len(lines), "module", "module"))
    return [region for region in regions if region[0] <= region[1]]


def _generic_symbol(line: str, suffix: str) -> tuple[str, str] | None:
    patterns: list[tuple[re.Pattern[str], str]] = []
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        patterns = [
            (
                re.compile(
                    r"^\s*(?:export\s+(?:default\s+)?)?(?:(?:async\s+)?function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
                ),
                "symbol",
            ),
            (
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
                ),
                "function",
            ),
        ]
    elif suffix == ".go":
        patterns = [(re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_]\w*)"), "symbol")]
    elif suffix == ".rs":
        patterns = [
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)"), "symbol")
        ]
    elif suffix in {".java", ".kt", ".kts", ".cs", ".cpp", ".c", ".h", ".hpp", ".php", ".rb", ".swift", ".scala"}:
        patterns = [
            (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed|data|open)\s+)*(?:class|interface|enum|struct|trait|record)\s+([A-Za-z_]\w*)"), "type")
        ]
    elif suffix == ".sql":
        patterns = [(re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TABLE|VIEW)\s+([\w.]+)", re.IGNORECASE), "sql_object")]
    elif suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        patterns = [(re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*(?:\(\))?\s*\{"), "function")]
    for pattern, kind in patterns:
        match = pattern.match(line)
        if match:
            return match.group(1), kind
    return None


def _generic_regions(text: str, suffix: str) -> list[tuple[int, int, str, str]]:
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        symbol = _generic_symbol(line, suffix)
        if symbol:
            starts.append((line_number, symbol[0], symbol[1]))
    if not starts:
        return [(1, len(lines), "module", "module")] if lines else []

    regions: list[tuple[int, int, str, str]] = []
    if starts[0][0] > 1:
        regions.append((1, starts[0][0] - 1, "module", "module"))
    for index, (start, name, kind) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        regions.append((start, end, name, kind))
    return regions


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or Path(path).name.lower()


def chunk_code_text(text: str, path: str, max_chars: int = 2200) -> list[dict[str, Any]]:
    """Split source on language symbols, with parser and line-window fallbacks."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []
    suffix = Path(path).suffix.lower()
    language = _language_for_path(path)

    if suffix in MARKDOWN_SUFFIXES | STRUCTURED_DATA_SUFFIXES | {".txt"}:
        document_chunks = chunk_document(normalized, path, max_chars=max_chars, overlap=max(0, min(160, max_chars // 10)))
        output: list[dict[str, Any]] = []
        for chunk in document_chunks:
            symbol = chunk.get("section_title") or "document"
            header = f"File: {path}\nSection: {symbol}\nLines: {chunk['start_line']}-{chunk['end_line']}"
            output.append({**chunk, "symbol": symbol, "symbol_kind": chunk["chunk_type"], "language": language, "header": header})
        return output

    lines = normalized.splitlines()
    regions = _python_regions(normalized, max_chars) if suffix == ".py" else _generic_regions(normalized, suffix)
    if not regions:
        regions = [(1, len(lines), "module", "module")]

    chunks: list[dict[str, Any]] = []
    symbol_occurrences: dict[str, int] = {}
    for start_line, end_line, symbol, symbol_kind in regions:
        symbol_occurrence = symbol_occurrences.get(symbol, 0)
        symbol_occurrences[symbol] = symbol_occurrence + 1
        parent_id = _stable_id("code-parent", path, symbol, symbol_occurrence)
        for part_index, (content, part_start, part_end) in enumerate(
            _line_windows(lines, start_line, end_line, max_chars=max_chars)
        ):
            if not content.strip():
                continue
            chunk_id = _stable_id("code", parent_id, part_index, content)
            part_suffix = f" (part {part_index + 1})" if part_index else ""
            chunks.append(
                {
                    "content": content,
                    "start_line": part_start,
                    "end_line": part_end,
                    "symbol": symbol,
                    "symbol_kind": symbol_kind,
                    "language": language,
                    "chunk_type": "code_symbol" if symbol != "module" else "code_module",
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "child_index": part_index,
                    "symbol_occurrence": symbol_occurrence,
                    "header": (
                        f"File: {path}\nLanguage: {language}\nSymbol: {symbol}{part_suffix}\n"
                        f"Lines: {part_start}-{part_end}"
                    ),
                }
            )
    return chunks


def chunk_pr_files(files: list[dict[str, Any]], max_chars: int = 1800) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for file in files:
        patch = file.get("patch") or ""
        filename = str(file.get("filename") or "patch.diff")
        for index, content in enumerate(chunk_text(patch, max_chars=max_chars, overlap=0)):
            chunks.append(
                {
                    "filename": filename,
                    "chunk_index": index,
                    "chunk_id": _stable_id("patch", filename, content),
                    "content": content,
                }
            )
    return chunks
