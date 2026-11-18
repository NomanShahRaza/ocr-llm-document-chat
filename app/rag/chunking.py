from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..textfix import literal_newlines

SKIP_LABELS = {
    'number',
    'footnote',
    'header',
    'header_image',
    'footer',
    'footer_image',
    'aside_text',
    'figure',
    'image',
    'chart',
    'seal',
}


def _clean_text(text: str) -> str:
    text = literal_newlines(text or '', ' ')
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', ' ', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'[#>*`_|]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_long(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            cut = text.rfind(' ', start + chunk_chars // 2, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def chunks_from_document(
    job_id: str,
    filename: str,
    document: dict[str, Any] | None,
    markdown: str | None,
    chunk_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    pages = (document or {}).get('pages') or []

    if pages:
        for page_idx, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                continue
            res = page.get('res') if isinstance(page.get('res'), dict) else page
            page_no = page.get('_app_page_number')
            if page_no is None:
                page_no = page_idx
            else:
                try:
                    page_no = int(page_no)
                except Exception:
                    page_no = page_idx
            if page_no < 1:
                page_no = page_idx
            blocks = res.get('parsing_res_list') or []
            buffer: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                label = str(block.get('block_label') or '')
                if label in SKIP_LABELS:
                    continue
                content = _clean_text(str(block.get('block_content') or ''))
                if len(content) < 12:
                    continue
                if label.endswith('title') or label in {'doc_title', 'paragraph_title'}:
                    if buffer:
                        chunks.extend(
                            _emit(job_id, filename, page_no, '\n'.join(buffer), chunk_chars, overlap, len(chunks))
                        )
                        buffer = []
                    buffer.append(content)
                else:
                    buffer.append(content)
            if buffer:
                # Newlines keep OCR block boundaries so bullets and list items stay separable.
                chunks.extend(
                    _emit(job_id, filename, page_no, '\n'.join(buffer), chunk_chars, overlap, len(chunks))
                )
        if chunks:
            return chunks

    text = _clean_text(markdown or '')
    if not text:
        return []
    for piece in _split_long(text, chunk_chars, overlap):
        chunks.append({
            'chunk_id': f'{job_id}:{len(chunks)}',
            'job_id': job_id,
            'filename': filename,
            'page': 1,
            'text': piece,
        })
    return chunks


def _emit(
    job_id: str,
    filename: str,
    page: int,
    text: str,
    chunk_chars: int,
    overlap: int,
    start_index: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for piece in _split_long(text, chunk_chars, overlap):
        out.append({
            'chunk_id': f'{job_id}:{start_index + len(out)}',
            'job_id': job_id,
            'filename': filename,
            'page': page,
            'text': piece,
        })
    return out


TITLE_LABELS = {'doc_title', 'paragraph_title', 'title'}


def document_outline(job_id: str, result_dir: Path, preview_chars: int = 220) -> dict[str, Any] | None:
    """Title plus per-page headings and a short body preview, used as document-level evidence."""
    folder = result_dir / job_id
    doc_path = folder / 'document.json'
    if not doc_path.exists():
        return None
    try:
        document = json.loads(doc_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    filename = document.get('source_file') or job_id
    meta_path = folder / 'job.json'
    if meta_path.exists():
        try:
            filename = json.loads(meta_path.read_text(encoding='utf-8')).get('filename') or filename
        except Exception:
            pass

    title = ''
    pages_out: list[dict[str, Any]] = []
    seen_headings: set[str] = set()
    for page_idx, page in enumerate(document.get('pages') or [], start=1):
        if not isinstance(page, dict):
            continue
        res = page.get('res') if isinstance(page.get('res'), dict) else page
        page_no = page.get('_app_page_number') or page_idx
        headings: list[str] = []
        body: list[str] = []
        for block in res.get('parsing_res_list') or []:
            if not isinstance(block, dict):
                continue
            label = str(block.get('block_label') or '')
            if label in SKIP_LABELS:
                continue
            content = _clean_text(str(block.get('block_content') or ''))
            if not content:
                continue
            if label in TITLE_LABELS or label.endswith('title'):
                if label == 'doc_title' and not title:
                    title = content
                key = content.lower()
                if key not in seen_headings and len(content) <= 160:
                    seen_headings.add(key)
                    headings.append(content)
            elif len(content) >= 20:
                body.append(content)
        preview = ' '.join(body)
        if len(preview) > preview_chars:
            cut = preview.rfind(' ', 0, preview_chars)
            preview = preview[: cut if cut > preview_chars // 2 else preview_chars].rstrip() + '…'
        if headings or preview:
            pages_out.append({'page': int(page_no), 'headings': headings, 'preview': preview})

    if not title:
        first = next((p for p in pages_out if p['headings']), None)
        title = first['headings'][0] if first else ''
    return {
        'job_id': job_id,
        'filename': filename,
        'title': title,
        'page_count': document.get('page_count') or len(pages_out),
        'pages': pages_out,
    }


def load_job_corpus(job_id: str, result_dir: Path, chunk_chars: int, overlap: int) -> list[dict[str, Any]]:
    folder = result_dir / job_id
    doc_path = folder / 'document.json'
    md_path = folder / 'document.md'
    meta_path = folder / 'job.json'
    if not doc_path.exists() and not md_path.exists():
        return []

    filename = job_id
    document = None
    markdown = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            filename = meta.get('filename') or filename
        except Exception:
            pass
    if doc_path.exists():
        try:
            document = json.loads(doc_path.read_text(encoding='utf-8'))
            filename = document.get('source_file') or filename
        except Exception:
            document = None
    if md_path.exists():
        markdown = md_path.read_text(encoding='utf-8', errors='replace')
    return chunks_from_document(job_id, filename, document, markdown, chunk_chars, overlap)
