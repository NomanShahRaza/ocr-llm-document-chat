from __future__ import annotations

import html
import re
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .config import XIAOMI_MODEL

DOCUMENT_PROMPT = (
    'Extract all information from the main body of the document image and represent it '
    'in markdown format, ignoring headers and footers. Tables should be expressed in OTSL '
    'format, formulas in the document should be represented using LATEX format, and the '
    'parsing should be organized according to the reading order.'
)

_OTSL_TOKEN = re.compile(r'<(fcel|ecel|lcel|ucel|nl)>', re.IGNORECASE)
_THINK_BLOCK = re.compile(r'<think>[\s\S]*?</think>', re.IGNORECASE)
_DISPLAY_LATEX = re.compile(r'\\\[([\s\S]+?)\\\]')
_INLINE_LATEX = re.compile(r'\\\((.+?)\\\)')


class XiaomiEngine:
    """Local Xiaomi-OCR-0 document parser (xiaomi-research/Xiaomi-OCR-0)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = 'cpu'
        self.loaded = False

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.float32 if self._device == 'cpu' else torch.float16
            self._processor = AutoProcessor.from_pretrained(XIAOMI_MODEL)
            try:
                model = AutoModelForImageTextToText.from_pretrained(XIAOMI_MODEL, dtype=dtype)
            except TypeError:
                model = AutoModelForImageTextToText.from_pretrained(XIAOMI_MODEL, torch_dtype=dtype)
            self._model = model.to(self._device).eval()
            self.loaded = True

    def release(self) -> None:
        self._model = None
        self._processor = None
        self.loaded = False
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def parse(
        self,
        input_path: Path,
        output_dir: Path,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        from .ocr_service import normalize_markdown_math

        started = time.perf_counter()
        _raise_if_cancelled(cancel_event)
        self.ensure_loaded()
        images = load_pages(input_path)
        if not images:
            raise RuntimeError('No pages found in the uploaded file')

        page_json: list[dict[str, Any]] = []
        markdown_parts: list[str] = []
        for index, image in enumerate(images, start=1):
            _raise_if_cancelled(cancel_event)
            page_dir = output_dir / f'page_{index:04d}'
            page_dir.mkdir(parents=True, exist_ok=True)
            page_image = page_dir / 'page.png'
            fit_image(image).save(page_image)
            markdown = self._read_page(page_image)
            markdown = prepare_markdown(markdown)
            (page_dir / 'page.md').write_text(markdown, encoding='utf-8')
            markdown_parts.append(markdown.strip())
            page_json.append({
                '_app_page_number': index,
                'parsing_res_list': [{
                    'block_label': 'document',
                    'block_content': markdown.strip(),
                }],
            })

        document_md = normalize_markdown_math('\n\n---\n\n'.join(part for part in markdown_parts if part))
        canonical = {
            'source_file': input_path.name,
            'pipeline_version': 'Xiaomi-OCR-0',
            'engine': 'xiaomi',
            'model': XIAOMI_MODEL,
            'page_count': len(page_json),
            'pages': page_json,
        }
        import json
        (output_dir / 'document.json').write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        (output_dir / 'document.md').write_text(document_md, encoding='utf-8')
        return {
            'document': canonical,
            'markdown': document_md,
            'inference_ms': int((time.perf_counter() - started) * 1000),
        }

    def _read_page(self, image_path: Path) -> str:
        import torch

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'url': str(image_path)},
                {'type': 'text', 'text': DOCUMENT_PROMPT},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self._model.device)
        try:
            with torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            smaller = fit_image(Image.open(image_path).convert('RGB'), max_side=1024)
            smaller.save(image_path)
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt',
            ).to(self._model.device)
            with torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        new_tokens = output[:, inputs['input_ids'].shape[1]:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip()


def load_pages(path: Path) -> list[Image.Image]:
    if path.suffix.lower() == '.pdf':
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        pages: list[Image.Image] = []
        try:
            for index in range(len(document)):
                bitmap = document[index].render(scale=2)
                pages.append(bitmap.to_pil().convert('RGB'))
        finally:
            document.close()
        return pages
    with Image.open(path) as image:
        return [image.convert('RGB')]


def fit_image(image: Image.Image, max_side: int = 1600) -> Image.Image:
    image = image.convert('RGB')
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    return resized


def prepare_markdown(text: str) -> str:
    cleaned = _THINK_BLOCK.sub('', text or '').strip()
    cleaned = otsl_to_html(cleaned)
    cleaned = _DISPLAY_LATEX.sub(lambda match: f'\n\n$$\n{match.group(1).strip()}\n$$\n\n', cleaned)
    cleaned = _INLINE_LATEX.sub(lambda match: f'${match.group(1).strip()}$', cleaned)
    return cleaned.strip() + '\n'


def otsl_to_html(text: str) -> str:
    if not _OTSL_TOKEN.search(text or ''):
        return text

    def convert(match: re.Match[str]) -> str:
        table = _otsl_table(match.group(0))
        return table or match.group(0)

    pattern = re.compile(r'(?:<(?:fcel|ecel|lcel|ucel|nl)>[\s\S]*?)+(?=\n\s*\n|$)', re.IGNORECASE)
    return pattern.sub(convert, text)


def _otsl_table(block: str) -> str:
    tokens = list(_OTSL_TOKEN.finditer(block))
    if not tokens:
        return ''
    rows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def close_row() -> None:
        nonlocal current
        if current:
            rows.append(current)
        current = []

    for index, token in enumerate(tokens):
        end = tokens[index + 1].start() if index + 1 < len(tokens) else len(block)
        content = block[token.end():end].strip()
        tag = token.group(1).lower()
        if tag == 'nl':
            close_row()
            continue
        if tag == 'lcel' and current:
            current[-1]['colspan'] += 1
            if content:
                current[-1]['text'] = f"{current[-1]['text']} {content}".strip()
            continue
        if tag == 'ucel' and rows:
            above = _cell_above(rows, len(current))
            if above is not None:
                above['rowspan'] += 1
                continue
        current.append({'text': '' if tag == 'ecel' else content, 'colspan': 1, 'rowspan': 1})
    close_row()
    if not rows:
        return ''

    body = []
    for row in rows:
        cells = ''.join(
            f'<td colspan="{cell["colspan"]}" rowspan="{cell["rowspan"]}">{html.escape(cell["text"])}</td>'
            if cell['colspan'] > 1 or cell['rowspan'] > 1
            else f'<td>{html.escape(cell["text"])}</td>'
            for cell in row
        )
        body.append(f'<tr>{cells}</tr>')
    return '<table>' + ''.join(body) + '</table>'


def _cell_above(rows: list[list[dict[str, Any]]], column: int) -> dict[str, Any] | None:
    if not rows:
        return None
    cursor = 0
    for cell in rows[-1]:
        if cursor <= column < cursor + cell['colspan']:
            return cell
        cursor += cell['colspan']
    return None


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        from .ocr_service import CancelledInference

        raise CancelledInference('Cancelled')


xiaomi_engine = XiaomiEngine()
