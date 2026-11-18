from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .bm25 import tokenize

STOPWORDS = {
    'a', 'about', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'between', 'by', 'can', 'compare',
    'comparison', 'describe', 'did', 'difference', 'differences', 'do', 'doc', 'document', 'documents',
    'does', 'explain', 'file', 'for', 'from', 'give', 'how', 'in', 'is', 'it', 'its', 'list', 'listed',
    'me', 'mention', 'mentioned', 'of', 'on', 'or', 'paper', 'pdf', 'presentation', 'slide', 'slides',
    'summarize', 'summary', 'tell', 'that', 'the', 'these', 'this', 'those', 'to', 'used', 'versus', 'vs',
    'was', 'were', 'what', 'when', 'where', 'which', 'who', 'why', 'with',
}

_SUMMARY = re.compile(
    r"\b(what('?s| is| are) (this|the|these) (file|document|doc|pdf|presentation|paper|slides?|deck)s?( (all|mainly))? about"
    r"|tell me (about|more about) (this|the|these) (file|document|doc|pdf|presentation|paper|slides?|deck)s?"
    r"|about (this|the|these) (file|document|doc|pdf|presentation|paper|slides?|deck)s?"
    r"|summar(y|ise|ize)|main (topic|idea|point|subject)s?|overview|gist"
    r"|purpose of (this|the)|what does (this|the) \w+ (discuss|cover|explain|describe|present))",
    re.IGNORECASE,
)
_COMPARISON = re.compile(r'\b(compare|comparison|contrast|differ(ence|ences|ent)?|vs\.?|versus)\b', re.IGNORECASE)
_LIST = re.compile(
    r'\b(list|enumerate|name all|give me all|all (the )?\w+ (mentioned|listed|used)|which \w+ (are|were) (mentioned|listed|used)'
    r'|what \w+ (are|were) (mentioned|listed|used))\b',
    re.IGNORECASE,
)
_EXPLAIN = re.compile(r'^\s*(explain|why|how|describe)\b', re.IGNORECASE)
_SPLIT = re.compile(r'(?<=[.!?。．])\s+|\s*[•■▪●・]\s*|\n+')
_LATEX = re.compile(r'\$\$.*?\$\$|\$[^$\n]*\$|\\[A-Za-z]+(\{[^}]*\})?', re.DOTALL)
_CAPTION = re.compile(r'^(\[.*\]|fig(ure)?\b|table\b)', re.IGNORECASE)
_BYLINE = re.compile(r'\$|@|\b(university|institute|lab|zurich|inc\.?)\b|https?://', re.IGNORECASE)
_GENERIC_HEADINGS = {'abstract', 'introduction', 'references', 'acknowledgements', 'acknowledgments', 'contents', 'appendix'}


def classify_question(query: str) -> str:
    if _SUMMARY.search(query):
        return 'summary'
    if _COMPARISON.search(query):
        return 'comparison'
    if _LIST.search(query):
        return 'list'
    if _EXPLAIN.search(query):
        return 'explanation'
    return 'factual'


def comparison_subjects(query: str) -> list[str]:
    text = query.strip().rstrip('?.! ')
    patterns = [
        r'difference(?:s)? between (.+?) and (.+)',
        r'compare (.+?) (?:and|with|to|vs\.?|versus) (.+)',
        r'how (?:does|do|is|are) (.+?) differ(?:ent)? from (.+)',
        r'(.+?) (?:vs\.?|versus) (.+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            subjects = [part.strip(' "\'') for part in match.groups()]
            if all(subjects):
                return subjects
    return []


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith('s') and not token.endswith('ss') else token


def _terms(text: str) -> set[str]:
    return {_stem(token) for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


def _strip_latex(text: str) -> str:
    return _LATEX.sub(' ', text or '')


def _is_noise(text: str) -> bool:
    digits = sum(ch.isdigit() for ch in text)
    return digits > 0.3 * max(len(text), 1)


def split_sentences(text: str, min_len: int = 25, max_len: int = 360) -> list[str]:
    out: list[str] = []
    for part in _SPLIT.split(_strip_latex(text)):
        cleaned = re.sub(r'\s+', ' ', part).strip(' -\t{}')
        if min_len <= len(cleaned) <= max_len and not _is_noise(cleaned):
            out.append(cleaned)
    return out


def _cite(hits: list[dict[str, Any]]) -> str:
    pages: dict[str, list[int]] = {}
    for hit in hits:
        bucket = pages.setdefault(hit['filename'], [])
        if hit['page'] not in bucket:
            bucket.append(hit['page'])
    parts = [f"{name} p.{', p.'.join(str(page) for page in sorted(nums))}" for name, nums in pages.items()]
    return f"[{'; '.join(parts)}]" if parts else ''


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return ' and '.join(items)
    return ', '.join(items[:-1]) + ', and ' + items[-1]


def _ranked_sentences(query: str, hits: list[dict[str, Any]]) -> list[tuple[float, str, dict[str, Any]]]:
    query_terms = _terms(query)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for rank, hit in enumerate(hits):
        for sentence in split_sentences(hit.get('text') or ''):
            terms = _terms(sentence)
            overlap = len(query_terms & terms)
            if not overlap:
                continue
            scored.append((overlap * 2.0 + overlap / max(len(terms), 1) - 0.05 * rank, sentence, hit))
    scored.sort(key=lambda item: -item[0])
    return scored


def _pick(scored: list[tuple[float, str, dict[str, Any]]], count: int) -> list[tuple[str, dict[str, Any]]]:
    picked: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for _, sentence, hit in scored:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append((sentence, hit))
        if len(picked) >= count:
            break
    return picked


def _recurring_terms(texts: list[str], exclude: set[str], count: int = 6) -> list[str]:
    counts: Counter[str] = Counter()
    casing: dict[str, Counter[str]] = {}
    for text in texts:
        for word in re.findall(r'[A-Za-z][A-Za-z0-9\-]{3,}', _strip_latex(text)):
            key = word.lower()
            if key in STOPWORDS or key in exclude:
                continue
            counts[key] += 1
            casing.setdefault(key, Counter())[word] += 1
    return [casing[key].most_common(1)[0][0] for key, freq in counts.most_common(count * 2) if freq >= 2][:count]


def _summary_answer(hits: list[dict[str, Any]], outlines: list[dict[str, Any]]) -> str:
    if not outlines or not outlines[0].get('title'):
        return (
            "The retrieved content is not sufficient to determine the document's overall subject reliably. "
            f'The closest matching passages are on {_cite(hits[:3])}.'
        )
    outline = outlines[0]
    title = outline['title'].rstrip(' .:;')
    headings: list[str] = []
    heading_pages: list[int] = []
    skipped: set[str] = set()
    for page in outline['pages']:
        for heading in page['headings']:
            if (
                heading.lower() == title.lower()
                or len(heading) < 4
                or len(heading) > 90
                or _is_noise(heading)
                or _CAPTION.match(heading)
                or _BYLINE.search(heading)
                or heading.strip().lower() in _GENERIC_HEADINGS
            ):
                skipped |= _terms(heading)
                continue
            if heading.lower() not in {h.lower() for h in headings}:
                headings.append(heading.rstrip(' .:;'))
                heading_pages.append(page['page'])
    sentences = [f'This document is about {title}.']
    if headings:
        sentences.append(f'Its main sections cover {_join(headings[:6])}.')
    texts = [hit['text'] for hit in hits] + [page['preview'] for page in outline['pages']]
    exclude = _terms(title) | skipped | {term for heading in headings[:6] for term in _terms(heading)}
    themes = _recurring_terms(texts, exclude)
    if themes:
        sentences.append(f'Recurring themes include {_join(themes)}.')
    pages = sorted({outline['pages'][0]['page'], *heading_pages[:6]}) if outline['pages'] else []
    cite = f"[{outline['filename']} p.{', p.'.join(str(page) for page in pages)}]" if pages else ''
    return ' '.join(sentences) + (f' {cite}' if cite else '')


def _comparison_answer(query: str, hits: list[dict[str, Any]], subjects: list[str]) -> str:
    if len(subjects) < 2:
        return _prose_answer(query, hits, 4)
    parts: list[str] = []
    missing: list[str] = []
    for subject in subjects:
        picked = _pick(_ranked_sentences(subject, hits), 2)
        if not picked:
            missing.append(subject)
            continue
        text = ' '.join(sentence for sentence, _ in picked)
        parts.append(f'{subject}: {text} {_cite([hit for _, hit in picked])}')
    if missing:
        parts.append(
            f"The documents contain no clear evidence about {_join(missing)}, so a full comparison is not possible."
        )
    return '\n\n'.join(parts)


def _list_answer(query: str, hits: list[dict[str, Any]]) -> str:
    query_terms = _terms(query)
    relevant = [hit for hit in hits if query_terms & _terms(hit['text'])] or hits[:2]
    items: list[str] = []
    used: list[dict[str, Any]] = []
    for hit in relevant:
        for fragment in split_sentences(hit['text'], min_len=3, max_len=60):
            if _CAPTION.match(fragment):
                continue
            if fragment.lower() not in {item.lower() for item in items}:
                items.append(fragment.rstrip(' .;:'))
                if hit not in used:
                    used.append(hit)
            if len(items) >= 12:
                break
        if len(items) >= 12:
            break
    if not items:
        return 'The retrieved content does not contain a list that answers this question.'
    return f'The document lists {_join(items)}. {_cite(used)}'


def _prose_answer(query: str, hits: list[dict[str, Any]], count: int) -> str:
    picked = _pick(_ranked_sentences(query, hits), count)
    if not picked:
        return (
            'The retrieved content does not directly answer this question. '
            f'The closest matching passages are on {_cite(hits[:3])}.'
        )
    picked.sort(key=lambda item: (item[1]['filename'], item[1]['page']))
    return f"{' '.join(sentence for sentence, _ in picked)} {_cite([hit for _, hit in picked])}"


def fallback_answer(
    query: str,
    qtype: str,
    hits: list[dict[str, Any]],
    outlines: list[dict[str, Any]],
    subjects: list[str],
) -> str:
    if qtype == 'summary':
        return _summary_answer(hits, outlines)
    if qtype == 'comparison':
        return _comparison_answer(query, hits, subjects)
    if qtype == 'list':
        return _list_answer(query, hits)
    return _prose_answer(query, hits, 4 if qtype == 'explanation' else 3)


def build_context(
    hits: list[dict[str, Any]],
    outlines: list[dict[str, Any]],
    full_outline: bool,
    outline_budget: int = 6000,
    chunk_chars: int = 1200,
) -> str:
    lines: list[str] = []
    if outlines:
        lines.append('DOCUMENT OVERVIEW (title, page headings, page previews)')
        used = 0
        for outline in outlines:
            lines.append(f"Document: {outline['filename']} ({outline['page_count']} pages)")
            if outline.get('title'):
                lines.append(f"Title: {outline['title']}")
            for page in outline['pages']:
                if not full_outline and not page['headings']:
                    continue
                entry = f"- p.{page['page']}: {'; '.join(page['headings']) or '(no heading)'}"
                if full_outline and page['preview']:
                    entry += f" | {page['preview']}"
                used += len(entry)
                if used > (outline_budget if full_outline else outline_budget // 4):
                    lines.append('- (remaining pages omitted)')
                    break
                lines.append(entry)
            lines.append('')
    lines.append('RETRIEVED EVIDENCE (OCR passages; may be noisy or out of order)')
    for idx, hit in enumerate(hits, start=1):
        text = hit['text'] if len(hit['text']) <= chunk_chars else hit['text'][:chunk_chars] + '…'
        lines.append(f"[{idx}] ({hit['filename']} p.{hit['page']}) {text}")
    return '\n'.join(lines)
