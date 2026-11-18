from __future__ import annotations

import logging
import re
import threading
from typing import Any

import numpy as np

from ..config import (
    LLM_CONTEXT_HITS,
    LLM_HISTORY_TURNS,
    RAG_AUTO_INDEX,
    RAG_CHUNK_CHARS,
    RAG_CHUNK_OVERLAP,
    RAG_DEVICE,
    RAG_RRF_K,
    RAG_TOP_K,
    RESULT_DIR,
    BGE_M3_MODEL,
)
from .answering import build_context, classify_question, comparison_subjects, fallback_answer
from .bm25 import BM25Okapi, tokenize
from .chunking import document_outline, load_job_corpus
from .embeddings import encoder
from .hybrid import reciprocal_rank_fusion
from .llm import answer_llm
from .store import RagStore

logger = logging.getLogger(__name__)

_CITATION = re.compile(r'\s*[\[(][^\[\]()]*\bp\.\s*\d+[^\[\]()]*[\])]')


def _strip_citations(text: str) -> str:
    text = _CITATION.sub('', text)
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'([.,;:])\1+', r'\1', text)
    return re.sub(r'[ \t]{2,}', ' ', text).strip()
_FOLLOW_UP = re.compile(
    r'^(and|also|what about|how about|why|more|explain|elaborate|tell me more|continue)\b'
    r'|\b(it|its|this|that|these|those|they|them|their|he|she|above|previous)\b',
    re.IGNORECASE,
)


def _retrieval_query(query: str, history: list[dict[str, str]]) -> str:
    # Follow-ups like "explain that more" carry no searchable terms; borrow the last user question.
    if not history or len(query.split()) > 14 or not _FOLLOW_UP.search(query):
        return query
    previous = next((turn['content'] for turn in reversed(history) if turn['role'] == 'user'), '')
    return f'{previous} {query}'.strip() if previous else query


class RagService:
    def __init__(self) -> None:
        self.store = RagStore()
        self._lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        self._indexing: set[str] = set()
        self._last_error: str | None = None
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._tokenized = [tokenize(chunk.get('text') or '') for chunk in self.store.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def health(self) -> dict[str, Any]:
        return {
            'status': 'ok',
            'model': BGE_M3_MODEL,
            'device': encoder.device if encoder.loaded else RAG_DEVICE,
            'encoder_loaded': encoder.loaded,
            'chunk_count': len(self.store.chunks),
            'document_count': len(self.store.job_ids()),
            'indexing': sorted(self._indexing),
            'last_error': self._last_error,
            'modes': ['hybrid', 'semantic', 'bm25'],
            'auto_index': RAG_AUTO_INDEX,
            'answer_model': answer_llm.model or None,
        }

    def index_job(self, job_id: str, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if job_id in self._indexing:
                return {'job_id': job_id, 'status': 'indexing', 'chunk_count': 0}
            if not force and job_id in self.store.job_ids():
                count = sum(1 for chunk in self.store.chunks if chunk['job_id'] == job_id)
                return {'job_id': job_id, 'status': 'ready', 'chunk_count': count}
            self._indexing.add(job_id)
            self._last_error = None
        try:
            chunks = load_job_corpus(job_id, RESULT_DIR, RAG_CHUNK_CHARS, RAG_CHUNK_OVERLAP)
            if not chunks:
                self.store.remove_job(job_id)
                with self._lock:
                    self._rebuild_bm25()
                return {'job_id': job_id, 'status': 'empty', 'chunk_count': 0}
            logger.info('RAG indexing %s (%s chunks)', job_id, len(chunks))
            vectors = encoder.encode([chunk['text'] for chunk in chunks])
            with self._lock:
                self.store.replace_job(job_id, chunks, vectors)
                self._rebuild_bm25()
            logger.info('RAG indexed %s', job_id)
            return {'job_id': job_id, 'status': 'indexed', 'chunk_count': len(chunks)}
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception('RAG index failed for %s', job_id)
            raise
        finally:
            with self._lock:
                self._indexing.discard(job_id)

    def index_job_async(self, job_id: str, force: bool = False) -> None:
        if not RAG_AUTO_INDEX and not force:
            return

        def _run() -> None:
            try:
                self.index_job(job_id, force=force)
            except Exception:
                # Logged inside index_job; keep worker thread alive for the API.
                pass

        thread = threading.Thread(
            target=_run,
            name=f'rag-index-{job_id[:8]}',
            daemon=True,
        )
        thread.start()

    def remove_job(self, job_id: str) -> None:
        with self._lock:
            self.store.remove_job(job_id)
            self._rebuild_bm25()

    def clear(self) -> None:
        with self._lock:
            self.store.clear()
            self._rebuild_bm25()

    def reindex_all(self, job_ids: list[str] | None = None) -> dict[str, Any]:
        ids = job_ids or sorted(
            path.name for path in RESULT_DIR.iterdir()
            if path.is_dir() and ((path / 'document.md').exists() or (path / 'document.json').exists())
        )
        results = [self.index_job(job_id, force=True) for job_id in ids]
        return {
            'indexed': len([item for item in results if item['chunk_count'] > 0]),
            'jobs': results,
            'chunk_count': len(self.store.chunks),
        }

    def search(
        self,
        query: str,
        mode: str = 'hybrid',
        top_k: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        query = (query or '').strip()
        mode = (mode or 'hybrid').lower()
        if mode not in {'hybrid', 'semantic', 'bm25'}:
            raise ValueError('mode must be hybrid, semantic, or bm25')
        limit = max(1, min(int(top_k or RAG_TOP_K), 50))
        if not query:
            return {'query': query, 'mode': mode, 'results': []}
        if not self.store.chunks:
            return {'query': query, 'mode': mode, 'results': [], 'warning': 'No indexed documents yet'}

        allowed = {
            idx for idx, chunk in enumerate(self.store.chunks)
            if job_id is None or chunk['job_id'] == job_id
        }
        if not allowed:
            return {'query': query, 'mode': mode, 'results': [], 'warning': 'No chunks for this document'}

        semantic_ranks: list[int] = []
        bm25_ranks: list[int] = []
        semantic_scores: dict[int, float] = {}
        bm25_scores: dict[int, float] = {}

        if mode in {'hybrid', 'semantic'}:
            query_vec = encoder.encode([query])[0]
            matrix = self.store.embeddings
            if len(matrix):
                sims = matrix @ query_vec
                order = np.argsort(-sims)
                for idx in order:
                    i = int(idx)
                    if i not in allowed:
                        continue
                    semantic_ranks.append(i)
                    semantic_scores[i] = float(sims[i])

        if mode in {'hybrid', 'bm25'}:
            tokens = tokenize(query)
            scores = self._bm25.get_scores(tokens) if self._bm25 is not None else []
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            for i in order:
                if i not in allowed or scores[i] <= 0:
                    continue
                bm25_ranks.append(i)
                bm25_scores[i] = float(scores[i])

        if mode == 'semantic':
            fused = [(idx, semantic_scores.get(idx, 0.0)) for idx in semantic_ranks[:limit]]
        elif mode == 'bm25':
            fused = [(idx, bm25_scores.get(idx, 0.0)) for idx in bm25_ranks[:limit]]
        else:
            fused = reciprocal_rank_fusion(
                [semantic_ranks[: max(limit * 4, 20)], bm25_ranks[: max(limit * 4, 20)]],
                k=RAG_RRF_K,
            )[:limit]

        results = []
        for idx, score in fused:
            chunk = self.store.chunks[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'job_id': chunk['job_id'],
                'filename': chunk['filename'],
                'page': chunk['page'],
                'text': chunk['text'],
                'score': round(float(score), 6),
                'semantic_score': round(semantic_scores.get(idx, 0.0), 6) if idx in semantic_scores else None,
                'bm25_score': round(bm25_scores.get(idx, 0.0), 6) if idx in bm25_scores else None,
            })
        return {'query': query, 'mode': mode, 'results': results}

    def ask(
        self,
        query: str,
        mode: str = 'hybrid',
        top_k: int | None = None,
        job_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        history = [turn for turn in (history or []) if turn.get('content')][-LLM_HISTORY_TURNS:]
        qtype = classify_question(query)
        limit = 16 if qtype == 'summary' else max(top_k or RAG_TOP_K, 12)
        payload = self.search(_retrieval_query(query, history), mode=mode, top_k=limit, job_id=job_id)
        payload['query'] = query
        hits = payload.get('results') or []

        subjects = comparison_subjects(query) if qtype == 'comparison' else []
        if subjects:
            # Retrieve evidence for each side so the answer cannot lean on one subject only.
            merged: dict[str, dict[str, Any]] = {}
            per_side = [self.search(subject, mode=mode, top_k=6, job_id=job_id).get('results') or [] for subject in subjects]
            for rank in range(6):
                for side in per_side:
                    if rank < len(side):
                        merged.setdefault(side[rank]['chunk_id'], side[rank])
            for hit in hits:
                merged.setdefault(hit['chunk_id'], hit)
            hits = list(merged.values())[:limit]

        payload['question_type'] = qtype
        if not hits:
            if job_id and job_id not in self.store.job_ids():
                self.index_job_async(job_id, force=True)
                payload['answer'] = (
                    'This document is not in the search index yet. Indexing has started — '
                    'search again in a moment.'
                )
                payload['warning'] = 'not_indexed'
            else:
                payload['answer'] = 'The indexed documents do not contain content relevant to this question.'
            payload['answer_source'] = 'none'
            return payload

        doc_ids = [job_id] if job_id else list(dict.fromkeys(hit['job_id'] for hit in hits))[:3]
        outlines = [outline for outline in (document_outline(doc, RESULT_DIR) for doc in doc_ids) if outline]

        answer = None
        if answer_llm.enabled:
            try:
                context = build_context(
                    hits[:LLM_CONTEXT_HITS],
                    outlines,
                    full_outline=qtype == 'summary',
                    outline_budget=3500,
                    chunk_chars=900,
                )
                answer = answer_llm.answer(query, context, qtype, history, citation_file=hits[0]['filename'])
                payload['answer_source'] = answer_llm.model
            except Exception as exc:
                logger.exception('LLM answer failed; using extractive fallback')
                payload['llm_error'] = str(exc)
        if answer is None:
            answer = fallback_answer(query, qtype, hits, outlines, subjects)
            payload['answer_source'] = 'extractive'

        payload['answer'] = _strip_citations(answer)
        payload['results'] = hits[:8]
        return payload


rag_service = RagService()
