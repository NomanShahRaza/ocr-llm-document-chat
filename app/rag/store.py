from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RAG_DIR


class RagStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or RAG_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.root / 'chunks.jsonl'
        self.embeddings_path = self.root / 'embeddings.npy'
        self.meta_path = self.root / 'meta.json'
        self._lock = threading.RLock()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray = np.zeros((0, 1024), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        chunks: list[dict[str, Any]] = []
        if self.chunks_path.exists():
            for line in self.chunks_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        embeddings = np.zeros((0, 1024), dtype=np.float32)
        if self.embeddings_path.exists():
            try:
                embeddings = np.load(self.embeddings_path).astype(np.float32)
            except Exception:
                embeddings = np.zeros((0, 1024), dtype=np.float32)
        if len(chunks) != len(embeddings):
            # Corrupt or partial index — start clean and let service rebuild.
            chunks = []
            embeddings = np.zeros((0, 1024), dtype=np.float32)
        self.chunks = chunks
        self.embeddings = embeddings

    def persist(self) -> None:
        with self._lock:
            tmp_chunks = self.chunks_path.with_suffix('.tmp')
            with tmp_chunks.open('w', encoding='utf-8') as handle:
                for chunk in self.chunks:
                    handle.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            tmp_chunks.replace(self.chunks_path)
            if len(self.embeddings):
                np.save(self.embeddings_path, self.embeddings)
            elif self.embeddings_path.exists():
                self.embeddings_path.unlink()
            meta = {
                'chunk_count': len(self.chunks),
                'job_ids': sorted({chunk['job_id'] for chunk in self.chunks}),
            }
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    def job_ids(self) -> set[str]:
        return {chunk['job_id'] for chunk in self.chunks}

    def remove_job(self, job_id: str) -> int:
        with self._lock:
            keep = [idx for idx, chunk in enumerate(self.chunks) if chunk['job_id'] != job_id]
            removed = len(self.chunks) - len(keep)
            if not removed:
                return 0
            self.chunks = [self.chunks[idx] for idx in keep]
            if len(self.embeddings):
                self.embeddings = self.embeddings[keep] if keep else np.zeros((0, self.embeddings.shape[1]), dtype=np.float32)
            self.persist()
            return removed

    def replace_job(self, job_id: str, chunks: list[dict[str, Any]], vectors: np.ndarray) -> None:
        with self._lock:
            keep = [idx for idx, chunk in enumerate(self.chunks) if chunk['job_id'] != job_id]
            self.chunks = [self.chunks[idx] for idx in keep] + chunks
            if keep and len(self.embeddings):
                base = self.embeddings[keep]
            else:
                dim = vectors.shape[1] if len(vectors) else (self.embeddings.shape[1] if len(self.embeddings) else 1024)
                base = np.zeros((0, dim), dtype=np.float32)
            if len(vectors):
                self.embeddings = np.vstack([base, vectors.astype(np.float32)]) if len(base) else vectors.astype(np.float32)
            else:
                self.embeddings = base
            self.persist()

    def clear(self) -> None:
        with self._lock:
            self.chunks = []
            self.embeddings = np.zeros((0, 1024), dtype=np.float32)
            self.persist()
