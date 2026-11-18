from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text or '')]


class BM25Okapi:
    """Classic Okapi BM25 for exact / lexical term matching."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0
        self.doc_freqs: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        for doc in corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs:
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def get_scores(self, query: Sequence[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not query or not self.corpus_size:
            return scores
        for term in query:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for idx, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / max(self.avgdl, 1e-9))
                scores[idx] += idf * (freq * (self.k1 + 1) / denom)
        return scores
