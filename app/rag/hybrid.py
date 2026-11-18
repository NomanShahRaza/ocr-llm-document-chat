from __future__ import annotations

from typing import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge ranked result lists with Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for ranks in ranked_lists:
        for rank, doc_idx in enumerate(ranks):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
