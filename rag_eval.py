from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: frozenset[str]
    retrieved_ids: tuple[str, ...]

def _k(k: int) -> None:
    if not isinstance(k, int) or not 1 <= k <= 1000:
        raise ValueError("k must be 1..1000")

def _unique_top(case: RetrievalCase, k: int) -> tuple[str, ...]:
    _k(k)
    if not isinstance(case, RetrievalCase):
        raise TypeError("case must be RetrievalCase")
    seen: set[str] = set()
    top: list[str] = []
    for item in case.retrieved_ids:
        if item in seen:
            continue
        seen.add(item)
        top.append(item)
        if len(top) == k:
            break
    return tuple(top)

def precision_at_k(c: RetrievalCase, k: int) -> float:
    top = _unique_top(c, k)
    return 0.0 if not top else sum(x in c.relevant_ids for x in top) / len(top)

def recall_at_k(c: RetrievalCase, k: int) -> float:
    top = _unique_top(c, k)
    return 0.0 if not c.relevant_ids else len(set(top) & c.relevant_ids) / len(c.relevant_ids)

def reciprocal_rank(c: RetrievalCase, k: int = 100) -> float:
    for i, item in enumerate(_unique_top(c, k), 1):
        if item in c.relevant_ids:
            return 1 / i
    return 0.0

def ndcg_at_k(c: RetrievalCase, k: int) -> float:
    top = _unique_top(c, k)
    dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(top) if item in c.relevant_ids)
    ideal_count = min(len(c.relevant_ids), k)
    if not ideal_count:
        return 0.0
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg

def lexical_groundedness(answer: str, contexts: Iterable[str]) -> float:
    if not isinstance(answer, str):
        raise TypeError("answer must be text")
    if isinstance(contexts, (str, bytes)):
        raise TypeError("contexts must be an iterable of strings")
    context_rows = tuple(contexts)
    if any(not isinstance(row, str) for row in context_rows):
        raise TypeError("contexts must contain strings")
    answer_tokens = {t.lower().strip(".,!?;:()[]{}\"'") for t in answer.split() if len(t) > 2}
    context_tokens = {
        t.lower().strip(".,!?;:()[]{}\"'")
        for row in context_rows
        for t in row.split()
        if len(t) > 2
    }
    answer_tokens.discard("")
    context_tokens.discard("")
    return 0.0 if not answer_tokens else len(answer_tokens & context_tokens) / len(answer_tokens)

def evaluate(cases: Iterable[RetrievalCase], k: int) -> dict[str, float]:
    case_rows = tuple(cases)
    if not case_rows:
        raise ValueError("cases required")
    if len(case_rows) > 10_000:
        raise ValueError("too many cases")
    rows = [
        [precision_at_k(c, k), recall_at_k(c, k), reciprocal_rank(c, k), ndcg_at_k(c, k)]
        for c in case_rows
    ]
    names = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]
    return {name: sum(row[i] for row in rows) / len(rows) for i, name in enumerate(names)}
