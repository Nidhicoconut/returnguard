"""Policy RAG: split the policy document into clauses and retrieve with TF-IDF cosine similarity.

TF-IDF is deliberate: the corpus is a handful of clauses, so a dense embedding model would add
a download and latency for no gain. Swap `retrieve` for an embedding search if the policy grows.
"""
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


@lru_cache(maxsize=1)
def _index():
    text = (config.POLICY_DIR / "return_policy.md").read_text()
    clauses = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        cid, _, title = head.partition(" ")
        clauses.append({"id": cid.strip(), "title": title.strip(), "text": body.strip()})
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform([f"{c['title']}. {c['text']}" for c in clauses])
    return clauses, vec, matrix


def all_clauses():
    return list(_index()[0])


def clause_ids():
    return {c["id"] for c in _index()[0]}


def retrieve(query: str, k: int = 3, must_include=()):
    clauses, vec, matrix = _index()
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    ranked = sorted(range(len(clauses)), key=lambda i: scores[i], reverse=True)
    picked = [i for i in ranked[:k] if scores[i] > 0]
    for cid in must_include:
        idx = next((i for i, c in enumerate(clauses) if c["id"] == cid), None)
        if idx is not None and idx not in picked:
            picked.append(idx)
    return [{**clauses[i], "score": round(float(scores[i]), 3)} for i in picked]
