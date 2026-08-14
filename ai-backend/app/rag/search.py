"""PDF-backed hybrid RAG search for TB and pneumonia guidelines.

The index is intentionally local-first: it extracts the two guideline PDFs
already stored in the repo, chunks them, caches the result, and ranks with a
small hybrid of BM25-style keyword scoring plus TF-IDF cosine similarity.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.rag.corpus import SEED_DOCUMENTS

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - import fallback for older installs
    PdfReader = None  # type: ignore[assignment]

try:
    from PyPDF2 import PdfReader as LegacyPdfReader
except Exception:  # pragma: no cover - optional fallback
    LegacyPdfReader = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - keyword-only fallback
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_SOURCES = [
    PROJECT_ROOT / "National-Tuberculosis-Management-Guidelines-2019_Nepal.pdf",
    PROJECT_ROOT / "TB_Pneumonia_Extract_NNF2018 (1).pdf",
]
CACHE_DIR = PROJECT_ROOT / "ai-backend" / ".rag-cache"
CACHE_PATH = CACHE_DIR / "guideline_chunks.json"

MEDICAL_SCOPE_TERMS = {
    "afb",
    "cavitation",
    "consolidation",
    "culture",
    "dst",
    "extrapulmonary",
    "genexpert",
    "hemoptysis",
    "isoniazid",
    "mtb",
    "pneumonia",
    "pulmonary",
    "rifampicin",
    "sputum",
    "tb",
    "tuberculosis",
    "xdr",
}


@dataclass(frozen=True)
class GuidelineChunk:
    id: str
    content: str
    source: str
    metadata: dict[str, Any]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9/-]*", text.lower())
        if len(token) > 1
    ]


def _semantic_chunks(text: str, *, max_words: int = 220, overlap: int = 45) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 120:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader_cls = PdfReader or LegacyPdfReader
    if reader_cls is None:
        raise RuntimeError(
            "PDF extraction requires pypdf. Install backend requirements first."
        )

    reader = reader_cls(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = _clean_text(text)
        if cleaned:
            pages.append((index, cleaned))
    return pages


def _pdf_signature(paths: list[Path]) -> dict[str, dict[str, Any]]:
    signature: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            signature[path.name] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
    return signature


def _build_pdf_chunks() -> list[GuidelineChunk]:
    chunks: list[GuidelineChunk] = []
    for path in PDF_SOURCES:
        if not path.exists():
            continue

        source_label = path.name
        for page_number, page_text in _extract_pdf_pages(path):
            for chunk_index, content in enumerate(_semantic_chunks(page_text), start=1):
                title_match = re.search(
                    r"(chapter\s+\d+[:.\s-]+[^.]{4,80}|section\s+\d+[:.\s-]+[^.]{4,80})",
                    content,
                    flags=re.IGNORECASE,
                )
                chunks.append(
                    GuidelineChunk(
                        id=f"{path.stem}:p{page_number}:c{chunk_index}",
                        content=content,
                        source=f"{source_label}#page={page_number}",
                        metadata={
                            "document": source_label,
                            "page": page_number,
                            "title_hint": title_match.group(1).strip()
                            if title_match
                            else "",
                        },
                    )
                )
    return chunks


def _seed_chunks() -> list[GuidelineChunk]:
    return [
        GuidelineChunk(
            id=item["id"],
            content=item["content"],
            source=item["source"],
            metadata={"seed": True},
        )
        for item in SEED_DOCUMENTS
    ]


def _load_or_create_chunks() -> list[GuidelineChunk]:
    signature = _pdf_signature(PDF_SOURCES)
    if CACHE_PATH.exists():
        try:
            payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if payload.get("signature") == signature:
                return [
                    GuidelineChunk(
                        id=item["id"],
                        content=item["content"],
                        source=item["source"],
                        metadata=item.get("metadata", {}),
                    )
                    for item in payload.get("chunks", [])
                ]
        except Exception:
            pass

    chunks = _build_pdf_chunks()
    if not chunks:
        chunks = _seed_chunks()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "signature": signature,
                "chunks": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return chunks


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> list[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    avg_len = sum(len(tokens) for tokens in docs_tokens) / max(len(docs_tokens), 1)
    doc_freq: dict[str, int] = {}
    for tokens in docs_tokens:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    total_docs = len(docs_tokens)

    for tokens in docs_tokens:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        score = 0.0
        doc_len = len(tokens) or 1
        for token in query_tokens:
            freq = counts.get(token, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq.get(token, 0) + 0.5) / (doc_freq.get(token, 0) + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            score += idf * (freq * (k1 + 1) / denom)
        scores.append(score)

    return scores


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    high = max(values)
    low = min(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@lru_cache(maxsize=1)
def _index() -> tuple[list[GuidelineChunk], list[list[str]], Any]:
    chunks = _load_or_create_chunks()
    tokenized = [_tokenize(f"{chunk.source} {chunk.content}") for chunk in chunks]
    vectorizer = None
    matrix = None
    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=12000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = vectorizer.fit_transform([chunk.content for chunk in chunks])
    return chunks, tokenized, (vectorizer, matrix)


def is_medical_scope(query: str) -> bool:
    tokens = set(_tokenize(query))
    return bool(tokens & MEDICAL_SCOPE_TERMS)


def search_knowledge(query: str, top_k: int = 4) -> list[dict]:
    chunks, tokenized, vector_data = _index()
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    bm25 = _normalize(_bm25_scores(query_tokens, tokenized))

    vectorizer, matrix = vector_data
    tfidf_scores = [0.0 for _ in chunks]
    if vectorizer is not None and matrix is not None and cosine_similarity is not None:
        query_vector = vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, matrix).ravel().tolist()
        tfidf_scores = _normalize([float(score) for score in tfidf_scores])

    scored: list[tuple[float, GuidelineChunk]] = []
    for index, chunk in enumerate(chunks):
        medical_boost = 0.08 if is_medical_scope(chunk.content) else 0.0
        score = (0.58 * bm25[index]) + (0.42 * tfidf_scores[index]) + medical_boost
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        scored = [(0.05, chunk) for chunk in chunks[:top_k]]

    return [
        {
            "id": chunk.id,
            "content": chunk.content,
            "score": round(float(score), 4),
            "source": chunk.source,
            "metadata": chunk.metadata,
        }
        for score, chunk in scored[:top_k]
    ]


def get_index_status() -> dict[str, Any]:
    chunks, _, _ = _index()
    documents = sorted({chunk.metadata.get("document", chunk.source) for chunk in chunks})
    return {
        "cache_path": str(CACHE_PATH),
        "chunk_count": len(chunks),
        "documents": documents,
        "pdf_sources": [str(path) for path in PDF_SOURCES],
    }
