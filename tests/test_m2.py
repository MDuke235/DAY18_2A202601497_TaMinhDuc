"""Tests for Module 2: Hybrid Search."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m2_search import (
    segment_vietnamese, BM25Search, HybridSearch,
    reciprocal_rank_fusion, SearchResult,
)

CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
    {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
]

def test_segment_returns_string():
    assert isinstance(segment_vietnamese("nghỉ phép năm"), str)

def test_bm25_search():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép", top_k=2)
    assert len(results) > 0 and results[0].method == "bm25"

def test_bm25_relevant_first():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép năm", top_k=2)
    if results:
        assert "nghỉ" in results[0].text.lower() or "12" in results[0].text

def test_rrf_merges():
    a = [SearchResult("doc1", 0.9, {}, "bm25"), SearchResult("doc2", 0.8, {}, "bm25")]
    b = [SearchResult("doc2", 0.95, {}, "dense"), SearchResult("doc3", 0.85, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(merged) > 0 and "doc2" in [r.text for r in merged]

def test_rrf_method():
    a = [SearchResult("d1", 0.9, {}, "bm25")]
    b = [SearchResult("d1", 0.8, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=1)
    if merged:
        assert merged[0].method == "hybrid"


def test_hybrid_search_falls_back_to_bm25_when_dense_is_unavailable():
    search = HybridSearch()
    search.bm25.index(CHUNKS)
    search.dense.search = lambda *args, **kwargs: (_ for _ in ()).throw(
        ConnectionError("Qdrant unavailable")
    )

    results = search.search("nghỉ phép", top_k=2)

    assert results
    assert results[0].method == "hybrid"
    assert results[0].metadata["rrf_methods"] == ["bm25"]


def test_hybrid_index_keeps_bm25_when_dense_index_is_unavailable():
    search = HybridSearch()
    search.dense.index = lambda *args, **kwargs: (_ for _ in ()).throw(
        ConnectionError("Qdrant unavailable")
    )

    search.index(CHUNKS)
    results = search.bm25.search("nghỉ phép", top_k=2)

    assert results
    assert results[0].method == "bm25"
