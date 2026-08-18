from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    underthesea nối từ ghép bằng "_" ("nghỉ_phép"). BM25 tokenize bằng split(" "),
    nên phải replace("_", " ") — nếu không, doc có token "nghỉ_phép" còn query
    có 2 token "nghỉ" + "phép" → không bao giờ khớp.
    """
    try:
        from underthesea import word_tokenize
    except ImportError:
        return text                                    # fallback: chưa cài underthesea

    try:
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text                                    # fallback: segmenter lỗi input lạ


def _tokenize(text: str) -> list[str]:
    """Segment + lowercase + split. Dùng chung cho index và query để token khớp nhau."""
    return segment_vietnamese(text).lower().split()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        # BM25Okapi chia cho avgdl → corpus rỗng gây ZeroDivisionError
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in ranked
            if scores[i] > 0                           # score 0 = không share token nào
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant. Collection bị tạo lại từ đầu mỗi lần index."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            return

        # recreate_collection() đã deprecated → delete + create
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
            )
            for i in range(len(chunks))
        ]
        for start in range(0, len(points), 64):        # batch để tránh payload quá lớn
            self.client.upsert(collection, points[start:start + 64])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors.

        Collection chưa index → trả [] kèm cảnh báo thay vì raise 404, để
        HybridSearch vẫn trả được kết quả BM25 (graceful degradation).
        """
        if not self.client.collection_exists(collection):
            print(f"[WARNING] Collection '{collection}' chưa tồn tại — bỏ qua dense search.")
            return []

        query_vector = self._get_encoder().encode(query).tolist()
        # qdrant-client 1.19: .search() đã bị bỏ, dùng query_points()
        response = self.client.query_points(collection, query=query_vector, limit=top_k)

        return [
            SearchResult(
                text=pt.payload.get("text", ""),
                score=float(pt.score),
                metadata=pt.payload or {},
                method="dense",
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1).

    Chỉ dùng rank, không dùng score gốc — nên BM25 score (0..∞) và
    cosine score (0..1) trộn được với nhau mà không cần normalize.
    """
    fused: dict[str, dict] = {}

    for results in results_list:
        for rank, result in enumerate(results):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result, "methods": []})
            entry["score"] += 1.0 / (k + rank + 1)
            entry["methods"].append(result.method)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:top_k]

    return [
        SearchResult(
            text=e["result"].text,
            score=e["score"],
            metadata={**e["result"].metadata, "rrf_methods": e["methods"]},
            method="hybrid",
        )
        for e in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        try:
            self.dense.index(chunks)
        except Exception as e:
            print(f"[WARNING] Dense index unavailable; BM25 remains active: {type(e).__name__}: {e}")

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        try:
            dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        except Exception as e:
            print(f"[WARNING] Dense search unavailable: {type(e).__name__}: {e}")
            dense_results = []
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


DEMO_COLLECTION = "lab18_m2_demo"      # tách khỏi COLLECTION_NAME để demo không xóa index của pipeline


def _demo(query: str = "nghỉ phép", top_k: int = 5) -> None:
    """Index toàn bộ data/ rồi so sánh BM25 / Dense / Hybrid trên 1 query.

    Qdrant chưa chạy → dense trả [], hybrid tự động chỉ còn BM25.
    """
    from src.m1_chunking import load_documents, chunk_structure_aware

    chunks = [
        {"text": c.text, "metadata": c.metadata}
        for doc in load_documents()
        for c in chunk_structure_aware(doc["text"], metadata=doc["metadata"])
    ]
    print(f"\nIndexing {len(chunks)} chunks (structure-aware)...")

    bm25 = BM25Search()
    bm25.index(chunks)
    bm25_results = bm25.search(query, top_k=top_k)

    dense_results: list[SearchResult] = []
    try:
        dense = DenseSearch()
        dense.index(chunks, collection=DEMO_COLLECTION)
        dense_results = dense.search(query, top_k=top_k, collection=DEMO_COLLECTION)
    except Exception as e:                             # Qdrant down → demo vẫn chạy được với BM25
        print(f"[WARNING] Dense search skipped: {type(e).__name__}: {e}")

    hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)

    print(f"\nQuery: {query!r}")
    for name, results in [("BM25", bm25_results), ("Dense", dense_results), ("Hybrid (RRF)", hybrid_results)]:
        print(f"\n--- {name}: {len(results)} hits ---")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.score:7.4f}  {str(r.metadata.get('source', '?')):30}  {r.text[:60]!r}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):             # console Windows mặc định cp1252 → tiếng Việt lỗi
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
    _demo()
