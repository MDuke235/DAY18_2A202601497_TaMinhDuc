from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_MODEL_CACHE: dict[str, object] = {}


def _get_cross_encoder(model_name: str):
    """Lazy-load + cache CrossEncoder theo tên model (load 1 lần cho cả process).

    Checkpoint bge-reranker-v2-m3 ~2.2GB. Tests khởi tạo CrossEncoderReranker()
    riêng cho từng test case, nên phải cache ở module-level, không per-instance.
    """
    if model_name not in _MODEL_CACHE:
        # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
        # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _get_cross_encoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        Cross-encoder đọc cặp (query, doc) cùng lúc nên chính xác hơn bi-encoder,
        nhưng chậm hơn — chỉ chạy trên top-k của retrieval, không chạy cả corpus.
        Model load lỗi → trả [] kèm cảnh báo để pipeline fallback về thứ tự hybrid.
        """
        if not documents:
            return []

        try:
            model = self._load_model()
        except Exception as e:
            print(f"[WARNING] Reranker '{self.model_name}' load failed: {type(e).__name__}: {e}")
            return []

        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)
        if not hasattr(scores, "__len__"):                  # 1 pair có thể trả scalar
            scores = [scores]

        # key chỉ lấy score: nếu 2 score trùng nhau, sorted sẽ so sánh dict và raise
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional.

    Model mặc định (ms-marco-TinyBERT-L-2-v2) train trên tiếng Anh nên chất lượng
    trên tiếng Việt kém hơn bge-reranker-v2-m3 rõ rệt — chỉ dùng khi latency là
    ràng buộc chính. So sánh bằng benchmark_reranker() ở __main__.
    """
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank bằng FlashRank. Chưa cài package hoặc tải model lỗi → trả []."""
        if not documents:
            return []

        try:
            from flashrank import RerankRequest

            model = self._load_model()
            passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"[WARNING] FlashRank skipped: {type(e).__name__}: {e}")
            return []

        # FlashRank trả sẵn thứ tự score giảm dần; id là index trong documents
        return [
            RerankResult(
                text=documents[r["id"]]["text"],
                original_score=float(documents[r["id"]].get("score", 0.0)),
                rerank_score=float(r["score"]),
                metadata=documents[r["id"]].get("metadata", {}),
                rank=i,
            )
            for i, r in enumerate(results[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):                  # console Windows mặc định cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    # score = điểm retrieval giả lập: doc đúng đang xếp thứ 3, reranker phải kéo lên #1
    docs = [
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.82, "metadata": {"source": "it.md"}},
        {"text": "VPN dùng WireGuard AES-256.", "score": 0.79, "metadata": {"source": "vpn.md"}},
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.75, "metadata": {"source": "hr.md"}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.71, "metadata": {"source": "hr.md"}},
        {"text": "Đăng ký phép năm trước 3 ngày làm việc.", "score": 0.68, "metadata": {"source": "hr.md"}},
    ]

    print(f"Query: {query}")
    print("\n--- Retrieval order (before rerank) ---")
    for i, d in enumerate(docs):
        print(f"  [{i}] {d['score']:.4f} | {d['text']}")

    reranker = CrossEncoderReranker()
    print(f"\n--- CrossEncoder rerank (top-{RERANK_TOP_K}) ---")
    for r in reranker.rerank(query, docs):
        print(f"  [{r.rank}] {r.rerank_score:8.4f} (retrieval {r.original_score:.4f}) | {r.text}")

    stats = benchmark_reranker(reranker, query, docs, n_runs=5)
    print(f"\nLatency ({len(docs)} docs, 5 runs): avg {stats['avg_ms']:.1f}ms "
          f"| min {stats['min_ms']:.1f}ms | max {stats['max_ms']:.1f}ms")

    flash = FlashrankReranker()
    flash_results = flash.rerank(query, docs)
    if flash_results:                                       # optional: chỉ in khi FlashRank chạy được
        print(f"\n--- FlashRank rerank (top-{RERANK_TOP_K}) ---")
        for r in flash_results:
            print(f"  [{r.rank}] {r.rerank_score:8.4f} | {r.text}")
        f_stats = benchmark_reranker(flash, query, docs, n_runs=5)
        print(f"Latency: avg {f_stats['avg_ms']:.1f}ms | min {f_stats['min_ms']:.1f}ms "
              f"| max {f_stats['max_ms']:.1f}ms")
