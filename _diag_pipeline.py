"""Temp diagnostic: why does the production pipeline answer 'Không tìm thấy'? Deleted after run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.m2_search import BM25Search, DenseSearch, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from config import COLLECTION_NAME, HYBRID_TOP_K, RERANK_TOP_K

QUERIES = [
    "Bao lâu phải đổi mật khẩu một lần?",
    "Nghỉ phép không lương 20 ngày cần ai phê duyệt?",
    "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?",
]

# Chunk đã enrich nằm trong Qdrant; lấy lại từ payload để khỏi gọi API enrich lần nữa
dense = DenseSearch()
info = dense.client.get_collection(COLLECTION_NAME)
print("collection points:", info.points_count)
pts, _ = dense.client.scroll(COLLECTION_NAME, limit=1000, with_payload=True)
chunks = [{"text": p.payload["text"], "metadata": p.payload} for p in pts]
print("scrolled:", len(chunks))
print("\n--- sample enriched chunk ---")
print(repr(chunks[0]["text"][:400]))

bm25 = BM25Search()
bm25.index(chunks)
reranker = CrossEncoderReranker()

for q in QUERIES:
    b = bm25.search(q, top_k=HYBRID_TOP_K)
    d = dense.search(q, top_k=HYBRID_TOP_K, collection=COLLECTION_NAME)
    hybrid = reciprocal_rank_fusion([b, d], top_k=HYBRID_TOP_K)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in hybrid]
    reranked = reranker.rerank(q, docs, top_k=RERANK_TOP_K)
    print(f"\n########## {q}")
    for r in reranked:
        print(f"  [{r.rank}] {r.rerank_score:.4f} {r.metadata.get('source')}")
        print(f"      {r.text[:300]!r}")
