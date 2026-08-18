# Báo cáo tổng kết — Lab 18: Production RAG

**Học viên:** Tạ Minh Đức
**Ngày hoàn thiện:** 18/08/2026

## Phân công và trạng thái

Đây là bài cá nhân; một học viên thực hiện toàn bộ năm module.

| Người thực hiện | Module | Trạng thái | Test hiện có |
|---|---|---|---:|
| Tạ Minh Đức | M1: Chunking | Hoàn thành | 15 |
| Tạ Minh Đức | M2: Hybrid Search | Hoàn thành | 7 |
| Tạ Minh Đức | M3: Reranking | Hoàn thành | 8 |
| Tạ Minh Đức | M4: Evaluation | Hoàn thành | 4 |
| Tạ Minh Đức | M5: Enrichment | Hoàn thành | 10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8250 | 0.7500 | -0.0750 |
| Answer Relevancy | 0.7153 | 0.6913 | -0.0240 |
| Context Precision | 0.9250 | 0.9167 | -0.0083 |
| Context Recall | 0.9250 | 0.8667 | -0.0583 |

Production đạt ≥0.70 ở ba metrics, đủ mức cao nhất của tiêu chí RAGAS trong rubric. Tuy nhiên, kết quả thấp hơn baseline cho thấy pipeline phức tạp hơn không tự động tốt hơn nếu chunk boundary, enrichment và version handling chưa được hiệu chỉnh trên test set.

## Latency breakdown từ lần chạy hoàn chỉnh

| Stage | Thời gian |
|---|---:|
| Chunk 26 tài liệu thành 104 chunks | 0.2 s |
| Combined enrichment, 104 API calls | 326.2 s |
| Dense/BM25 indexing | 79.9 s |
| RAGAS evaluation | 42.3 s |
| Tổng `python main.py` | 963.3 s |

Enrichment là bước build-time tốn thời gian nhất; query-time còn chịu cold-start của reranker khoảng 36.9 giây ở lần tải đầu. Model được cache trong process nên các lần sau nhanh hơn.

Smoke test 3 truy vấn trên collection production hiện có, gồm đầy đủ OpenAI generation:

| Query stage | Avg | P50 | P95/Max (n=3) |
|---|---:|---:|---:|
| Retrieve | 6,416.3 ms | 3,727.0 ms | 12,903.5 ms |
| Rerank | 15,993.0 ms | 16,510.0 ms | 17,708.2 ms |
| Generate | 3,563.0 ms | 2,395.8 ms | 6,169.0 ms |
| Total | 25,972.3 ms | 22,361.2 ms | 36,780.7 ms |

Mẫu chỉ có ba truy vấn nên P95 trùng max; số liệu cho thấy cần warm-up và tối ưu CPU inference trước khi đặt SLO production.

## Key findings

1. **Điểm mạnh:** Hybrid retrieval và reranker tìm được đúng các đoạn liên quan cho các câu lookup; context precision đạt 0.9167.
2. **Thách thức lớn nhất:** Câu multi-hop, phủ định và xung đột phiên bản cần logic ngoài similarity search thuần túy.
3. **Phát hiện bất ngờ:** Baseline dense-only đạt điểm cao hơn production trên test set hiện tại; cần đánh giá theo dữ liệu thay vì giả định thêm module luôn cải thiện chất lượng.

## Presentation notes

1. Trình bày bảng RAGAS và nhấn mạnh 3/4 metrics đạt ≥0.70.
2. Demo câu “Bao lâu phải đổi mật khẩu?” để minh họa lỗi version conflict.
3. Walkthrough Error Tree: retrieval đúng → context đủ → generator không ưu tiên v2.
4. Tối ưu tiếp theo: version metadata, parent expansion và calculator cho numeric queries.
