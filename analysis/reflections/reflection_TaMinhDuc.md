# Reflection — Tạ Minh Đức

## 1. Mapping bài giảng vào implementation

| Lecture concept | Module | Hàm cụ thể | Observation từ bài lab |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo 208 semantic chunks so với 51 basic chunks trên corpus; semantic boundary chi tiết hơn nhưng có nhiều chunk rất ngắn. |
| Parent–child chunking | M1 | `chunk_hierarchical()` | Tạo child để retrieve chính xác và liên kết về parent bằng `parent_id`; ID cần chứa source để không trùng giữa tài liệu. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF trộn rank mà không cần chuẩn hóa BM25 score và cosine score. Query “nghỉ phép” trả kết quả liên quan ở cả BM25, dense và hybrid. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Tài liệu đúng từ retrieval rank 3 được kéo lên rank 1 với rerank score 0.9914. Cold-start trên máy thử nghiệm khoảng 36.9 giây. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Answer relevancy thấp nhất (0.6913); failure analysis cho thấy generator từ chối, version conflict và numeric reasoning là ba nhóm lỗi chính. |
| Contextual enrichment | M5 | `_enrich_single_call()` | Combined mode giảm từ 4 xuống 1 API call/chunk, nhưng 104 chunks vẫn mất 326.2 giây và chưa cải thiện RAGAS trên test set này. |

## 2. Khó khăn và cách giải quyết

### Lỗi 1 — Windows console encoding

- **Exact error:** `UnicodeEncodeError: 'charmap' codec can't encode characters ...`
- **Cách debug:** Chạy entry point với CP1252, lần theo stack trace đến câu cảnh báo chứa emoji.
- **Giải pháp:** Dùng thông báo cảnh báo ASCII để module chạy được trên console Windows mặc định.

### Lỗi 2 — Trùng parent ID

- **Biểu hiện:** Mỗi lần chunk một tài liệu đều sinh `parent_0`, khiến child của nhiều tài liệu có ID giống nhau.
- **Cách debug:** Theo data flow từ `pipeline.py` vào `chunk_hierarchical()` và so sánh tập ID của hai nguồn.
- **Giải pháp:** Namespace ID theo source, ví dụ `source.md::parent_0`, và thêm regression test.

### Lỗi 3 — Thiếu pagefile khi chạy toàn bộ test

- **Exact error:** `OSError: The paging file is too small for this operation to complete. (os error 1455)`.
- **Cách debug:** M3 chạy riêng đạt 5/5 nhưng thất bại sau M1 trong cùng process; checkpoint reranker chiếm khoảng 2.14 GB và model semantic M1 vẫn được cache.
- **Giải pháp:** Vẫn ưu tiên BGE CrossEncoder, nhưng dùng lexical rerank có thứ tự khi model load hoặc inference lỗi để pipeline không trả rỗng.

### Lỗi 4 — Qdrant có thể gián đoạn

- **Biểu hiện:** `collection_exists`, index hoặc search có thể phát sinh connection error và làm toàn pipeline dừng dù BM25 đã sẵn sàng.
- **Giải pháp:** Graceful degradation: giữ BM25 index và fusion danh sách BM25 khi dense backend không khả dụng.

### Kiến thức cần bổ sung

- Thiết kế metadata theo version/effective date.
- Parent/neighbor expansion cho multi-hop retrieval.
- Đánh giá thống kê có lặp lại thay vì dựa vào một lần chạy RAGAS.
- Quản lý memory/cold-start cho model lớn trên Windows.

## 3. Action plan cho project cá nhân

### Project: Trợ lý tra cứu chính sách nội bộ

#### Hiện tại

- Pipeline: hierarchical chunks → combined enrichment → BM25 + BGE-M3/Qdrant → RRF → BGE reranker → grounded answer → RAGAS.
- Known issues: version conflict, multi-hop recall, numeric calculation, cold-start và chi phí enrichment.

#### Plan áp dụng

1. [ ] **Chunking:** dùng structure-aware parent sections và child 256 ký tự; trả parent/neighbor sau khi child match.
2. [ ] **Search:** giữ hybrid BM25 + dense; thêm metadata filter cho version và phòng ban.
3. [ ] **Reranking:** dùng `bge-reranker-v2-m3` cho top-20→top-3; warm-up khi service khởi động và giữ lexical fallback.
4. [ ] **Evaluation:** giữ RAGAS, bổ sung exact-match cho con số, version accuracy, negation accuracy và retrieval hit@k.
5. [ ] **Enrichment:** chỉ enrich offline khi ingest tài liệu mới; cache theo content hash để không gọi lại API cho chunk không đổi.

#### Timeline

- **Tuần 1:** chuẩn hóa metadata, version/effective date và bộ regression test.
- **Tuần 2:** triển khai parent/neighbor expansion và diverse retrieval.
- **Tuần 3:** thêm calculator cho numeric queries, prompt cho negation/version conflict.
- **Tuần 4:** benchmark chất lượng/latency/cost, chạy RAGAS ba lần và chọn cấu hình production.

#### Tiêu chí hoàn thành

- Không regression ở 47 auto-tests hiện có.
- Bốn RAGAS metrics ≥0.75 trên cùng test set hoặc có phân tích thống kê giải thích sai lệch.
- P95 query latency <3 giây sau warm-up, không tính ingestion/enrichment.
- 100% câu version, negation và numeric trong regression set trả đúng.
