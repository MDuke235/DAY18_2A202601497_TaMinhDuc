# Failure Analysis — Lab 18: Production RAG

**Cá nhân:** Tạ Minh Đức
**Dữ liệu đánh giá:** 20 câu hỏi trong `test_set.json`

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8250 | 0.7500 | -0.0750 |
| Answer Relevancy | 0.7153 | 0.6913 | -0.0240 |
| Context Precision | 0.9250 | 0.9167 | -0.0083 |
| Context Recall | 0.9250 | 0.8667 | -0.0583 |

Production đạt ngưỡng rubric ở 3/4 metrics, nhưng chưa vượt baseline. Enrichment tạo thêm câu ngữ cảnh và hierarchical chunking tạo nhiều đoạn nhỏ hơn; điều này giúp tìm đúng tài liệu nhưng đôi khi làm mất quan hệ giữa nhiều quy định hoặc khiến generator đánh giá context là chưa đủ.

## Error Tree

```text
Output sai
├─ Context không chứa đủ bằng chứng
│  ├─ Câu hỏi multi-hop cần nhiều section/tài liệu
│  ├─ Chunk boundary tách điều kiện khỏi mức phê duyệt
│  └─ Top-3 sau rerank thiếu một nhánh bằng chứng
├─ Context có bằng chứng nhưng answer sai
│  ├─ Generator từ chối: "Không tìm thấy"
│  ├─ Không xử lý phủ định trực tiếp
│  ├─ Không ưu tiên phiên bản mới nhất
│  └─ Tính toán số học thiếu pro-rata
└─ Context đúng và answer đúng nhưng metric thấp
   ├─ Ground truth yêu cầu nhiều chi tiết hơn câu trả lời
   └─ Context chứa thêm đoạn không liên quan
```

## Bottom-5 Failures

### #1 — Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt; xác nhận cấu hình từ CNTT; tối thiểu 3 báo giá.
- **Got:** “Không tìm thấy.”
- **Metrics:** faithfulness 0.0000; answer relevancy 0.0000; context precision 1.0000; context recall 0.3333.
- **Error Tree:** Output sai → context chỉ có một phần bằng chứng → query multi-hop chưa được bao phủ.
- **Root cause:** Câu hỏi cần hợp nhất ba quy định: ngưỡng phê duyệt, xác nhận CNTT và báo giá. Top context chỉ thu hồi khoảng một phần ba ground truth.
- **Suggested fix:** Multi-query theo ba ý; tăng candidate pool; dùng diverse reranking hoặc parent retrieval để giữ các điều kiện cùng nhau.

### #2 — Nghỉ không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** CEO phê duyệt; trên 14 ngày nhân viên tự đóng phần bảo hiểm của mình.
- **Got:** “Không tìm thấy.”
- **Metrics:** faithfulness 0.0000; answer relevancy 0.0000; context precision 1.0000; context recall 0.5000.
- **Error Tree:** Output sai → context có mức 16–30 ngày nhưng thiếu/không sử dụng đủ → generator từ chối.
- **Root cause:** Retrieval đã lấy đoạn chứa CEO nhưng generator cũ quá thận trọng với context rời rạc; chi tiết bảo hiểm nằm ở section kế tiếp.
- **Suggested fix:** Prompt yêu cầu trả lời khi chỉ một đoạn đã đủ, đồng thời trả parent chunk hoặc thêm neighbor section.

### #3 — Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày theo chính sách v2.0; v1.0 yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** “Không tìm thấy.”
- **Metrics:** faithfulness 0.0000; answer relevancy 0.0000; context precision 0.8333; context recall 1.0000.
- **Error Tree:** Output sai → context đúng và đủ → generator không giải quyết xung đột phiên bản.
- **Root cause:** Cả v1 và v2 cùng được retrieve. Metadata chưa có trường version/effective date dùng để lọc trước generation.
- **Suggested fix:** Trích xuất version và ngày hiệu lực vào metadata; boost phiên bản mới; prompt nêu rõ ưu tiên chính sách mới nhất.

### #4 — Hoàn ứng trễ 5 ngày

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 300.000 VNĐ/tháng, pro-rata khoảng 50.000 VNĐ cho 5 ngày.
- **Got:** 300.000 VNĐ.
- **Metrics:** faithfulness 0.0000; answer relevancy 0.8953; context precision 0.8333; context recall 1.0000.
- **Error Tree:** Output gần đúng → context đúng → bước tính toán sai.
- **Root cause:** Model áp dụng trực tiếp mức 2%/tháng nhưng không quy đổi theo 5 ngày quá hạn.
- **Suggested fix:** Tách số liệu có đơn vị, dùng calculator cho công thức `15.000.000 × 2% × 5/30`, rồi sinh câu trả lời từ kết quả đã kiểm tra.

### #5 — PVI cho nhân viên thử việc

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Không; chỉ tham gia bảo hiểm xã hội bắt buộc.
- **Got:** “Không tìm thấy.”
- **Metrics:** faithfulness 1.0000; answer relevancy 0.0000; context precision 1.0000; context recall 1.0000.
- **Error Tree:** Output sai → context đúng và đủ → generator xử lý phủ định kém.
- **Root cause:** Context nói rõ “chưa được hưởng PVI”, nhưng generator cũ không chuyển bằng chứng phủ định thành câu trả lời trực tiếp.
- **Suggested fix:** Prompt bắt buộc trả lời Có/Không trước, sau đó trích dẫn điều kiện; thêm test riêng cho negation queries.

## Case Study — Chu kỳ đổi mật khẩu

1. **Output đúng?** Không; hệ thống trả “Không tìm thấy”.
2. **Context đúng?** Có; context recall đạt 1.0 và chứa cả 90 ngày lẫn 120 ngày.
3. **Query/retrieval đúng?** Có, nhưng không phân biệt phiên bản hiện hành.
4. **Fix phù hợp:** metadata version + effective date, boost tài liệu v2 và prompt ưu tiên bản mới nhất.

## Nếu có thêm 1 giờ

1. Thêm version-aware retrieval cho `mat_khau_v1.md`/`mat_khau_v2.md`.
2. Thêm parent/neighbor expansion cho câu hỏi multi-hop.
3. Dùng calculator cho numeric queries.
4. Tạo regression set riêng cho negation, version conflict và multi-hop.
