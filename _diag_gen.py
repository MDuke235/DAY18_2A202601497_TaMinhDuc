"""Temp diagnostic 2: reproduce the generation step with the exact reranked contexts."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from openai import OpenAI

CASES = {
    "Bao lâu phải đổi mật khẩu một lần?": [
        "Đoạn văn nằm trong tài liệu hướng dẫn về bảo mật mật khẩu.\n\nPhương thức MFA được chấp nhận: ứng dụng Authenticator (ưu tiên), SMS OTP, hoặc YubiKey. ## Chu kỳ thay đổi Mật khẩu phải được thay đổi **mỗi 120 ngày**. Hệ thống tự động nhắc nhở trước 14 ngày.",
        "Đoạn văn nằm trong phần hướng dẫn về bảo mật mật khẩu trong tài liệu mat_khau_v1.md.\n\nKhông được sử dụng tên đăng nhập hoặc các thông tin cá nhân dễ đoán làm mật khẩu. ## Chu kỳ thay đổi Mật khẩu phải được thay đổi **mỗi 90 ngày**. Hệ thống sẽ tự động nhắc nhở trước 7 ngày.",
        "Đoạn văn nằm ở phần cập nhật chính sách mật khẩu trong tài liệu.\n\nChính sách này đã được thay thế bởi Chính sách mật khẩu v2.0 từ ngày 01/07/2024.",
    ],
    "Nghỉ phép không lương 20 ngày cần ai phê duyệt?": [
        "Đoạn văn nằm trong phần quy định về nghỉ phép trong tài liệu nghi_phep_khong_luong.md.\n\nNghỉ từ 1-5 ngày: trưởng phòng phê duyệt. Nghỉ từ 6-15 ngày: cần thêm phê duyệt của Giám đốc Nhân sự. Nghỉ từ 16-30 ngày: cần phê duyệt của **Giám đốc điều hành (CEO)**. ## Ảnh hưởng đến phúc lợi",
        "Đoạn văn nằm trong phần quy định về nghỉ phép trong tài liệu.\n\nPhép năm phải được đăng ký trước ít nhất 3 ngày làm việc qua hệ thống HR Portal.",
        "Đoạn văn nằm trong phần quy định về nghỉ phép trong tài liệu nghi_phep_nam_v2024.md.\n\nVí dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3).",
    ],
    "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?": [
        "Đoạn văn nằm trong phần quy định về nhân viên thử việc trong tài liệu.\n\nNhân viên thử việc được tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng gói bảo hiểm sức khỏe PVI. ## Đánh giá thử việc",
        "Đoạn văn nằm trong phần chính sách bảo hiểm sức khỏe của tài liệu.\n\n# Chính sách bảo hiểm sức khỏe Công ty cung cấp gói bảo hiểm sức khỏe toàn diện qua **PVI Insurance** cho tất cả nhân viên chính thức.",
        "Đoạn văn nằm trong phần mô tả về bảo hiểm sức khỏe trong tài liệu bao_hiem_suc_khoe.md.\n\nChi phí gói gia đình: công ty hỗ trợ 50%.",
    ],
}

client = OpenAI()
SYS_OLD = "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"
SYS_NEW = (
    "Bạn là trợ lý tra cứu quy định nội bộ. Trả lời CHỈ dựa trên context được cung cấp.\n"
    "- Trả lời trực tiếp, đầy đủ câu hỏi, kèm con số/tên chức danh cụ thể nếu context có.\n"
    "- Context là các đoạn rời rạc, có thể lẫn tiêu đề mục hoặc câu không liên quan: "
    "chỉ cần MỘT đoạn chứa thông tin là đã đủ để trả lời.\n"
    "- Nếu context có nhiều phiên bản khác nhau, ưu tiên phiên bản mới nhất và nói rõ phiên bản.\n"
    "- Chỉ trả lời 'Không tìm thấy.' khi thực sự không có đoạn nào liên quan."
)

for q, ctxs in CASES.items():
    ctx = "\n\n".join(ctxs)
    for label, sysmsg, temp in (("OLD t=1", SYS_OLD, 1.0), ("OLD t=0", SYS_OLD, 0.0), ("NEW t=0", SYS_NEW, 0.0)):
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=temp,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": f"Context:\n{ctx}\n\nCâu hỏi: {q}"}])
        print(f"[{label}] {q}\n    -> {r.choices[0].message.content[:220]}")
    print()
