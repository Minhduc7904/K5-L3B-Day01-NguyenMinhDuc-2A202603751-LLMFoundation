# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature điều khiển độ ngẫu nhiên khi chọn token tiếp theo. Ở 0,0 câu trả lời gần như luôn giống nhau giữa các lần gọi và chọn ý "an toàn" nhất. Tăng lên 0,5 và 1,0 thì mỗi lần gọi chọn một sự thật khác nhau, cách diễn đạt đa dạng và sáng tạo hơn. Ở 1,5 phản hồi dễ lan man, lạ hoặc kém chính xác, thậm chí có thể sai sự thật. Nghĩa là temperature càng cao thì càng đa dạng nhưng càng kém ổn định và kém đáng tin.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình đặt temperature thấp, khoảng 0,0 đến 0,3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán và đúng chính sách, nên cùng một câu hỏi phải ra cùng một câu trả lời, không nên "sáng tạo" thêm thông tin sai về giá hay điều khoản. Nhiệt độ thấp cũng dễ kiểm thử và dễ tái hiện lỗi hơn.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> Cả giá input và output của `gpt-6.1-terra` đều gấp đúng 10 lần `gpt-6-luna` ($1,00 so với $0,10 và $5,00 so với $0,50 mỗi 1M token), nên bất kể tỉ lệ input/output, `gpt-6.1-terra` đắt hơn **đúng 10 lần**. Workload: 10.000 người × 3 lần gọi = 30.000 lần gọi mỗi ngày × 350 token = 10,5 triệu token mỗi ngày. Nếu giả định mỗi lần gọi gồm 250 token input và 100 token output thì có 7,5 triệu token input và 3 triệu token output: `gpt-6.1-terra` tốn 7,5 × $1,00 + 3 × $5,00 = $22,50 mỗi ngày (khoảng $675 mỗi tháng), còn `gpt-6-luna` tốn 7,5 × $0,10 + 3 × $0,50 = $2,25 mỗi ngày (khoảng $67,50 mỗi tháng). Chênh khoảng $20 mỗi ngày.

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> **`gpt-6.1-terra` xứng đáng** khi sai sót tốn kém hoặc bài toán khó: ví dụ tóm tắt hợp đồng pháp lý, phân tích tài chính, sinh code phức tạp hoặc suy luận nhiều bước, nơi chất lượng cao hơn đáng với giá gấp 10 lần. **`gpt-6-luna` tốt hơn** khi bài toán đơn giản, khối lượng lớn và cần nhanh, rẻ: ví dụ phân loại tin nhắn, trích xuất trường dữ liệu, trả lời câu hỏi thường gặp. Ngay trong lab này, trên tập SUB 40 tin, `gpt-6-luna` đạt F1 0,950 so với 0,974 của `gpt-6.1-terra` (chênh khoảng 1 tin) với giá chỉ bằng 1/10.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất ở các ứng dụng tương tác với người dùng thật, như chatbot hay trợ lý viết, khi câu trả lời dài và người dùng phải chờ. Chữ hiện ra ngay khi token đầu tiên được sinh nên thời gian chờ cảm nhận được thấp hơn nhiều, dù tổng thời gian sinh xong không đổi. Ngược lại, non-streaming phù hợp hơn khi cần cả kết quả trọn vẹn rồi mới xử lý: ví dụ phân loại nhãn, trích xuất JSON, chạy batch hay pipeline nền, nơi không có người đang đọc và phải parse toàn bộ câu trả lời. Khi đó streaming chỉ làm code phức tạp hơn mà không có lợi ích gì.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
