# DatXe: Ride-hailing Dynamic Pricing & Acceptance Analysis

Dự án này là một hệ thống mô phỏng, phân tích, và tối ưu hóa **Giá Động (Dynamic Pricing)** cho một nền tảng gọi xe công nghệ. Bằng cách xây dựng các bộ dữ liệu tổng hợp (Synthetic Data) và ứng dụng Khoa học Dữ liệu (Linear & Logistic Regression), dự án bóc tách được hành vi đối lập giữa Khách hàng và Tài xế, từ đó tìm ra điểm cân bằng mang lại lợi nhuận cao nhất cho nền tảng.

*Lưu ý: Đây là ví dụ xây dựng một dự án demo với dữ liệu synthetic, tài liệu chỉ được sử dụng để tham khảo, không nên sử dụng dữ liệu trong dự án này với mục đích thương mại.*

📚 **Báo cáo Chi tiết**: Có thể xem toàn bộ phương pháp luận, cơ sở toán học và đánh giá chuyên sâu của dự án tại Báo cáo Học thuật (PDF): **[reports/academic_report/main.pdf](reports/academic_report/main.pdf)**.

---

## 🎯 1. Mục tiêu Dự án
1. **Mô phỏng Hệ thống**: Tạo ra hai bộ dữ liệu độc lập phản ánh đúng thực tế của thị trường gọi xe (ảnh hưởng của mưa, giờ cao điểm, khoảng cách, và hiệu ứng cá nhân).
2. **Phân tích Hành vi (Price Acceptance)**: Định lượng bằng toán học xem sự biến động của giá, thời tiết và giao thông ảnh hưởng thế nào đến quyết định *Nhận cuốc (Tài xế)* và *Đặt xe (Khách hàng)*.
3. **Tối ưu hóa Giá (Dynamic Pricing)**: Thuật toán hóa bài toán tìm điểm giá "Ngọt" (Sweet Spot) để tối đa hóa Lợi nhuận kỳ vọng của nền tảng mà không làm gãy vỡ tỷ lệ ghép nối (Match Rate).
4. **Kiểm định Thống kê (Statistical Validation)**: Đảm bảo mọi kết luận đều dựa trên nền tảng vững chắc (P-value, Confidence Intervals, Train/Test Split) thay vì ngẫu nhiên.

---

## 📂 2. Cấu trúc Dự án

```text
.
├── data/
│   ├── driver_synthetic_trips.csv      # Tập dữ liệu hành vi của Tài xế
│   └── customer_synthetic_trips.csv    # Tập dữ liệu hành vi của Khách hàng
├── docs/
│   ├── policy_recommendations.md       # Báo cáo Khuyến nghị Chính sách Kinh doanh
│   └── superpowers/                    # Tài liệu Design & Specs ban đầu
├── notebooks/
│   ├── driver_price_acceptance.ipynb       # Phân tích mô hình giá & tài xế
│   ├── optimize_betas_4_contexts.ipynb     # Tìm Beta định giá theo Dummy Variables
│   ├── customer_price_acceptance.ipynb     # Phân tích đường cầu (Demand Curve) của khách
│   └── dynamic_pricing_optimization.ipynb  # Heatmap tối ưu hóa giá 2 chiều
├── src/
│   ├── generate_driver_data.py         # Script sinh data Tài xế
│   ├── generate_customer_data.py       # Script sinh data Khách hàng
│   ├── create_context_nb.py            # Code tự động tạo Notebook Context
│   ├── create_customer_nb.py           # Code tự động tạo Notebook Customer
│   └── create_dynamic_pricing_nb.py    # Code tự động tạo Notebook Optimization
├── tests/
│   ├── test_generate_driver_data.py    # Unit tests & Stats tests cho data Tài xế
│   └── test_generate_customer_data.py  # Unit tests & Stats tests cho data Khách hàng
└── README.md                           # File bạn đang đọc
```

---

## 📊 3. Kết quả Phân tích (Key Findings)

### A. Phân tích phía Tài xế (Driver Side)
- **Cấu trúc phụ phí**: Thuật toán Linear Regression đã bóc tách chính xác công thức cấu thành nên giá (Base Price + Phụ phí Mưa + Phụ phí Cao điểm).
- **Hành vi Nhận cuốc**: Tài xế **nhạy cảm dương** với giá. Giá thù lao càng cao, khả năng họ chấp nhận chạy càng lớn. Tuy nhiên, yếu tố thời tiết khắc nghiệt mang **dấu âm**, nghĩa là mưa và tắc đường tạo ra một lực cản khiến tài xế dễ dàng tắt app nếu không được bù đắp xứng đáng bằng vật chất.

### B. Phân tích phía Khách hàng (Customer Side)
- **Đường Cầu dốc xuống**: Đúng với kinh tế học, $\alpha_{\text{price}}$ của khách hàng là số **ÂM**. Nghĩa là mức giá càng đắt đỏ, xác suất ấn nút "Đặt xe" càng sụt giảm thảm hại (Demand Curve).
- **Cầu thiếu co giãn (Inelastic Demand)**: Tuy nhiên, điều thần kỳ xảy ra khi thêm yếu tố Mưa và Cao điểm. $\alpha_{\text{rain}}$ và $\alpha_{\text{rush}}$ là những số dương rất lớn. Ở các hoàn cảnh bất lợi này, khách hàng bị chi phối bởi "Sự cấp thiết" nhiều hơn "Độ nhạy giá", họ sẵn sàng cắn răng trả mức giá cao để có xe.

### C. Tối ưu hóa Giá Động (Dynamic Pricing)
Bằng cách thiết lập hàm: 
$$\mathbb{E}[\text{Profit}] = (\text{Thu của Khách} - \text{Trả cho Tài xế}) \times P(\text{Match})$$
Và dùng thuật toán Grid Search quét hàng vạn tổ hợp giá, ta tìm ra:
- **Ngữ cảnh bình thường**: Nền tảng thiết lập mức giá Thu $47k$ và Trả $38k$, thu về biên độ lợi nhuận khoảng $9k/chuyến$.
- **Ngữ cảnh Mưa + Cao điểm (Surge Pricing)**: Thuật toán tự động đẩy giá Thu lên $87k$ và Trả lên $78k$. Nền tảng vẫn duy trì lợi nhuận $9k$, nhưng **cứu sống được toàn bộ Match Rate** (nếu giữ nguyên mức 47k thì sẽ không tài xế nào chịu chạy).
- **Heatmap 2D**: Minh họa rõ "Sweet Spot" nơi nền tảng có thể tối đa hóa doanh thu mà không khiến cả khách lẫn tài rời bỏ.

---

## 💼 4. Khuyến nghị Chính sách (Policy Recommendations)

Tất cả các kiểm định `statsmodels` (P-values) và `scikit-learn` (ROC AUC) trên tập Test đều xác nhận mô hình hoàn toàn **có ý nghĩa thống kê và đáng tin cậy**. Dựa trên cơ sở đó, nền tảng nên áp dụng:

1. ⚡ **Mạnh dạn đẩy Surge Pricing khi thời tiết xấu**: Không cần lo mất khách vì lúc này cầu của khách là *thiếu co giãn*, họ cần xe hơn cần tiết kiệm. Bắt buộc phải tăng giá để giữ chân tài xế.
2. 🧮 **Định giá theo Context (Context-aware Pricing)**: Dùng ma trận giá tối ưu cho riêng 4 tổ hợp ngữ cảnh thời tiết/giao thông thay vì dùng 1 công thức tuyến tính.
3. 💸 **Trợ giá chéo (Cross-Subsidization)**: Thu lợi nhuận khủng từ những người chấp nhận giá cao giờ cao điểm, dùng số tiền đó tạo Quỹ thưởng (Quest) cho tài xế chạy vào các khung giờ siêu thấp điểm nhằm duy trì hệ sinh thái 24/7.
4. 🎁 **Cá nhân hóa Voucher (Personalization)**: Phát hiện ra khách hàng nào dễ hủy chuyến do giá (nhạy cảm giá cao) để phân phối voucher mồi, thay vì đốt tiền phát đại trà cho cả những khách hàng VIP ít bận tâm về giá.

*(Đọc báo cáo chi tiết tại: [docs/policy_recommendations.md](docs/policy_recommendations.md))*
