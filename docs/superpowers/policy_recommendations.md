# Khuyến nghị Chính sách (Policy Recommendations) cho Nền tảng Xe công nghệ

Dựa trên các bài test thống kê chặt chẽ trong `tests/test_generate_customer_data.py` cũng như kết quả trực quan từ các notebook, dữ liệu và mô hình phân tích hành vi Khách hàng & Tài xế đã được chứng minh là hoàn toàn có thể tin tưởng được (Statistically Significant). Các hệ số trong mô hình phản ánh đúng quy luật kinh tế học:
- **Tài xế**: Nhạy cảm dương với giá (Giá cao -> Nhận cuốc nhiều hơn).
- **Khách hàng**: Nhạy cảm âm với giá (Giá cao -> Đặt xe ít hơn), nhưng độ nhạy cảm giảm đáng kể khi có biến động thời tiết/giao thông.

Dưới đây là các chính sách được khuyến nghị để tối ưu hóa vận hành và lợi nhuận cho nền tảng:

## 1. Mạnh dạn triển khai Surge Pricing (Phụ phí tăng giá) khi thời tiết xấu
> [!TIP]
> **Khuyến nghị**: Hệ thống cần áp dụng ngay lập tức phụ phí cao khi trời mưa hoặc giờ cao điểm. 

**Bằng chứng phân tích**: 
Từ mô hình Logistic Regression của khách hàng, hệ số $\alpha_{\text{rain}}$ và $\alpha_{\text{rush}}$ là các số **dương rất lớn**, đủ để bù trừ lại hiệu ứng tiêu cực của giá ($\alpha_{\text{price}} < 0$). Điều này chứng minh rằng "Nhu cầu bắt xe khi trời mưa/cao điểm là nhu cầu thiếu co giãn (inelastic demand)". Khách hàng trong hoàn cảnh này ưu tiên **"có xe"** hơn là **"giá rẻ"**.
Nếu không tăng giá, nền tảng sẽ mất cơ hội tạo doanh thu, đồng thời tài xế sẽ tắt app vì giá thấp không bõ công chạy đường tắc.

## 2. Áp dụng Bảng giá linh hoạt theo Ngữ cảnh (Context-aware Pricing)
> [!NOTE]
> **Khuyến nghị**: Không sử dụng một công thức giá chung cho mọi thời điểm. Thay vào đó, áp dụng trực tiếp 4 mức phụ phí tối ưu ($\beta$) cho 4 tổ hợp ngữ cảnh: (Không mưa-Bình thường, Mưa-Bình thường, Không mưa-Cao điểm, Mưa-Cao điểm).

**Bằng chứng phân tích**:
Notebook `optimize_betas_4_contexts.ipynb` đã chứng minh mô hình tương tác (Dummy variables cho 4 ngữ cảnh) mang lại chỉ số RMSE thấp hơn và khớp $100\%$ với giá trị trung bình thực tế so với mô hình cộng dồn độc lập. Bằng cách thiết lập hệ thống định giá theo tập luật ngữ cảnh này, công ty sẽ vừa tối đa hóa thặng dư tiêu dùng (consumer surplus) vừa cân đối được tỷ lệ hủy chuyến.

## 3. Chính sách Trợ giá chéo (Cross-Subsidization)
> [!IMPORTANT]
> **Khuyến nghị**: Trích một phần lợi nhuận (margin) từ các chuyến xe giá siêu cao trong ngữ cảnh 4 (Mưa + Cao điểm) để tạo quỹ thưởng (Quest/Bonus) cho tài xế vào các khung giờ thấp điểm.

**Bằng chứng phân tích**:
Biểu đồ Dual-axis ở cuối `customer_price_acceptance.ipynb` cho thấy tỷ lệ đồng ý (Acceptance Rate) ở ngữ cảnh 4 vẫn rất cao dù giá chạm đỉnh. Đây là điểm tạo ra biên lợi nhuận lớn nhất cho hãng. Tuy nhiên, ở các khung giờ thấp điểm (ngữ cảnh 1), giá rẻ khiến khách đặt nhiều nhưng tài xế dễ ngó lơ vì lợi nhuận kém. Dùng thặng dư từ ngữ cảnh 4 bù đắp cho tài xế ở ngữ cảnh 1 sẽ giúp duy trì nguồn cung tài xế luôn ở mức ổn định 24/7.

## 4. Cá nhân hóa (Personalization) dựa trên hiệu ứng người dùng
> [!TIP]
> **Khuyến nghị**: Xây dựng hệ thống Voucher thông minh nhắm vào các khách hàng nhạy cảm về giá, thay vì rải voucher đại trà.

**Bằng chứng phân tích**:
Cả mô hình tài xế và khách hàng đều chỉ ra rằng khi đưa biến ID người dùng/tài xế vào mô hình, sai số (RMSE) giảm rõ rệt. Điều này chứng tỏ mỗi người có một "đường cơ sở" (baseline) riêng. Những khách hàng có độ nhạy giá cao (dễ hủy chuyến khi giá tăng nhẹ) cần được tự động phân phát Voucher để kích cầu, trong khi những khách VIP ít nhạy cảm về giá không cần tốn chi phí Voucher.
