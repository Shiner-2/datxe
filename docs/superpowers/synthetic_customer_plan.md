# Customer Price Acceptance Synthetic Data Design

Mục tiêu là xây dựng tập dữ liệu mô phỏng (synthetic data) và quy trình phân tích từ góc nhìn của **Customer (Khách hàng)** thay vì Driver (Tài xế). Khách hàng sẽ có hành vi ngược lại với tài xế: giá càng cao thì xác suất khách hàng đồng ý (đặt xe) càng giảm. Tuy nhiên, trong điều kiện mưa hoặc giờ cao điểm, mức độ sẵn sàng trả giá cao của khách hàng có thể tăng lên.

## User Review Required

1. **Hệ số tác động của giá (alpha_price)**: Dự kiến thiết lập hệ số này là số âm để phản ánh đúng thực tế: giá càng cao, tỷ lệ đặt xe càng thấp.
2. **Ảnh hưởng của thời tiết/giao thông**: Trong logic Logistic Regression của khách hàng, các biến `rain` và `rush_hour` sẽ mang hệ số dương, vì khi trời mưa hoặc cao điểm, khách hàng thường sẵn sàng chấp nhận mức giá cao hơn bình thường để có xe.

## Proposed Changes

### 1. Script tạo dữ liệu `src/generate_customer_data.py`
Tạo một script mới dựa trên logic của tài xế nhưng thay thế và bổ sung:
- Thay `driver_id` thành `customer_id`.
- Kế thừa lại mô hình sinh `delta_price` (phụ phí do mưa, cao điểm, khu vực).
- **Bổ sung mô hình Customer Acceptance**:
  - `customer_accept_prob = sigmoid(alpha0 + alpha_price * final_price + alpha_rain * rain + alpha_rush * rush_hour + customer_effect + noise)`
  - Điều kiện: `alpha_price < 0`, `alpha_rain > 0`, `alpha_rush > 0`.
  - Sinh biến nhị phân `customer_accept` (0 hoặc 1) dựa trên xác suất này.
- Đầu ra lưu vào `data/customer_synthetic_trips.csv`.

### 2. Phân tích Notebook `notebooks/customer_price_acceptance.ipynb`
Tạo một notebook (thông qua 1 script generator tương tự như đã làm) bao gồm:
- **Load và explore dữ liệu**: Kiểm tra phân phối giá và tỷ lệ accept.
- **Mô hình tuyến tính (Linear Regression)**: Xác định các $\beta$ của `delta_price` tương tự để hiểu cấu trúc giá.
- **Mô hình Logistic Regression**: Phân tích hành vi `customer_accept ~ final_price + rain + rush_hour` để chứng minh giá ảnh hưởng âm (negative) đến quyết định của khách hàng.
- **Tối ưu hóa và so sánh 4 ngữ cảnh**: Trực quan hóa tỷ lệ accept của khách hàng theo 4 ngữ cảnh (Không Mưa-Bình Thường, Mưa-Bình Thường, v.v.) và xem xét cách giá ảnh hưởng trong từng ngữ cảnh.

## Verification Plan

### Automated Tests
- Chạy script tạo data và kiểm tra file CSV xem có cột `customer_accept` không.
- Chạy script sinh notebook để tự động sinh `customer_price_acceptance.ipynb`.

### Manual Verification
- Mở notebook để đảm bảo hệ số Logistic của `final_price` là số âm, và hệ số của `rain`, `rush_hour` trong mô hình Logistic là số dương.
- Kiểm tra các biểu đồ trực quan trong notebook xem có hợp lý hay không.
