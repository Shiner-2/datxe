# Kế hoạch xây dựng Synthetic Data cho bài toán phân tích giá đặt xe công nghệ

## 1. Mục tiêu nghiên cứu

Xét dữ liệu lịch sử gồm $n$ cuốc xe:

$$
i=1,\ldots,n.
$$

Mục tiêu của bài toán là:

1. Ước lượng ảnh hưởng của thời tiết mưa đến phần chênh lệch giá của chuyến xe.
2. Ước lượng ảnh hưởng của giờ cao điểm đến phần chênh lệch giá của chuyến xe.
3. Kiểm tra xem các feature khác như điểm xuất phát, điểm đích, khu vực hoạt động và tài xế có ảnh hưởng đến giá hay không.
4. Ước lượng mức ảnh hưởng riêng của từng tài xế.
5. So sánh các mô hình để xác định feature nào giúp cải thiện khả năng giải thích và dự đoán giá.
6. Trực quan hóa dữ liệu bằng scatter plot, fitted values và đường tham chiếu.

---

## 2. Định nghĩa các biến

Với chuyến xe thứ $i$, định nghĩa:

- $P_i$: giá cuối cùng của chuyến xe.
- $P_{\text{base},i}$: giá cơ sở của chuyến xe.
- $\Delta P_i$: phần giá chênh lệch so với giá cơ sở.
- $R_i$: biến chỉ báo trời mưa.
- $H_i$: biến chỉ báo giờ cao điểm.
- $O_i$: điểm xuất phát của chuyến xe.
- $G_i$: điểm đích của chuyến xe.
- $D_i$: khoảng cách của chuyến xe.
- $T_i$: thời gian di chuyển dự kiến.
- $d(i)$: tài xế thực hiện chuyến xe thứ $i$.

Phần chênh lệch giá được định nghĩa theo công thức:

$$
\Delta P_i
=

P_i-P_{\text{base},i}.
$$

Do đó:

$$
P_i
=

P_{\text{base},i}
+
\Delta P_i.
$$

Biến thời tiết được định nghĩa như sau:

$$
R_i
=

\begin{cases}
1, & \text{nếu chuyến xe diễn ra khi trời mưa},\\
0, & \text{nếu chuyến xe diễn ra khi không mưa}.
\end{cases}
$$

Biến giờ cao điểm được định nghĩa như sau:

$$
H_i
=

\begin{cases}
1, & \text{nếu chuyến xe diễn ra trong giờ cao điểm},\\
0, & \text{nếu chuyến xe diễn ra trong giờ bình thường}.
\end{cases}
$$

---

## 3. Mô hình hồi quy cơ sở

Mô hình hồi quy tuyến tính cơ sở là:

$$
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\varepsilon_i.
$$

Trong đó:

- $\beta_0$ là phần chênh lệch giá trung bình khi không mưa và không phải giờ cao điểm.
- $\beta_1$ là mức thay đổi trung bình của $\Delta P_i$ khi trời mưa, trong điều kiện giữ nguyên trạng thái giờ cao điểm.
- $\beta_2$ là mức thay đổi trung bình của $\Delta P_i$ trong giờ cao điểm, trong điều kiện giữ nguyên trạng thái thời tiết.
- $\varepsilon_i$ là sai số ngẫu nhiên chưa được mô hình giải thích.

Giả định sai số:

$$
\varepsilon_i
\sim
\mathcal{N}\left(0,\sigma^2\right).
$$

Bốn trường hợp dự đoán từ mô hình cơ sở là:

$$
\mathbb{E}
\left[
\Delta P_i
\mid
R_i=0,H_i=0
\right]
=

\beta_0,
$$

$$
\mathbb{E}
\left[
\Delta P_i
\mid
R_i=1,H_i=0
\right]
=

\beta_0+\beta_1,
$$

$$
\mathbb{E}
\left[
\Delta P_i
\mid
R_i=0,H_i=1
\right]
=

\beta_0+\beta_2,
$$

và:

$$
\mathbb{E}
\left[
\Delta P_i
\mid
R_i=1,H_i=1
\right]
=

\beta_0+\beta_1+\beta_2.
$$

---

## 4. Cấu trúc synthetic dataset

| Tên biến           | Ý nghĩa                           |
| ------------------ | --------------------------------- |
| `trip_id`          | Mã định danh chuyến xe            |
| `driver_id`        | Mã định danh tài xế               |
| `trip_time`        | Thời điểm bắt đầu chuyến xe       |
| `origin_zone`      | Khu vực xuất phát                 |
| `destination_zone` | Khu vực điểm đích                 |
| `origin_x`         | Tọa độ $x$ của điểm xuất phát     |
| `origin_y`         | Tọa độ $y$ của điểm xuất phát     |
| `destination_x`    | Tọa độ $x$ của điểm đích          |
| `destination_y`    | Tọa độ $y$ của điểm đích          |
| `distance_km`      | Khoảng cách chuyến xe             |
| `duration_minute`  | Thời gian di chuyển dự kiến       |
| `rain`             | Trạng thái mưa hoặc không mưa     |
| `rush_hour`        | Giờ cao điểm hoặc giờ bình thường |
| `base_price`       | Giá cơ sở                         |
| `delta_price`      | Phần chênh lệch giá               |
| `final_price`      | Giá cuối cùng                     |

---

## 5. Kế hoạch tạo synthetic data

Giả sử tạo:

$$
n=5{,}000
$$

chuyến xe và:

$$
K=50
$$

tài xế.

Mỗi tài xế nên có tối thiểu khoảng $50$ đến $100$ chuyến xe để việc ước lượng driver effect không quá thiếu ổn định.

### 5.1. Bước 1: Sinh tài xế

Tài xế của chuyến xe thứ $i$ được sinh từ tập:

$$
d(i)
\in
{1,2,\ldots,K}.
$$

Có thể giả định xác suất các tài xế nhận chuyến là bằng nhau:

$$
d(i)
\sim
\operatorname{Categorical}
\left(
\frac{1}{K},
\ldots,
\frac{1}{K}
\right).
$$

Mỗi tài xế có một driver effect riêng:

$$
\alpha_d
\sim
\mathcal{N}
\left(
0,
\sigma_{\text{driver}}^2
\right).
$$

Ví dụ:

$$
\sigma_{\text{driver}}
=

2{,}000.
$$

Hệ số $\alpha_d$ thể hiện phần chênh lệch giá trung bình liên quan đến tài xế $d$, sau khi đã kiểm soát các feature khác.

### 5.2. Bước 2: Sinh điểm xuất phát và điểm đích

Giả sử thành phố synthetic được biểu diễn trên mặt phẳng tọa độ:

$$
$$
[0,10]\times[0,10].
$$
$$

Điểm xuất phát của chuyến xe thứ $i$ là:

$$
O_i
=

\left(
x_i^{O},
y_i^{O}
\right).
$$

Điểm đích của chuyến xe thứ $i$ là:

$$
G_i
=

\left(
x_i^{G},
y_i^{G}
\right).
$$

Các tọa độ có thể được sinh theo phân phối đều:

$$
x_i^{O},
y_i^{O},
x_i^{G},
y_i^{G}
\sim
\operatorname{Uniform}(0,10).
$$

Khoảng cách Euclidean được tính bằng:

$$
D_i
=

\sqrt{
\left(
x_i^{G}-x_i^{O}
\right)^2
+
\left(
y_i^{G}-y_i^{O}
\right)^2
}.
$$

Để tránh những chuyến xe có khoảng cách bằng $0$ hoặc quá ngắn, có thể chỉ giữ những chuyến thỏa mãn:

$$
D_i
\geq
0.5.
$$

Nếu sử dụng kinh độ và vĩ độ thực tế, khoảng cách nên được tính bằng công thức Haversine thay vì khoảng cách Euclidean.

### 5.3. Bước 3: Chia thành các khu vực

Không nên đưa trực tiếp tên điểm xuất phát và điểm đích vào hồi quy nếu mỗi tọa độ gần như là duy nhất.

Thay vào đó, có thể chia thành phố thành $J$ khu vực:

$$
O_i
\in
{1,\ldots,J},
$$

và:

$$
G_i
\in
{1,\ldots,J}.
$$

Ví dụ:

$$
J=10.
$$

Các khu vực có thể đại diện cho trung tâm thành phố, sân bay, khu dân cư, khu thương mại hoặc khu ngoại ô.

Mỗi khu vực xuất phát có một hiệu ứng riêng:

$$
\gamma_j^{O},
\qquad
j=1,\ldots,J.
$$

Mỗi khu vực điểm đích có một hiệu ứng riêng:

$$
\gamma_j^{G},
\qquad
j=1,\ldots,J.
$$

### 5.4. Bước 4: Sinh thời điểm chuyến xe

Giờ bắt đầu chuyến xe được sinh từ:

$$
\operatorname{Hour}_i
\in
{0,1,\ldots,23}.
$$

Có thể định nghĩa giờ cao điểm buổi sáng và buổi chiều:

$$
H_i=1
$$

nếu:

$$
\operatorname{Hour}_i
\in
{7,8,9,16,17,18}.
$$

Ngược lại:

$$
H_i=0.
$$

Cách định nghĩa này hợp lý hơn việc sinh biến `rush_hour` hoàn toàn ngẫu nhiên, vì biến này được xác định trực tiếp từ thời điểm chuyến xe.

### 5.5. Bước 5: Sinh trạng thái thời tiết

Biến mưa có thể được sinh theo phân phối Bernoulli:

$$
R_i
\sim
\operatorname{Bernoulli}(p_{\text{rain}}).
$$

Ví dụ:

$$
p_{\text{rain}}
=

0.25.
$$

Điều này có nghĩa khoảng $25%$ số chuyến xe diễn ra khi trời mưa.

Để dữ liệu hợp lý hơn, có thể sinh trạng thái mưa theo ngày hoặc theo từng khoảng thời gian thay vì độc lập cho từng chuyến. Khi đó, các chuyến xe diễn ra trong cùng một khoảng thời gian sẽ có trạng thái thời tiết tương tự nhau.

### 5.6. Bước 6: Sinh thời gian di chuyển

Thời gian di chuyển phụ thuộc vào khoảng cách, giờ cao điểm và thời tiết:

$$
T_i
=

3
+
2.5D_i
+
8H_i
+
3R_i
+
\eta_i.
$$

Trong đó:

$$
\eta_i
\sim
\mathcal{N}
\left(
0,
2^2
\right).
$$

Để tránh thời gian âm hoặc không hợp lệ, sử dụng:

$$
T_i
=

\max
\left(
T_i,1
\right).
$$

### 5.7. Bước 7: Tính giá cơ sở

Giá cơ sở được xác định từ phí mở cửa, khoảng cách và thời gian di chuyển:

$$
P_{\text{base},i}
=

10{,}000
+
6{,}000D_i
+
500T_i.
$$

Trong đó:

- $10{,}000$ đồng là phí mở cửa.
- $6{,}000$ đồng là chi phí trên mỗi kilomet.
- $500$ đồng là chi phí trên mỗi phút di chuyển.

Vì khoảng cách và thời gian đã được sử dụng để tính $P_{\text{base},i}$, không nhất thiết phải tiếp tục đưa chúng vào mô hình của $\Delta P_i$.

Nếu đưa khoảng cách hoặc thời gian vào mô hình của $\Delta P_i$, cần giải thích rằng chúng ảnh hưởng đến phần phụ phí chứ không chỉ ảnh hưởng đến giá cơ sở.

### 5.8. Bước 8: Sinh phần chênh lệch giá

Để kiểm tra khả năng khôi phục các hệ số của mô hình cơ sở, có thể sinh dữ liệu từ:

$$
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\alpha_{d(i)}
+
\gamma_{O_i}^{O}
+
\gamma_{G_i}^{G}
+
\varepsilon_i.
$$

Có thể chọn các hệ số thật như sau:

$$
\beta_0
=

2{,}000,
$$

$$
\beta_1
=

8{,}000,
$$

và:

$$
\beta_2
=

12{,}000.
$$

Driver effect được sinh từ:

$$
\alpha_d
\sim
\mathcal{N}
\left(
0,
2{,}000^2
\right).
$$

Origin-zone effect được sinh từ:

$$
\gamma_j^{O}
\sim
\mathcal{N}
\left(
0,
1{,}500^2
\right).
$$

Destination-zone effect được sinh từ:

$$
\gamma_j^{G}
\sim
\mathcal{N}
\left(
0,
1{,}000^2
\right).
$$

Sai số ngẫu nhiên được sinh từ:

$$
\varepsilon_i
\sim
\mathcal{N}
\left(
0,
3{,}000^2
\right).
$$

Giá cuối cùng của chuyến xe là:

$$
P_i
=

P_{\text{base},i}
+
\Delta P_i.
$$

Để tránh giá cuối cùng âm hoặc quá thấp, có thể sử dụng:

$$
P_i
=

\max
\left(
P_i,
P_{\text{base},i}
\right).
$$

---

## 6. Thiết kế dữ liệu cân bằng

Để ước lượng chính xác hệ số của từng tài xế, mỗi tài xế nên có dữ liệu trong cả bốn nhóm:

1. Không mưa và giờ bình thường.
2. Có mưa và giờ bình thường.
3. Không mưa và giờ cao điểm.
4. Có mưa và giờ cao điểm.

Cụ thể, với mỗi tài xế $d$, nên đảm bảo:

$$
n_{d,0,0}>0,
$$

$$
n_{d,1,0}>0,
$$

$$
n_{d,0,1}>0,
$$

và:

$$
n_{d,1,1}>0.
$$

Trong đó $n_{d,r,h}$ là số chuyến của tài xế $d$ có trạng thái mưa $r$ và trạng thái giờ cao điểm $h$.

Nếu một tài xế chỉ có chuyến xe trong giờ bình thường, mô hình sẽ không thể xác định đáng tin cậy tác động của giờ cao điểm đối với tài xế đó.

---

## 7. Ước lượng mô hình cơ sở

Mô hình baseline là:

$$
\mathcal{M}_0:
\qquad
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\varepsilon_i.
$$

Đặt:

$$
\mathbf{y}
=

\begin{bmatrix}
\Delta P_1\\
\Delta P_2\\
\vdots\\
\Delta P_n
\end{bmatrix}.
$$

Ma trận thiết kế là:

$$
\mathbf{X}
=

\begin{bmatrix}
1 & R_1 & H_1\\
1 & R_2 & H_2\\
\vdots & \vdots & \vdots\\
1 & R_n & H_n
\end{bmatrix}.
$$

Vector hệ số là:

$$
\boldsymbol{\beta}
=

\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2
\end{bmatrix}.
$$

Ước lượng Ordinary Least Squares là:

$$
\widehat{\boldsymbol{\beta}}
=

\left(
\mathbf{X}^{\top}\mathbf{X}
\right)^{-1}
\mathbf{X}^{\top}\mathbf{y}.
$$

Các hệ số ước lượng kỳ vọng gần với hệ số thật:

$$
\widehat{\beta}_0
\approx
2{,}000,
$$

$$
\widehat{\beta}_1
\approx
8{,}000,
$$

và:

$$
\widehat{\beta}_2
\approx
12{,}000.
$$

Tuy nhiên, nếu bỏ qua driver effect và location effect, các hệ số trên có thể bị sai lệch.

---

## 8. Kiểm tra các feature bổ sung

### 8.1. Mô hình cơ sở

$$
\mathcal{M}_0:
\qquad
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\varepsilon_i.
$$

### 8.2. Mô hình có khu vực xuất phát và điểm đích

$$
\mathcal{M}_1:
\qquad
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\gamma_{O_i}^{O}
+
\gamma_{G_i}^{G}
+
\varepsilon_i.
$$

### 8.3. Mô hình có driver fixed effects

$$
\mathcal{M}_2:
\qquad
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\gamma_{O_i}^{O}
+
\gamma_{G_i}^{G}
+
\sum_{d=2}^{K}
\alpha_d
\mathbb{1}
\left{
d(i)=d
\right}
+
\varepsilon_i.
$$

Tài xế thứ nhất được chọn làm nhóm tham chiếu.

Hệ số $\alpha_d$ biểu diễn phần chênh lệch giá trung bình của tài xế $d$ so với tài xế tham chiếu, sau khi đã kiểm soát mưa, giờ cao điểm và khu vực chuyến xe.

---

## 9. Ước lượng hệ số riêng cho từng tài xế

Nếu mục tiêu là ước lượng một mức giá nền riêng cho từng tài xế, có thể sử dụng random-intercept model:

$$
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\gamma_{O_i}^{O}
+
\gamma_{G_i}^{G}
+
\alpha_{d(i)}
+
\varepsilon_i.
$$

Trong đó:

$$
\alpha_d
\sim
\mathcal{N}
\left(
0,
\sigma_{\text{driver}}^2
\right).
$$

Intercept riêng của tài xế $d$ là:

$$
\beta_{0,d}
=

\beta_0+\alpha_d.
$$

Nếu muốn tác động của mưa và giờ cao điểm khác nhau giữa các tài xế, sử dụng random-slopes model:

$$
\Delta P_i
=

\beta_0
+
\alpha_{d(i)}
+
\left(
\beta_1+u_{1,d(i)}
\right)R_i
+
\left(
\beta_2+u_{2,d(i)}
\right)H_i
+
\gamma_{O_i}^{O}
+
\gamma_{G_i}^{G}
+
\varepsilon_i.
$$

Tác động của mưa đối với tài xế $d$ là:

$$
\beta_{1,d}
=

\beta_1+u_{1,d}.
$$

Tác động của giờ cao điểm đối với tài xế $d$ là:

$$
\beta_{2,d}
=

\beta_2+u_{2,d}.
$$

Không nên huấn luyện một mô hình hồi quy hoàn toàn riêng biệt cho từng tài xế nếu mỗi tài xế chỉ có ít chuyến xe.

Mixed-effects model phù hợp hơn vì nó sử dụng thông tin chung của toàn bộ tài xế và vẫn cho phép mỗi tài xế có hệ số riêng.

---

## 10. Kiểm tra feature nào có ảnh hưởng

Để xác định một feature có ảnh hưởng đến $\Delta P_i$ hay không, có thể sử dụng các tiêu chí sau:

1. Dấu và độ lớn của hệ số ước lượng.
2. Khoảng tin cậy $95%$ của hệ số.
3. $p$-value của hệ số.
4. Adjusted $R^2$.
5. Mean Absolute Error trên tập test.
6. Root Mean Squared Error trên tập test.
7. Akaike Information Criterion.
8. Bayesian Information Criterion.

Một feature có bằng chứng ảnh hưởng nếu:

- Hệ số của feature có độ lớn đáng kể.
- Khoảng tin cậy của hệ số không chứa $0$.
- $p$-value nhỏ hơn mức ý nghĩa được lựa chọn.
- Mô hình mở rộng có sai số dự đoán thấp hơn trên tập test.
- Adjusted $R^2$ tăng sau khi thêm feature.

Có thể kiểm định:

$$
H_0:
\gamma_j^{O}=0
\quad
\text{với mọi }j,
$$

để kiểm tra ảnh hưởng của khu vực xuất phát.

Tương tự:

$$
H_0:
\gamma_j^{G}=0
\quad
\text{với mọi }j,
$$

để kiểm tra ảnh hưởng của khu vực điểm đích.

Đối với driver effect, có thể kiểm tra:

$$
H_0:
\sigma_{\text{driver}}^2=0.
$$

Nếu bác bỏ giả thuyết trên, có bằng chứng cho thấy sự khác biệt giữa các tài xế ảnh hưởng đến $\Delta P_i$.

---

## 11. Chia train-test data

Có thể chia dữ liệu thành:

$$
80%
\text{ training data},
$$

và:

$$
20%
\text{ testing data}.
$$

Nếu dữ liệu có thứ tự thời gian, nên sử dụng các chuyến xe xảy ra trước làm tập train và các chuyến xe xảy ra sau làm tập test.

Ví dụ:

$$
\mathcal{D}_{\text{train}}
=

\left{
i:
t_i
\leq
t_{0.8}
\right},
$$

và:

$$
\mathcal{D}_{\text{test}}
=

\left{
i:
t_i

>

t_{0.8}
\right}.
$$

Cách chia theo thời gian giúp mô phỏng đúng hơn tình huống sử dụng dữ liệu lịch sử để dự đoán các chuyến xe trong tương lai.

---

## 12. Đánh giá mô hình

Mean Absolute Error được tính bằng:

$$
\operatorname{MAE}
=

\frac{1}{n_{\text{test}}}
\sum_{i\in\mathcal{D}_{\text{test}}}
\left|
\Delta P_i
-

\widehat{\Delta P}_i
\right|.
$$

Root Mean Squared Error được tính bằng:

$$
\operatorname{RMSE}
=

\sqrt{
\frac{1}{n_{\text{test}}}
\sum_{i\in\mathcal{D}_{\text{test}}}
\left(
\Delta P_i
-

\widehat{\Delta P}_i
\right)^2
}.
$$

Hệ số xác định được tính bằng:

$$
R^2
=

1-

\frac{
\sum_{i=1}^{n}
\left(
\Delta P_i
-

\widehat{\Delta P}_i
\right)^2
}{
\sum_{i=1}^{n}
\left(
\Delta P_i
-

\overline{\Delta P}
\right)^2
}.
$$

Nên so sánh các mô hình:

$$
\mathcal{M}_0,
\qquad
\mathcal{M}_1,
\qquad
\mathcal{M}_2.
$$

Nếu $\mathcal{M}_1$ hoặc $\mathcal{M}_2$ có MAE và RMSE thấp hơn đáng kể so với $\mathcal{M}_0$, điều đó cho thấy có các feature khác ngoài mưa và giờ cao điểm ảnh hưởng đến giá.

---

## 13. Trực quan hóa dữ liệu

### 13.1. Lưu ý về đường hồi quy

Vì $R_i$ và $H_i$ đều là biến nhị phân, mô hình cơ sở không tạo ra một đường hồi quy liên tục thông thường.

Thay vào đó, mô hình tạo ra bốn fitted values tương ứng với bốn nhóm thời tiết và thời gian.

Đường hồi quy cũng không bắt buộc đi qua tất cả các điểm dữ liệu. Nó được chọn để tối thiểu hóa tổng bình phương phần dư:

$$
\min_{\boldsymbol{\beta}}
\sum_{i=1}^{n}
\left(
\Delta P_i
-

\widehat{\Delta P}_i
\right)^2.
$$

Để biểu diễn bằng một đường liên tục, có thể bổ sung một feature liên tục, chẳng hạn khoảng cách $D_i$, vào mô hình minh họa:

$$
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\beta_3D_i
+
\varepsilon_i.
$$

Khi cố định:

$$
R_i=0
$$

và:

$$
H_i=0,
$$

đường fitted theo khoảng cách là:

$$
\widehat{\Delta P}_i
=

\widehat{\beta}_0
+
\widehat{\beta}_3D_i.
$$

### 13.2. Scatter plot và fitted line

Để vẽ scatter plot, có thể sử dụng:

- Trục $x$: khoảng cách $D_i$.
- Trục $y$: phần chênh lệch giá $\Delta P_i$.
- Các điểm: từng chuyến xe.
- Đường fitted: giá trị dự đoán từ mô hình hồi quy.

Ví dụ, đường fitted có dạng:

$$
\widehat{\Delta P}
=

2.1+0.9D.
$$

Các hệ số trên chỉ mang tính minh họa. Sau khi huấn luyện mô hình, cần thay bằng các hệ số ước lượng thực tế.

---

## 14. Biểu đồ actual versus predicted

Một biểu đồ phù hợp hơn đối với mô hình có nhiều feature là so sánh giá trị thực tế và giá trị dự đoán:

$$
\Delta P_i
\quad
\text{và}
\quad
\widehat{\Delta P}_i.
$$

Trong biểu đồ:

- Trục $x$ là giá trị dự đoán $\widehat{\Delta P}_i$.
- Trục $y$ là giá trị thực tế $\Delta P_i$.
- Mỗi điểm đại diện cho một chuyến xe.
- Đường tham chiếu là:

$$
y=x.
$$

Nếu mô hình dự đoán tốt, các điểm sẽ nằm gần đường $y=x$.

---

## 15. Biểu đồ fitted values theo bốn nhóm

Đối với mô hình cơ sở chỉ gồm $R_i$ và $H_i$, có thể tính bốn fitted values:

$$
\widehat{\mu}_{00}
=

\widehat{\beta}_0,
$$

$$
\widehat{\mu}_{10}
=

\widehat{\beta}_0
+
\widehat{\beta}_1,
$$

$$
\widehat{\mu}_{01}
=

\widehat{\beta}_0
+
\widehat{\beta}_2,
$$

và:

$$
\widehat{\mu}_{11}
=

\widehat{\beta}_0
+
\widehat{\beta}_1
+
\widehat{\beta}_2.
$$

Bốn nhóm tương ứng là:

| Nhóm | Điều kiện                  | Fitted value                                            |
| ---- | -------------------------- | ------------------------------------------------------- |
| 1    | Không mưa, giờ bình thường | $\widehat{\beta}_0$                                     |
| 2    | Có mưa, giờ bình thường    | $\widehat{\beta}_0+\widehat{\beta}_1$                   |
| 3    | Không mưa, giờ cao điểm    | $\widehat{\beta}_0+\widehat{\beta}_2$                   |
| 4    | Có mưa, giờ cao điểm       | $\widehat{\beta}_0+\widehat{\beta}_1+\widehat{\beta}_2$ |

Có thể vẽ các fitted values bằng bar chart hoặc profile plot.

Đường nối giữa các nhóm chỉ giúp quan sát sự thay đổi. Đây không phải là một đường hồi quy liên tục theo biến số thực.

---

## 16. Kế hoạch thực nghiệm

Quy trình thực nghiệm được đề xuất như sau:

1. Tạo $5{,}000$ chuyến xe synthetic và $50$ tài xế.
2. Sinh thời điểm, thời tiết, điểm xuất phát và điểm đích cho mỗi chuyến.
3. Tính khoảng cách từ tọa độ điểm xuất phát và điểm đích.
4. Tính thời gian di chuyển từ khoảng cách, mưa và giờ cao điểm.
5. Tính giá cơ sở từ khoảng cách và thời gian di chuyển.
6. Sinh driver effect, origin-zone effect và destination-zone effect.
7. Sinh $\Delta P_i$ từ các hệ số thật đã được lựa chọn.
8. Tính giá cuối cùng $P_i$.
9. Kiểm tra số chuyến của mỗi tài xế trong bốn nhóm thời tiết và thời gian.
10. Chia dữ liệu thành tập train và tập test.
11. Huấn luyện mô hình cơ sở $\mathcal{M}_0$.
12. Bổ sung khu vực xuất phát và điểm đích để tạo mô hình $\mathcal{M}_1$.
13. Bổ sung driver effect để tạo mô hình $\mathcal{M}_2$.
14. So sánh các hệ số ước lượng với hệ số thật được sử dụng để sinh dữ liệu.
15. So sánh MAE, RMSE, adjusted $R^2$, AIC và BIC giữa các mô hình.
16. Vẽ fitted values cho bốn nhóm mưa và giờ cao điểm.
17. Vẽ biểu đồ actual versus predicted.
18. Nếu bổ sung feature liên tục, vẽ scatter plot và fitted regression line.

---

## 17. Kết quả kỳ vọng

Nếu quy trình tạo dữ liệu và ước lượng mô hình được thực hiện đúng, các hệ số ước lượng phải gần với hệ số thật:

$$
\widehat{\beta}_0
\approx
\beta_0,
$$

$$
\widehat{\beta}_1
\approx
\beta_1,
$$

và:

$$
\widehat{\beta}_2
\approx
\beta_2.
$$

Nếu mô hình cơ sở bỏ qua location effect hoặc driver effect, hệ số của mưa và giờ cao điểm có thể bị sai lệch.

Mô hình mở rộng được kỳ vọng có:

$$
\operatorname{RMSE}_{\mathcal{M}_2}
<
\operatorname{RMSE}_{\mathcal{M}_1}
<
\operatorname{RMSE}_{\mathcal{M}_0}.
$$

Tuy nhiên, điều này chỉ xảy ra khi các feature bổ sung thực sự có ảnh hưởng và mô hình không bị overfitting.

---

## 18. Kết luận

Mô hình cơ sở:

$$
\Delta P_i
=

\beta_0
+
\beta_1R_i
+
\beta_2H_i
+
\varepsilon_i
$$

được sử dụng để đo tác động trung bình của mưa và giờ cao điểm đến phần chênh lệch giá.

Sau đó, mô hình được mở rộng bằng khu vực xuất phát, khu vực điểm đích và driver effect.

Để xác định ảnh hưởng của từng tài xế, mixed-effects model phù hợp hơn việc chạy một mô hình hoàn toàn riêng cho từng tài xế, đặc biệt khi số chuyến của mỗi tài xế không lớn.

Do hai feature chính $R_i$ và $H_i$ đều là biến nhị phân, mô hình cơ sở tạo ra bốn fitted values thay vì một đường hồi quy liên tục.

Để vẽ một fitted regression line, cần bổ sung một feature liên tục như khoảng cách, thời gian di chuyển hoặc tỷ lệ cung cầu.
