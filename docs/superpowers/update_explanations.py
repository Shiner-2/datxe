import nbformat as nbf
from pathlib import Path

def update_driver_nb():
    nb_path = Path('notebooks/driver_price_acceptance.ipynb')
    if not nb_path.exists():
        return
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    new_cells = []
    for cell in nb.cells:
        new_cells.append(cell)
        if cell.cell_type == 'code':
            src = cell.source
            if "df.describe(" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Các lệnh trên hiển thị số chiều (dòng, cột) của dữ liệu, kiểu dữ liệu từng cột và bảng tóm tắt thống kê. Điều này giúp ta đảm bảo dữ liệu không có rác, các biến liên tục nằm trong dải hợp lý và các biến phân loại (`rain`, `rush_hour`) có tỷ lệ phân bố cân bằng."))
            elif "summary =" in src and "group_counts" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Bảng summary này kiểm tra mức độ đa dạng của dữ liệu (số lượng tài xế, khu vực) và quan trọng nhất là `min_driver_weather_time_group_count`. Việc đảm bảo số mẫu tối thiểu trong mỗi tổ hợp giúp cho kết quả hồi quy sau này vững (robust) và không bị sai lệch do thiếu dữ liệu."))
            elif "baseline_coefficients" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Mô hình cơ sở (M0) ước lượng `delta_price` dựa trên `rain` và `rush_hour`. Hệ số $\\beta_0$ tương ứng với mức phụ phí cơ bản khi thời tiết đẹp và không tắc đường. $\\beta_1$ và $\\beta_2$ ước lượng phần phụ phí tăng thêm. Ta thấy kết quả `estimate` cực kỳ sát với `true_value` (chính là giá trị trong thuật toán sinh dữ liệu), chứng tỏ Linear Regression trích xuất thông tin rất chuẩn xác."))
            elif "sns.barplot(data=fitted_groups" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Biểu đồ này trực quan hóa 4 mức phụ phí dự báo (fitted values). Rõ ràng ngữ cảnh 'Mua - cao điểm' là đắt đỏ nhất vì nó cộng dồn cả hai loại phụ phí $\\beta_1$ và $\\beta_2$. Điều này phản ánh tính cộng tính (additive) của mô hình hồi quy tuyến tính không có tương tác."))
            elif "metrics_df =" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Bước này đánh giá độ chính xác của 3 mô hình từ đơn giản (M0) đến phức tạp (M2). Khi ta đưa thêm các biến khu vực (origin/destination) và định danh tài xế (`driver_id`), ta thấy chỉ số **RMSE** giảm dần và **R2** tăng lên. Điều này ngụ ý rằng giá thực tế có bị ảnh hưởng bởi khu vực và thói quen/hiệu ứng riêng của từng bác tài."))
            elif "sns.barplot(data=metrics_df" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Biểu đồ cột minh họa rõ ràng sự giảm thiểu sai số RMSE. Mô hình M2 (có cả yếu tố vị trí và tài xế) là mô hình dự đoán giá chính xác nhất trên tập Test, do đó M2 sẽ được chọn làm mô hình tối ưu."))
            elif "sns.scatterplot(data=actual_vs_predicted" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Biểu đồ Thực tế (Actual) vs Dự đoán (Predicted). Đường đứt nét màu đỏ $y = x$ đại diện cho một dự đoán hoàn hảo vô sai số. Các chấm xanh quần tụ rất khít dọc theo đường đỏ chứng tỏ mô hình M2 bám sát dữ liệu thực; khoảng phân tán mờ nhạt xung quanh chính là sai số ngẫu nhiên (noise) không thể tránh khỏi ở đời thực."))
            elif "sns.scatterplot(data=sample" in src and "distance_km" in src:
                new_cells.append(nbf.v4.new_markdown_cell("> **💡 Giải thích kết quả:** Một bài test cực kỳ thú vị! Trong thiết kế giá, `distance_km` (khoảng cách) chỉ cấu thành nên giá cơ bản (`base_price`), chứ hoàn toàn không ảnh hưởng đến phần phụ phí (`delta_price`). \n> Khi vẽ Scatter plot giữa khoảng cách và phụ phí, ta thu được một đường **Fitted line nằm ngang**. Hệ số góc = 0 chứng minh phụ phí và khoảng cách độc lập hoàn toàn, xác nhận hệ thống sinh dữ liệu và mô hình phân tích hoạt động đúng triết lý thiết kế."))

    nb.cells = new_cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Updated driver_price_acceptance.ipynb")

def update_create_context_nb():
    path = Path('src/create_context_nb.py')
    content = path.read_text()
    
    # We will just patch the nb['cells'] array construction
    old_cells_array = """    nb['cells'] = [
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
        nbf.v4.new_markdown_cell(md4),
        nbf.v4.new_code_cell(code4)
    ]"""

    new_cells_array = """
    md1_exp = r\"> **💡 Giải thích kết quả:** Ta dùng hàm phân loại để biến đổi 2 cột `rain` và `rush_hour` thành 1 cột `context` duy nhất (có 4 giá trị rời rạc). Bảng kết quả `Actual_Delta_Price_Mean` chính là chân lý (ground truth) của mức phụ phí trung bình trong từng hoàn cảnh. Chúng ta sẽ dùng nó để kiểm chứng độ chính xác của các mô hình hồi quy bên dưới.\"
    md2_exp = r\"> **💡 Giải thích kết quả:** Mô hình M0 giải định mưa và tắc đường cộng dồn độc lập với nhau. Kết quả in ra $\\beta_0, \\beta_1, \\beta_2$ là các hệ số tuyến tính chung nhất. Lát nữa ta sẽ thấy, dùng mô hình độc lập này tuy đơn giản nhưng không thể bao quát hoàn hảo sự khác biệt riêng rẽ của 4 ngữ cảnh.\"
    md3_exp = r\"> **💡 Giải thích kết quả:** Ở mô hình này, chúng ta không hồi quy trên biến mưa/cao điểm nữa, mà hồi quy trực tiếp trên Dummy Variables của 4 ngữ cảnh. Bằng cách ép `fit_intercept=False`, OLS buộc phải tìm ra 4 giá trị $\\beta$ sao cho khớp nhất. Kết quả in ra cho thấy các $\\beta$ này **chính xác bằng** giá trị trung bình `Actual_Delta_Price_Mean` mà ta đã tính ở bước 1. Đây là cách toán học chỉ ra mức giá tối ưu nhất (về mặt cực tiểu hóa sai số MSE) cho từng tập khách hàng.\"
    md4_exp = r\"> **💡 Giải thích kết quả:** Bảng và biểu đồ cột tổng hợp lại tất cả. \n> - **Cột cam (Pred Independent)**: Dự báo bằng $\\beta$ độc lập cộng dồn, đôi lúc có sự chênh lệch nhỏ so với giá trị thực tế do nhiễu.\n> - **Cột xanh lá (Pred Optimized)**: Dự báo bằng 4 $\\beta$ tối ưu. Nó khớp $100\\%$ với cột xanh dương (Actual Mean).\n> - Nhìn xuống chỉ số **RMSE** ở cuối, ta xác nhận `RMSE (Optimized Betas)` thấp hơn hoặc bằng `RMSE (Independent Model)`. Điều này khẳng định nếu muốn tối ưu định giá cho từng ngữ cảnh, việc tinh chỉnh Beta theo tổ hợp (interaction) luôn mang lại độ chính xác cao hơn là tính độc lập.\"

    nb['cells'] = [
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md1_exp),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md2_exp),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
        nbf.v4.new_markdown_cell(md3_exp),
        nbf.v4.new_markdown_cell(md4),
        nbf.v4.new_code_cell(code4),
        nbf.v4.new_markdown_cell(md4_exp)
    ]"""
    
    if old_cells_array in content:
        content = content.replace(old_cells_array, new_cells_array)
        path.write_text(content)
        print("Updated src/create_context_nb.py")

def update_create_customer_nb():
    path = Path('src/create_customer_nb.py')
    content = path.read_text()
    
    old_cells_array = """    nb['cells'] = [
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
        nbf.v4.new_markdown_cell(md4),
        nbf.v4.new_code_cell(code4),
        nbf.v4.new_markdown_cell(md5),
        nbf.v4.new_code_cell(code5)
    ]"""

    new_cells_array = """
    md1_exp = r\"> **💡 Giải thích kết quả:** Các thống kê cơ bản này xác nhận tập dữ liệu mô phỏng hoạt động tốt. Cột `customer_accept` là biến nhị phân (0 hoặc 1), cho biết khách hàng có đồng ý giá đó không. Mức `customer_accept_prob` dao động từ cực thấp đến cực cao cho phép mô hình Logistic học được biên giới quyết định (decision boundary).\"
    md2_exp = r\"> **💡 Giải thích kết quả:** Dùng Linear Regression, ta bóc tách được cấu trúc giá hệ thống đang offer. Giá cuối (`final_price`) ở ngữ cảnh 4 (Mưa + Cao điểm) là đắt nhất vì nó bị gánh cả $\\beta_1$ và $\\beta_2$. Điều này hoàn toàn logic.\"
    md3_exp = r\"> **💡 Giải thích kết quả:** Trái tim của phân tích hành vi khách hàng nằm ở đây! \n> - Nhìn vào hệ số **Alpha_price**, ta thấy nó là một số **ÂM (-)**. Tức là Giá càng tăng $\\rightarrow$ Xác suất đồng ý đặt xe càng giảm. \n> - Hệ số **Alpha_rain** và **Alpha_rush** lại mang dấu **DƯƠNG (+)**. Tức là dù giá đắt, nhưng nếu trời đang mưa hoặc tắc đường, khách hàng trở nên khao khát có xe hơn (inelastic demand), họ sẵn sàng bỏ qua rào cản về giá để bấm nút Đặt xe. \n> - Chỉ số ROC AUC rất cao chứng minh mô hình phân loại (chấp nhận/từ chối) hoạt động vô cùng hiệu quả.\"
    md4_exp = r\"> **💡 Giải thích kết quả:** Bằng cách chia giá (`final_price`) thành 10 phân khúc (bins) bằng nhau từ rẻ nhất đến đắt nhất, ta thu được một \"Đường Cầu\" (Demand Curve) dốc xuống kinh điển. Ở các phân khúc giá rẻ (bên trái), tỷ lệ đặt xe là gần như tuyệt đối. Khi giá tịnh tiến dần sang phải (đắt hơn), người dùng bắt đầu do dự và tỷ lệ đồng ý sụp đổ.\"
    md5_exp = r\"> **💡 Giải thích kết quả:** Biểu đồ Dual-axis này mang lại insight đắt giá nhất cho vận hành kinh doanh: \n> - Ở ngữ cảnh 4 (Mưa + Cao điểm), cột giá (màu đỏ) cao vượt trội. \n> - Tuy nhiên, đường dây tỷ lệ đồng ý (màu xanh) lại **KHÔNG HỀ CHẠM ĐÁY**, thậm chí nó còn cao hơn ngữ cảnh 1 (bình thường, giá rẻ). \n> \n> **Kết luận kinh doanh**: Hành vi của khách hàng trong điều kiện thời tiết/giao thông xấu bị chi phối bởi \"sự cấp thiết\" (urgence) nhiều hơn là \"sự nhạy cảm về giá\". Đây là cơ sở vững chắc để nền tảng có thể mạnh dạn áp dụng Surge Pricing (phụ phí tăng giá) nhằm lôi kéo tài xế mà không sợ bị mất trắng khách hàng.\"

    nb['cells'] = [
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md1_exp),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md2_exp),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
        nbf.v4.new_markdown_cell(md3_exp),
        nbf.v4.new_markdown_cell(md4),
        nbf.v4.new_code_cell(code4),
        nbf.v4.new_markdown_cell(md4_exp),
        nbf.v4.new_markdown_cell(md5),
        nbf.v4.new_code_cell(code5),
        nbf.v4.new_markdown_cell(md5_exp)
    ]"""

    if old_cells_array in content:
        content = content.replace(old_cells_array, new_cells_array)
        path.write_text(content)
        print("Updated src/create_customer_nb.py")

if __name__ == '__main__':
    update_driver_nb()
    update_create_context_nb()
    update_create_customer_nb()
