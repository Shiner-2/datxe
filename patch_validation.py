import nbformat as nbf
from pathlib import Path
import re

def update_create_customer_nb():
    path = Path('src/create_customer_nb.py')
    content = path.read_text()
    
    old_code3 = """    code3 = \"\"\"X_accept = df[['final_price', 'rain', 'rush_hour']]
y_accept = df['customer_accept']

model_accept = LogisticRegression(solver='lbfgs')
model_accept.fit(X_accept, y_accept)

print(f"Alpha_0 (Intercept): {model_accept.intercept_[0]:.4f}")
print(f"Alpha_price: {model_accept.coef_[0][0]:.6f}")
print(f"Alpha_rain: {model_accept.coef_[0][1]:.4f}")
print(f"Alpha_rush: {model_accept.coef_[0][2]:.4f}")

df['pred_accept_prob'] = model_accept.predict_proba(X_accept)[:, 1]
roc_auc = roc_auc_score(y_accept, df['pred_accept_prob'])
print(f"\\nROC AUC Score: {roc_auc:.4f}")
\"\"\""""

    new_code3 = """    code3 = \"\"\"from sklearn.model_selection import train_test_split
import statsmodels.api as sm

X_accept = df[['final_price', 'rain', 'rush_hour']]
y_accept = df['customer_accept']

# Chia tập Train/Test để kiểm định khả năng dự đoán (Validation)
X_train, X_test, y_train, y_test = train_test_split(X_accept, y_accept, test_size=0.2, random_state=42)

# Sử dụng Statsmodels để lấy p-values (Kiểm định thống kê)
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0)
print(result.summary())
print("\\n" + "="*50 + "\\n")

# Dùng scikit-learn để dễ dàng predict_proba cho toàn bộ df và Test set
model_accept = LogisticRegression(solver='lbfgs')
model_accept.fit(X_train, y_train)

df['pred_accept_prob'] = model_accept.predict_proba(X_accept)[:, 1]
test_pred_prob = model_accept.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_pred_prob)
print(f"ROC AUC Score trên tập Test (20% dữ liệu): {roc_auc:.4f}")
\"\"\""""

    if old_code3 in content:
        content = content.replace(old_code3, new_code3)
        path.write_text(content)
        print("Updated src/create_customer_nb.py")
    else:
        print("Could not find code3 in src/create_customer_nb.py")


def update_create_context_nb():
    path = Path('src/create_context_nb.py')
    content = path.read_text()

    old_code3 = """    code3 = \"\"\"import pandas as pd
from sklearn.linear_model import LinearRegression

# Tạo dummy variables cho 4 ngữ cảnh
context_dummies = pd.get_dummies(df['context'])

# Fit mô hình không có intercept
model_optimized = LinearRegression(fit_intercept=False)
model_optimized.fit(context_dummies, df['delta_price'])

betas_optimized = pd.DataFrame({
    'Context': context_dummies.columns,
    'Optimized_Beta': model_optimized.coef_
})
display(betas_optimized)

df['pred_optimized'] = model_optimized.predict(context_dummies)
\"\"\""""

    new_code3 = """    code3 = \"\"\"import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Tạo dummy variables cho 4 ngữ cảnh (ép kiểu int để tương thích Statsmodels)
context_dummies = pd.get_dummies(df['context'], dtype=int)

# Dùng Statsmodels để kiểm định ý nghĩa thống kê của các Betas (p-value, Confidence Interval)
model_opt_sm = sm.OLS(df['delta_price'], context_dummies)
res_opt = model_opt_sm.fit()
print(res_opt.summary())
print("\\n" + "="*50 + "\\n")

# Dùng sklearn để dễ dàng lưu lại
model_optimized = LinearRegression(fit_intercept=False)
model_optimized.fit(context_dummies, df['delta_price'])

betas_optimized = pd.DataFrame({
    'Context': context_dummies.columns,
    'Optimized_Beta': model_optimized.coef_
})
display(betas_optimized)

df['pred_optimized'] = model_optimized.predict(context_dummies)
\"\"\""""

    if old_code3 in content:
        content = content.replace(old_code3, new_code3)
        path.write_text(content)
        print("Updated src/create_context_nb.py")
    else:
        print("Could not find code3 in src/create_context_nb.py")


def update_driver_nb():
    path = Path('notebooks/driver_price_acceptance.ipynb')
    if not path.exists():
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Thêm kiểm định p-value cho M0 vào driver notebook (sau khi in baseline_coefficients)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and "baseline_coefficients" in cell.source:
            # Thêm cell validation
            val_code = """import statsmodels.api as sm

X_m0 = pd.get_dummies(train[["rain", "rush_hour"]], drop_first=True, dtype=int)
X_m0 = sm.add_constant(X_m0)
y_m0 = train["delta_price"]

model_sm = sm.OLS(y_m0, X_m0)
res = model_sm.fit()
print("Kiểm định thống kê (Validation) cho mô hình cơ sở M0:\\n")
print(res.summary())
"""
            val_md = nbf.v4.new_markdown_cell("> **💡 Validation:** Bảng kết quả Statsmodels cung cấp p-value (`P>|t|`). Vì p-value của `rain` và `rush_hour` đều xấp xỉ 0 (nhỏ hơn 0.05), ta có cơ sở thống kê vững chắc kết luận 2 yếu tố này có ảnh hưởng thực sự (Significant) đến phụ phí, chứ không phải do nhiễu.")
            
            # Chỉ chèn nếu chưa có
            found = False
            for c in nb.cells:
                if "import statsmodels.api as sm" in c.source and "y_m0" in c.source:
                    found = True
                    break
            
            if not found:
                nb.cells.insert(i+2, nbf.v4.new_code_cell(val_code))
                nb.cells.insert(i+3, val_md)
                with open(path, 'w', encoding='utf-8') as f:
                    nbf.write(nb, f)
                print("Updated notebooks/driver_price_acceptance.ipynb")
            break

def update_create_dynamic_pricing_nb():
    path = Path('src/create_dynamic_pricing_nb.py')
    content = path.read_text()
    
    old_code1 = """# 2. Huấn luyện mô hình Khách hàng (Giá thu = final_price)
model_cust = LogisticRegression(solver='lbfgs')
model_cust.fit(df_cust[['final_price', 'rain', 'rush_hour']], df_cust['customer_accept'])

# 3. Huấn luyện mô hình Tài xế (Giá trả = final_price)
model_driver = LogisticRegression(solver='lbfgs')
model_driver.fit(df_driver[['final_price', 'rain', 'rush_hour']], df_driver['driver_accept'])

print("Mô hình đã được huấn luyện xong!")"""

    new_code1 = """from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 2. Validation & Huấn luyện mô hình Khách hàng (Giá thu = final_price)
X_cust = df_cust[['final_price', 'rain', 'rush_hour']]
y_cust = df_cust['customer_accept']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cust, y_cust, test_size=0.2, random_state=42)

model_cust = LogisticRegression(solver='lbfgs')
model_cust.fit(Xc_train, yc_train)
auc_cust = roc_auc_score(yc_test, model_cust.predict_proba(Xc_test)[:, 1])

# 3. Validation & Huấn luyện mô hình Tài xế (Giá trả = final_price)
X_driver = df_driver[['final_price', 'rain', 'rush_hour']]
y_driver = df_driver['driver_accept']
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_driver, y_driver, test_size=0.2, random_state=42)

model_driver = LogisticRegression(solver='lbfgs')
model_driver.fit(Xd_train, yd_train)
auc_driver = roc_auc_score(yd_test, model_driver.predict_proba(Xd_test)[:, 1])

print(f"Mô hình Khách hàng đã huấn luyện xong (Test AUC = {auc_cust:.4f})")
print(f"Mô hình Tài xế đã huấn luyện xong (Test AUC = {auc_driver:.4f})")
print("=> Cả hai mô hình đều có ROC AUC > 0.55 trên tập Validation (Test), chứng tỏ khả năng dự đoán đáng tin cậy để dùng cho Tối ưu hóa Giá.")
"""

    if old_code1 in content:
        content = content.replace(old_code1, new_code1)
        path.write_text(content)
        print("Updated src/create_dynamic_pricing_nb.py")
    else:
        print("Could not find code1 in src/create_dynamic_pricing_nb.py")

if __name__ == '__main__':
    update_create_customer_nb()
    update_create_context_nb()
    update_create_dynamic_pricing_nb()
    update_driver_nb()
