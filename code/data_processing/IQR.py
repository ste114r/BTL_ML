import pandas as pd

df = pd.read_csv('C:/Users/Admin/Documents/0TL/ml/BTL/data/test_old.csv')
# Lọc các cột kiểu số, nhưng bỏ qua cột dạng binary (có giá trị 0 hoặc 1)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Lọc ra các cột dạng binary (giá trị chỉ có 0 và 1)
binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]

# Lấy các cột liên tục (loại bỏ binary cols)
continuous_cols = [col for col in numeric_cols if col not in binary_cols]

# Tạo một bản sao của DataFrame
df_cleaned = df.copy()

# Xử lý ngoại lai cho các cột liên tục
for col in continuous_cols:
    # Tính Q1, Q3 và IQR
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1

    # Ngưỡng giới hạn ngoại lai
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Thay thế giá trị ngoại lai nhỏ hơn lower_bound và lớn hơn upper_bound
    df_cleaned[col] = df_cleaned[col].apply(lambda x: lower_bound if x < lower_bound else x)
    df_cleaned[col] = df_cleaned[col].apply(lambda x: upper_bound if x > upper_bound else x)

# Xuất ra file CSV mới
df_cleaned.to_csv('C:/Users/Admin/Documents/0TL/ml/BTL/data/test_clean.csv', index=False)

print("Dữ liệu đã được xử lý và lưu vào file.")
