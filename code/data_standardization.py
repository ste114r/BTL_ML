import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ file CSV
file_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train_clean.csv'
data = pd.read_csv(file_path)

# Các cột liên tục mà bạn muốn chuẩn hóa
continuous_columns = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt',
                      'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

# Tạo đối tượng chuẩn hóa
scaler = StandardScaler()

# Chuẩn hóa các cột liên tục đã xác định
data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

# Lưu dữ liệu đã chuẩn hóa vào file mới
standardized_file_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train_clean_standardized.csv'
data.to_csv(standardized_file_path, index=False)

# Hiển thị vài dòng đầu của dữ liệu đã chuẩn hóa
print(data.head())
