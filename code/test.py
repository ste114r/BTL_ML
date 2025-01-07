# import numpy as np
# import pandas as pd
#
# # Tải dữ liệu kiểm tra (không có cột price_range)
# file_path_test = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/test 100 standardized.csv'
# data_test = pd.read_csv(file_path_test)
#
# # Loại bỏ các cột không cần thiết để đảm bảo số đặc trưng khớp với tập huấn luyện
# # (giả sử tập huấn luyện không bao gồm cột 'price_range')
# X_test = data_test.drop(columns=['extra_column_if_exists'], errors='ignore').values  # Bỏ cột không cần thiết nếu có
#
# # Tải trọng số và bias đã huấn luyện
# w = np.load('svm_weights.npy')
# b = np.load('svm_bias.npy')
#
#
# # Hàm dự đoán
# def predict_svm(X, w, b):
#     linear_output = np.dot(X, w) + b
#     return np.where(linear_output >= 0, 1, -1)
#
#
# # Dự đoán nhãn cho tập kiểm tra
# y_pred = predict_svm(X_test, w, b)
#
# # Lưu kết quả dự đoán
# data_test['predicted_price_range'] = np.where(y_pred == -1, 0, 1)  # Chuyển từ -1 về 0 nếu cần
# output_path = 'test_predictions_manual_svm.csv'
# data_test.to_csv(output_path, index=False)
# print(f"Kết quả dự đoán đã được lưu tại {output_path}")
