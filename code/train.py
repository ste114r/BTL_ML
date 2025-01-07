# import numpy as np
# import pandas as pd
#
# # Tải dữ liệu huấn luyện
# file_path_train = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train 100 standardized.csv'
# data_train = pd.read_csv(file_path_train)
#
# # Chọn đặc trưng (features) và nhãn (target)
# X_train = data_train.drop(columns=['price_range']).values  # Bỏ cột 'price_range' để lấy các đặc trưng
# y_train = data_train['price_range'].values  # Nhãn cần phân loại
#
# # Chuẩn hóa nhãn (-1, 1) để phù hợp với SVM (giả định bài toán nhị phân)
# y_train = np.where(y_train == 0, -1, 1)
#
#
# # Hàm huấn luyện SVM bằng cách tối ưu hóa bài toán bậc hai
# def train_svm(X, y, C=1.0, max_iter=1000, learning_rate=0.001):
#     n_samples, n_features = X.shape
#     w = np.zeros(n_features)  # Trọng số ban đầu
#     b = 0  # Hệ số bias ban đầu
#
#     for _ in range(max_iter):
#         for idx, x_i in enumerate(X):
#             condition = y[idx] * (np.dot(x_i, w) + b) >= 1
#             if condition:
#                 w -= learning_rate * (2 * C * w)  # Chỉ cập nhật regularization
#             else:
#                 w -= learning_rate * (2 * C * w - np.dot(x_i, y[idx]))
#                 b -= learning_rate * y[idx]
#     return w, b
#
#
# # Huấn luyện mô hình
# w, b = train_svm(X_train, y_train)
#
# # Lưu trọng số và bias
# np.save('svm_weights.npy', w)
# np.save('svm_bias.npy', b)
# print(f"Trọng số (w): {w}")
# print(f"Bias (b): {b}")
