# train_ovo_svm.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations

matplotlib.use('TkAgg')


class OVOSVM:
    """
    One-vs-One Support Vector Machine (OvO SVM)
    
    # Mô tả:
    - Implementation SVM sử dụng phương pháp One-vs-One
    - Mỗi classifier được train cho một cặp classes
    - Kết quả cuối được chọn bằng voting
    """

    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.001):
        self.C = C  # Hệ số điều chỉnh (regularization)
        self.max_iter = max_iter  # Số lần lặp tối đa
        self.learning_rate = learning_rate  # Tốc độ học
        self.models = {}  # Lưu trữ các mô hình cho từng cặp classes

    def _train_binary_svm(self, X, y):
        """Train SVM nhị phân cho một cặp classes"""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)  # Vector trọng số
        b = 0  # Độ lệch (bias)

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                # Cập nhật trọng số theo thuật toán gradient descent
                condition = y[idx] * (np.dot(x_i, w) + b) >= 1
                if condition:
                    w -= self.learning_rate * (2 * self.C * w)
                else:
                    w -= self.learning_rate * (2 * self.C * w - np.dot(x_i, y[idx]))
                    b -= self.learning_rate * y[idx]

        return w, b

    def fit(self, X, y):
        """Train mô hình OvO SVM"""
        self.classes = np.unique(y)
        class_pairs = list(combinations(self.classes, 2))

        # Train từng cặp classes
        for class_1, class_2 in class_pairs:
            # Lọc data cho cặp class hiện tại
            mask = (y == class_1) | (y == class_2)
            X_pair = X[mask]
            y_pair = np.where(y[mask] == class_1, 1, -1)

            # Train và lưu model
            w, b = self._train_binary_svm(X_pair, y_pair)
            self.models[(class_1, class_2)] = (w, b)

        return self


def plot_training_data(X, y, model, class_pair):
    """
    Vẽ dữ liệu training và hyperplane cho một cặp classes
    
    # Thông số:
    X: dữ liệu training
    y: nhãn
    model: mô hình đã train
    class_pair: tuple chứa cặp classes đang xét
    """
    class_1, class_2 = class_pair
    w, b = model.models[class_pair]

    plt.figure(figsize=(10, 8))

    # Vẽ hyperplane
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k-', label=f'Hyperplane {class_1} vs {class_2}')

    # Vẽ margin
    margin = 1 / np.sqrt(np.sum(w ** 2))
    y_vals_up = -(w[0] * x_vals + b + 1) / w[1]
    y_vals_down = -(w[0] * x_vals + b - 1) / w[1]
    plt.plot(x_vals, y_vals_up, 'k--', alpha=0.5)
    plt.plot(x_vals, y_vals_down, 'k--', alpha=0.5)

    # Vẽ điểm dữ liệu
    mask = (y == class_1) | (y == class_2)
    plt.scatter(X[y == class_1][:, 0], X[y == class_1][:, 1],
                label=f'Class {class_1}', alpha=0.7)
    plt.scatter(X[y == class_2][:, 0], X[y == class_2][:, 1],
                label=f'Class {class_2}', alpha=0.7)

    plt.title(f'Training Data và Hyperplane: Class {class_1} vs {class_2}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'ovo_svm_train_{class_1}_vs_{class_2}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Load dữ liệu training
    train_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train 100 standardized.csv'
    train_data = pd.read_csv(train_path)

    # Tách features và labels
    X_train = train_data.drop(columns=['price_range']).values
    y_train = train_data['price_range'].values

    # Khởi tạo và train model
    model = OVOSVM(C=1.0, max_iter=1000, learning_rate=0.001)
    model.fit(X_train, y_train)

    # Vẽ hyperplane cho từng cặp classes
    class_pairs = list(combinations(np.unique(y_train), 2))
    for class_pair in class_pairs:
        plot_training_data(X_train, y_train, model, class_pair)

    # Lưu model
    np.save('ovo_svm_models.npy', model.models)
    print("Đã train xong và lưu model!")
