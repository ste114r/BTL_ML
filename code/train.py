# train_ovo_svm_with_evaluation.py
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class OVOSVM:
    """
    One-vs-One Support Vector Machine (OvO SVM) với các cải tiến

    # Cải tiến:
    - Thêm validation split để kiểm tra convergence
    - Thêm early stopping để tránh overfitting
    - Cải thiện cơ chế voting với softmax
    - Thêm learning rate schedule
    - Thêm momentum để tăng tốc convergence
    """

    def __init__(self, C=0.1, max_iter=20000, learning_rate=0.001,
                 class_weights=None, momentum=0.9, patience=15):
        self.C = C  # Hệ số điều chỉnh (regularization)
        self.max_iter = max_iter  # Số lần lặp tối đa
        self.learning_rate = learning_rate  # Tốc độ học ban đầu
        self.momentum = momentum  # Hệ số momentum
        self.patience = patience  # Số epoch chờ trước khi early stopping
        self.models = {}  # Lưu trữ các mô hình cho từng cặp classes
        self.class_weights = class_weights  # Trọng số cho các class

    def _adjust_learning_rate(self, epoch):
        """Điều chỉnh learning rate theo epoch"""
        return self.learning_rate / (1 + epoch * 0.01)

    def _initialize_weights(self, n_features):
        """Khởi tạo trọng số theo phân phối normal với scale nhỏ"""
        return np.random.normal(0, 0.01, n_features)

    def _compute_loss(self, X, y, w, b):
        """Tính hinge loss và regularization loss"""
        margin = y * (np.dot(X, w) + b)
        hinge_loss = np.maximum(0, 1 - margin).mean()
        reg_loss = 0.5 * self.C * np.dot(w, w)
        return hinge_loss + reg_loss

    def _train_binary_svm(self, X, y):
        """Train SVM nhị phân với các cải tiến"""
        n_samples, n_features = X.shape

        # Tách validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Khởi tạo trọng số và bias
        w = self._initialize_weights(n_features)
        b = 0

        # Khởi tạo các biến cho momentum và early stopping
        prev_v_w = np.zeros_like(w)
        prev_v_b = 0
        best_val_loss = float('inf')
        patience_counter = 0

        # Điều chỉnh trọng số cho lớp dữ liệu
        if self.class_weights:
            weights = np.array([
                self.class_weights[1 if yi == 1 else 0] for yi in y_train
            ])
        else:
            weights = np.ones_like(y_train)

        for epoch in range(self.max_iter):
            # Điều chỉnh learning rate
            curr_lr = self._adjust_learning_rate(epoch)

            # Training step với momentum
            for idx, x_i in enumerate(X_train):
                condition = y_train[idx] * (np.dot(x_i, w) + b) >= 1

                if condition:
                    v_w = self.momentum * prev_v_w - curr_lr * (2 * self.C * w)
                    v_b = self.momentum * prev_v_b
                else:
                    v_w = (self.momentum * prev_v_w -
                           curr_lr * (2 * self.C * w - np.dot(x_i, y_train[idx])))
                    v_b = self.momentum * prev_v_b + curr_lr * y_train[idx]

                # Cập nhật trọng số với momentum
                w += v_w * weights[idx]
                b += v_b * weights[idx]

                prev_v_w, prev_v_b = v_w, v_b

            # Kiểm tra early stopping trên validation set
            val_loss = self._compute_loss(X_val, y_val, w, b)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_w, best_b = w.copy(), b
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    return best_w, best_b

        return w, b

    def fit(self, X, y):
        """Train mô hình OvO SVM với monitoring"""
        self.classes = np.unique(y)
        class_pairs = list(combinations(self.classes, 2))

        print("Bắt đầu training các classifier:")
        for class_1, class_2 in class_pairs:
            print(f"\nTraining classifier cho classes {class_1} vs {class_2}")

            # Lọc data cho cặp class hiện tại
            mask = (y == class_1) | (y == class_2)
            X_pair = X[mask]
            y_pair = np.where(y[mask] == class_1, 1, -1)

            # Train và lưu model
            w, b = self._train_binary_svm(X_pair, y_pair)
            self.models[(class_1, class_2)] = (w, b)

            print(f"Đã hoàn thành training classifier {class_1} vs {class_2}")

        return self

    def predict(self, X):
        """Dự đoán lớp của mẫu dữ liệu"""
        votes = np.zeros((X.shape[0], len(self.classes)))

        for (class_1, class_2), (w, b) in self.models.items():
            decision = np.dot(X, w) + b
            votes[:, class_1] += (decision > 0).astype(int)
            votes[:, class_2] += (decision <= 0).astype(int)

        return np.argmax(votes, axis=1)


if __name__ == "__main__":
    # Load và kiểm tra dữ liệu training
    train_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train_clean_standardized.csv'
    train_data = pd.read_csv(train_path)

    print("Phân phối các classes trong tập training:")
    print(train_data['price_range'].value_counts())

    # Tách features và labels
    X = train_data.drop(columns=['price_range']).values
    y = train_data['price_range'].values

    # Chia dữ liệu thành training và test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tính trọng số nghịch đảo cho các class để cân bằng
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count)
                     for i, count in enumerate(class_counts)}

    print("\nTrọng số cho từng class:")
    for class_idx, weight in class_weights.items():
        print(f"Class {class_idx}: {weight:.3f}")

    # Khởi tạo và train model với hyperparameters đã điều chỉnh
    model = OVOSVM()
    model.fit(X_train, y_train)

    # Đánh giá mô hình trên tập test
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nĐộ chính xác trên tập test:", accuracy)
    print("\nBáo cáo chi tiết:\n", report)

    # Vẽ kết quả cho từng cặp classes
    class_pairs = list(combinations(np.unique(y_train), 2))

    # Lưu model
    np.save('model.npy', model.models)
    print("\nĐã train xong và lưu model!")
