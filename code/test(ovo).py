# test_ovo_svm.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')


def predict_ovo(X, models):
    """
    Dự đoán sử dụng mô hình OvO SVM
    
    # Thông số:
    X: ma trận features cần dự đoán
    models: dictionary chứa các mô hình đã train
    """
    n_classes = len(np.unique([k for pair in models.keys() for k in pair]))
    votes = np.zeros((X.shape[0], n_classes))

    # Voting từ tất cả các classifier
    for (class_1, class_2), (w, b) in models.items():
        predictions = np.dot(X, w) + b
        votes[:, class_1] += (predictions >= 0).astype(int)
        votes[:, class_2] += (predictions < 0).astype(int)

    return np.argmax(votes, axis=1)


def plot_test_results(X, y_pred, models, class_pair):
    """
    Vẽ kết quả test cho một cặp classes
    
    # Thông số:
    X: dữ liệu test
    y_pred: nhãn dự đoán
    models: dictionary chứa các mô hình
    class_pair: tuple chứa cặp classes đang xét
    """
    class_1, class_2 = class_pair
    w, b = models[class_pair]


    plt.figure(figsize=(10, 8))

    # Vẽ hyperplane
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k-', label=f'Hyperplane {class_1} vs {class_2}')

    # Vẽ điểm dữ liệu test
    mask = (y_pred == class_1) | (y_pred == class_2)
    if np.any(y_pred == class_1):
        plt.scatter(X[y_pred == class_1][:, 0], X[y_pred == class_1][:, 1],
                    label=f'Predicted Class {class_1}', alpha=0.7)
    if np.any(y_pred == class_2):
        plt.scatter(X[y_pred == class_2][:, 0], X[y_pred == class_2][:, 1],
                    label=f'Predicted Class {class_2}', alpha=0.7)

    plt.title(f'Test Data và Dự Đoán: Class {class_1} vs {class_2}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'ovo_svm_test_{class_1}_vs_{class_2}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Load dữ liệu test
    test_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/test 100 standardized.csv'
    test_data = pd.read_csv(test_path)
    X_test = test_data.values

    # Load model đã train
    svm_models = np.load('ovo_svm_models.npy', allow_pickle=True).item()

    # Dự đoán
    y_pred = predict_ovo(X_test, svm_models)

    # Vẽ kết quả cho từng cặp classes
    class_pairs = list(svm_models.keys())
    for class_pair in class_pairs:
        plot_test_results(X_test, y_pred, svm_models, class_pair)

    # Lưu kết quả dự đoán
    test_data['predicted_price_range'] = y_pred
    test_data.to_csv('test_predictions_ovo_svm.csv', index=False)
    print("Đã dự đoán xong và lưu kết quả!")
