import numpy as np
import pandas as pd


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

    # Điều chỉnh bỏ phiếu của các class thiếu số phiếu
    for i in range(n_classes):
        if np.sum(votes[:, i]) == 0:
            votes[:, i] += 1  # Thêm 1 phiếu để tránh trường hợp bỏ trống hoàn toàn

    return np.argmax(votes, axis=1)


if __name__ == "__main__":
    # Load dữ liệu test
    test_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/test_clean_standardized.csv'
    test_data = pd.read_csv(test_path)
    X_test = test_data.values

    # Load model đã train
    svm_models = np.load('model.npy', allow_pickle=True).item()

    # Dự đoán
    y_pred = predict_ovo(X_test, svm_models)

    # Vẽ kết quả cho từng cặp classes
    class_pairs = list(svm_models.keys())

    # Lưu kết quả dự đoán
    test_data['predicted_price_range'] = y_pred
    test_data.to_csv('C:/Users/Admin/Documents/0TL/ml/BTL/model_prediction_result.csv', index=False)
    print("Đã dự đoán xong và lưu kết quả!")
