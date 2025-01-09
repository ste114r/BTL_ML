# Đề tài: Phân loại Giá Điện Thoại

## Thuật toán sử dụng
- **SVM (One-vs-One)**

## Các bước chạy

1. **Xử lý các data set**
   - Data sets: `train_old`, `test_old`
   - Chạy file `IQR.py`
   - *Lưu ý:* Đổi đường dẫn thư mục trong code cho mỗi file.

2. **Chuẩn hóa các data set sau khi xử lý**
   - Data sets: `train_clean`, `data_clean`
   - Chạy file `data_standardization.py`
   - *Lưu ý:* Đổi đường dẫn thư mục trong code cho mỗi file.

3. **Train model**
   - Dùng data set đã xử lý và chuẩn hóa: `train_clean_standardized`
   - Chạy file `train.py`
   - Kết quả: Tạo file `model.npy`.

4. **Thử model đã train**
   - Dùng test set: `test_clean_standardized`
   - Chạy file `test.py`
   - Kết quả: File `model_prediction_result.csv`.

## Chú ý
- Đảm bảo các đường dẫn được thay đổi đúng trong các script trước khi chạy.
