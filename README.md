Đề tài: Phân loại giá điện thoại	  

Thuật toán sử dụng: SVM (One-vs-One) 

Các bước chạy:

1. Xử lý các data set (train_old, test_old) => Chạy IQR.py (đổi đường dẫn thư mục trong code cho mỗi file)
   
2. Chuẩn hóa các data set sau khi đã xử lý (train_clean, data_clean) => Chạy data_standardization.py (đổi đường dẫn thư mục trong code cho mỗi file

3. Train model bằng train set đã xử lý và chuẩn hóa (train_clean_standardized) => Chạy train.py => Cho ra file model.npy 

4. Thử model đã train dùng test set (test_clean_standardized) => Chạy test.py => Cho ra kết quả ở file 'model_prediction_result.csv'
