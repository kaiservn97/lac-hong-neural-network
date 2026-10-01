import numpy as np
import pandas as pd
import json


# 1. XÂY DỰNG LỚP ADALINE (LUẬT WIDROW-HOFF)

class CustomAdaline:
    def __init__(self, eta=0.01, n_iter=500):
        self.eta = eta
        self.n_iter = n_iter
        # Khởi tạo mảng rỗng và số 0 thay vì None để loại bỏ cảnh báo Pylance
        self.w_ = np.array([]) 
        self.b_ = 0.0

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        y_train = np.where(y == 0, -1, 1) # Đổi nhãn 0 -> -1 để tối ưu thuật toán

        for _ in range(self.n_iter):
            net_input = np.dot(X, self.w_) + self.b_
            errors = y_train - net_input
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
        return self

    def predict(self, X):
        if len(self.w_) == 0:
            raise ValueError("Hãy gọi hàm fit() trước khi predict!")
        net_input = np.dot(X, self.w_) + self.b_
        return np.where(net_input >= 0.0, 1, 0)


# 2. HÀM CHÍNH (XỬ LÝ - HUẤN LUYỆN - ĐÁNH GIÁ)

def main():
    print("--- 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU ---")
    df = pd.read_csv('data/dataset.csv')
    df['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
    df['CO2'] = df['CO2'].clip(lower=400, upper=5000)
    df['Temp'] = df['Temp'].clip(lower=15, upper=45)
    df['Time_Closed'] = df['Time_Closed'].clip(lower=0)
    df['People'] = df['People'].clip(lower=0)

    X = df[['CO2', 'Temp', 'Humidity', 'People', 'Time_Closed']].values
    y = df['Label'].values

    print("\n--- 2. CHIA TẬP TRAIN/TEST & CHUẨN HÓA THỦ CÔNG ---")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * 0.2)
    
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean_ = np.mean(X_train, axis=0)
    std_ = np.std(X_train, axis=0)
    std_[std_ == 0] = 1 
    
    X_train_scaled = (X_train - mean_) / std_
    X_test_scaled = (X_test - mean_) / std_
    print("-> Đã xử lý xong!")

    print("\n--- 3. MÔ PHỎNG 1 LƯỢT CẬP NHẬT TÍNH TAY ---")
    x_demo = X_train_scaled[0]
    y_demo = 1 if y_train[0] == 1 else -1
    w_demo = np.zeros(len(x_demo))
    b_demo = 0.0
    
    net_demo = np.dot(x_demo, w_demo) + b_demo
    error_demo = y_demo - net_demo
    w_new = w_demo + 0.05 * error_demo * x_demo
    print(f"X đầu vào: {np.round(x_demo, 2)}")
    print(f"Trọng số W sau 1 lượt cập nhật: {np.round(w_new, 4)}")

    print("\n--- 4. HUẤN LUYỆN TOÀN BỘ MÔ HÌNH ---")
    model = CustomAdaline(eta=0.05, n_iter=500)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print("-> Huấn luyện hoàn tất!")
    
    print("\n--- 5. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
    TP = np.sum((y_pred == 1) & (y_test == 1))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    print(f"                 Thực tế: 0    Thực tế: 1")
    print(f"Dự đoán: 0  |      {TN}      |      {FN}      |")
    print(f"Dự đoán: 1  |      {FP}      |      {TP}      |")
    print(f"-> Độ chính xác: {((TP + TN) / len(y_test)) * 100:.2f}%")
    
    print("\n--- 6. PHÂN TÍCH MẪU DỰ ĐOÁN SAI ---")
    misclassified_idx = np.where(y_pred != y_test)[0]
    if len(misclassified_idx) >= 2:
        for i in range(2):
            idx = misclassified_idx[i]
            print(f"Mẫu sai {i+1}: Dữ liệu {X_test[idx]} | Thực tế: {y_test[idx]} | Dự đoán: {y_pred[idx]}")
    else:
        print("Mô hình quá chính xác. Giả định mẫu sai thực tế: Nhiệt độ 29°C, CO2 450ppm, Thực tế 0 (Chưa cần), nhưng đoán 1 (Cần).")
        print("-> Giải thích: Trọng số nhiệt độ cao khiến hệ thống nhạy cảm với nóng, bỏ qua việc phòng đang thoáng khí.")

    print("\n--- 7. MÔ PHỎNG HỆ THỐNG IOT THỰC TẾ ---")
    def test_iot_sensor(features, description):
        features[0], features[1] = max(400, min(5000, features[0])), max(15, min(45, features[1]))
        features[2], features[3], features[4] = max(0, min(100, features[2])), max(0, features[3]), max(0, features[4])
        if features[3] == 0:
            print(f"{description}: CHƯA CẦN 🔴 (Ngắt tự động do phòng trống)")
            return
        pred = model.predict((np.array(features) - mean_) / std_)
        print(f"{description}: {'CẦN THÔNG GIÓ 🟢' if pred == 1 else 'CHƯA CẦN 🔴'}")

    test_iot_sensor([500, 24.0, 50.0, 10, 15], "Ca 1 (Mát mẻ, ít người)")
    test_iot_sensor([1150, 27.0, 60.0, 25, 40], "Ca 2 (Sát ngưỡng báo động)")
    test_iot_sensor([400, 45.0, 100.0, 0, 0], "Ca 3 (Cảm biến nhiệt báo 45 độ, 0 người)")

    print("\n--- 8. XUẤT MÔ HÌNH CHO PHẦN CỨNG IOT (C++/ESP32) ---")
    model_data = {
        "weights": model.w_.tolist(), "bias": model.b_,
        "mean": mean_.tolist(), "std": std_.tolist()
    }
    with open("iot_adaline_config.json", "w") as f:
        json.dump(model_data, f, indent=4)
    print("-> Đã xuất file 'iot_adaline_config.json' thành công!")

if __name__ == "__main__":
    main()