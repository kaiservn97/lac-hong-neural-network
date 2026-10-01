# BUỔI 4 · TÌNH HUỐNG 07 — CẢNH BÁO THÔNG GIÓ PHÒNG HỌC
#
# Bối cảnh:
# Một lớp học muốn dùng cảm biến để cảnh báo khi cần mở cửa hoặc bật quạt thông gió.
#
# Nhãn cần dự đoán:
# 0 = chưa cần thông gió
# 1 = cần thông gió
#
# Feature có thể cân nhắc:
# nồng độ CO2, nhiệt độ, độ ẩm, số người trong phòng, thời gian đã đóng cửa.
#
# Nhiệm vụ suy nghĩ:
# 1. Xác định thời điểm và tần suất đọc cảm biến để tạo từng dòng dữ liệu.
# 2. Đề xuất cách gắn nhãn mà không dùng trực tiếp một feature làm đáp án.
# 3. Xử lý giá trị cảm biến bị thiếu, âm hoặc vượt phạm vi hợp lý.
# 4. Huấn luyện Perceptron và phân tích các mẫu nằm gần ranh giới quyết định.
# 5. Thiết kế ba ca kiểm thử: bình thường, sát ngưỡng và đầu vào không hợp lệ.
#
# Không có code mẫu hoặc lời giải trong file này.
# Sinh viên bắt đầu phần triển khai bên dưới.

import pandas as pd
import joblib
import os
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class VentilationSystem:
    def __init__(self, model_path='ventilation_adaline.pkl'):
        self.model_path = model_path
        # Sử dụng Pipeline gộp chung chuẩn hóa (Scaler) và mô hình (Adaline)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('adaline', SGDClassifier(
                loss='squared_error', # Luật học Widrow-Hoff
                max_iter=2000, 
                tol=1e-3, 
                learning_rate='constant', 
                eta0=0.01, 
                random_state=42
            ))
        ])

    def clean_data(self, df):
        """Tiền xử lý và làm sạch dữ liệu"""
        df = df.copy()
        df['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
        df['CO2'] = df['CO2'].clip(lower=400, upper=5000)
        df['Temp'] = df['Temp'].clip(lower=15, upper=45)
        df['Humidity'] = df['Humidity'].clip(lower=0, upper=100)
        df['People'] = df['People'].clip(lower=0)
        df['Time_Closed'] = df['Time_Closed'].clip(lower=0)
        return df

    def train(self, data_path):
        """Đọc dữ liệu, huấn luyện và lưu mô hình"""
        print("1. Đang đọc và làm sạch dữ liệu...")
        df = pd.read_csv(data_path)
        df = self.clean_data(df)
        
        X = df[['CO2', 'Temp', 'Humidity', 'People', 'Time_Closed']].values
        y = df['Label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("2. Đang huấn luyện mô hình Adaline...")
        self.pipeline.fit(X_train, y_train)
        
        # Lưu mô hình xuống ổ cứng
        joblib.dump(self.pipeline, self.model_path)
        print(f"-> Đã lưu mô hình tại: {self.model_path}")
        
        print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
        y_pred = self.pipeline.predict(X_test)
        print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("\nBáo cáo chi tiết:")
        print(classification_report(y_test, y_pred, target_names=['Chưa cần (0)', 'Cần thông gió (1)']))

    def load_model(self):
        """Tải mô hình đã lưu từ ổ cứng lên"""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            return True
        return False

    def predict_status(self, features):
        """Dự đoán trạng thái từ dữ liệu cảm biến đầu vào"""
        # Pipeline tự động chuẩn hóa features trước khi đưa vào Adaline
        pred = self.pipeline.predict([features])[0]
        return "CẦN THÔNG GIÓ (1)" if pred == 1 else "CHƯA CẦN (0)"

# ==========================================
# KHỐI CHẠY CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    system = VentilationSystem()
    
    # 1. Huấn luyện mô hình (Chỉ cần chạy 1 lần khi có data mới)
    system.train('data/dataset.csv')
    
    # 2. Tải mô hình lên (Giả lập tình huống hệ thống khởi động lại)
    print("\n--- KHỞI ĐỘNG HỆ THỐNG CẢNH BÁO ---")
    if system.load_model():
        print("Tải mô hình thành công. Đang nhận dữ liệu cảm biến...")
        
        # 3. Chạy 3 ca kiểm thử
        print(f"[Cảm biến 1 - Bình thường]: {system.predict_status([500, 24.0, 50.0, 10, 15])}")
        print(f"[Cảm biến 2 - Sát ngưỡng]: {system.predict_status([1150, 27.0, 60.0, 25, 40])}")
        print(f"[Cảm biến 3 - Lỗi đầu vào]: {system.predict_status([400, 45.0, 100.0, 0, 0])}")
    else:
        print("Không tìm thấy file mô hình!")

if __name__ == "__main__":
    main()