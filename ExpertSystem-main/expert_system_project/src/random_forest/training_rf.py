import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Tìm đường dẫn tuyệt đối đến file CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "..", "data", "raw", "Disease_symptom_and_patient_profile_dataset.csv")

# Kiểm tra file tồn tại
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file CSV tại: {csv_path}")

# Đọc dữ liệu
df = pd.read_csv(csv_path)

# Tiền xử lý
X = pd.get_dummies(df.drop("Outcome Variable", axis=1))
y = (df["Outcome Variable"] == "Positive").astype(int)

# Chia dữ liệu
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(
    n_estimators=100,       # số lượng cây
    max_depth=None,         # độ sâu tối đa (None = cho đến khi lá thuần nhất)
    random_state=42
)
model.fit(X_train, y_train)

# Tạo thư mục lưu mô hình nếu chưa có
model_dir = os.path.join(base_dir, "..", "..", "data", "models")
os.makedirs(model_dir, exist_ok=True)

# Lưu mô hình
model_path = os.path.join(model_dir, "random_forest_model.pkl")
joblib.dump(model, model_path)

print(f"✅ Random Forest model đã được lưu tại: {model_path}")
