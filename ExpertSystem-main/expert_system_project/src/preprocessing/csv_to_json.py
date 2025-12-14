import pandas as pd
import os
import json

# Đường dẫn
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "..", "data", "raw", "Disease_symptom_and_patient_profile_dataset.csv")
json_path_vi = os.path.join(base_dir, "..", "..", "data", "json", "dataset_vi.json")

# Bảng dịch cột sang tiếng Việt
column_map = {
    "Fever": "Sốt",
    "Cough": "Ho",
    "Fatigue": "Mệt mỏi",
    "Difficulty Breathing": "Khó thở",
    "Age": "Tuổi",
    "Gender": "Giới tính",
    "Blood Pressure": "Huyết áp",
    "Cholesterol Level": "Mỡ máu",
    "Disease": "Bệnh"
}

# Bảng dịch giá trị Yes/No
value_map = {
    "Yes": "Có",
    "No": "Không",
    "Male": "Nam",
    "Female": "Nữ",
    "Low": "Thấp",
    "Normal": "Bình thường",
    "High": "Cao"
}

# Đọc CSV
df = pd.read_csv(csv_path)

# Đổi tên cột sang tiếng Việt
df.rename(columns=column_map, inplace=True)

# Tạo rules
rules = []
for idx, row in df.iterrows():
    conditions = {}
    for col in df.columns:
        if col != "Bệnh":
            val = row[col]
            # dịch giá trị nếu có trong value_map
            if str(val) in value_map:
                val = value_map[str(val)]
            conditions[col] = val
    rules.append({
        "code": f"R{idx+1:04d}",
        "conclusion": row["Bệnh"],
        "conditions": conditions,
        "advice": "Tham khảo ý kiến bác sĩ nếu triệu chứng kéo dài."
    })

# Tạo symptom_keywords thuần Việt
symptom_keywords = {}
for col in df.columns:
    if col != "Bệnh":
        symptom_keywords[col] = [col.lower()]

# Gộp thành JSON
data = {
    "rules": rules,
    "symptom_keywords": symptom_keywords
}

# Ghi ra file JSON thuần Việt
os.makedirs(os.path.dirname(json_path_vi), exist_ok=True)
with open(json_path_vi, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ Đã tạo file JSON thuần Việt tại: {json_path_vi}")
