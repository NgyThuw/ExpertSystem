import os
import joblib
import pandas as pd

class LogisticPredictor:
    def __init__(self, model_path="data/models/logistic_model.pkl"):
        self.model = None
        self.feature_names = None

        if os.path.exists(model_path):
            saved = joblib.load(model_path)
            self.model = saved.get("model")
            self.feature_names = saved.get("feature_names")
        else:
            raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại: {model_path}")

    def preprocess_input(self, input_text):
        if not self.feature_names:
            return None

        # Tạo DataFrame với tất cả đặc trưng = 0
        input_df = pd.DataFrame([[0] * len(self.feature_names)], columns=self.feature_names)

        # Đánh dấu các đặc trưng có mặt trong văn bản
        text = input_text.lower()
        for feature in self.feature_names:
            if feature.lower() in text:
                input_df.at[0, feature] = 1

        return input_df

    def predict(self, input_text):
        if not self.model:
            return None

        X_input = self.preprocess_input(input_text)
        if X_input is None:
            return None

        pred = self.model.predict(X_input)[0]
        prob = self.model.predict_proba(X_input).max() * 100
        return {"name": "Positive" if pred == 1 else "Negative", "percent": prob}
