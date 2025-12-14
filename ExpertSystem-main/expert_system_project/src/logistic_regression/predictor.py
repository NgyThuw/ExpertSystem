import joblib
import pandas as pd
from pathlib import Path

class LogisticPredictor:
    def __init__(self, model_path="data/models/logistic_model.pkl"):
        self.model = None
        self.feature_names = None

        # 🔑 ROOT PROJECT (expert_system_project)
        BASE_DIR = Path(__file__).resolve().parents[2]
        model_path = BASE_DIR / model_path

        if model_path.exists():
            saved = joblib.load(model_path)
            self.model = saved.get("model")
            self.feature_names = saved.get("feature_names")
        else:
            raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại: {model_path}")

    def preprocess_input(self, input_text):
        if not self.feature_names:
            return None

        df = pd.DataFrame([[0]*len(self.feature_names)], columns=self.feature_names)
        text = input_text.lower()

        for f in self.feature_names:
            if f.lower() in text:
                df.at[0, f] = 1
        return df

    def predict(self, input_text):
        if not self.model:
            return None

        X = self.preprocess_input(input_text)
        if X is None:
            return None

        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X).max() * 100

        return {
            "name": "Positive" if pred == 1 else "Negative",
            "percent": prob
        }
