import json
import os
import unicodedata
from pathlib import Path

class ExpertSystem:
    def __init__(self, json_path):
        self.rules = []
        self.symptom_keywords = {}

        BASE_DIR = Path(__file__).resolve().parents[2]
        json_path = BASE_DIR / json_path

        if not json_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file dataset: {json_path}")
        
        self.load_knowledge_base(json_path)

    def load_knowledge_base(self, json_path):
        if not os.path.exists(json_path):
            print(f"❌ Không tìm thấy file: {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.rules = data.get("rules", [])
            self.symptom_keywords = data.get("symptom_keywords", {})

    @staticmethod
    def normalize_text(text):
        """Chuẩn hóa tiếng Việt: bỏ dấu, lowercase."""
        text = text.lower()
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        return text

    def extract_symptoms(self, symptom_text, facts):
        """
        Trích xuất triệu chứng từ input người dùng.
        - Nếu có symptom_keywords thì dùng để mapping nhiều cách diễn đạt.
        - Nếu không có thì fallback về conditions trong rules.
        """
        text_norm = self.normalize_text(symptom_text)
        detected = set()

        if self.symptom_keywords:
            for symptom, keywords in self.symptom_keywords.items():
                for kw in keywords:
                    kw_norm = self.normalize_text(kw)
                    if kw_norm in text_norm:
                        if "khong" in text_norm or "không" in symptom_text.lower():
                            facts[symptom] = "Không"
                        else:
                            facts[symptom] = "Có"
                        detected.add((symptom, facts[symptom]))
        else:
            for rule in self.rules:
                for cond in rule["conditions"].keys():
                    cond_norm = self.normalize_text(cond)
                    if cond_norm in text_norm:
                        if "khong" in text_norm or "không" in symptom_text.lower():
                            facts[cond] = "Không"
                        else:
                            facts[cond] = "Có"
                        detected.add((cond, facts[cond]))

        return detected

    def forward_chaining(self, facts):
        results = []
        for rule in self.rules:
            conditions = rule["conditions"]
            matched = 0
            total = len(conditions)

            for cond, expected in conditions.items():
                if cond in facts and str(facts[cond]).lower() == str(expected).lower():
                    matched += 1

            percent = (matched / total) * 100 if total > 0 else 0

            if percent > 0:
                results.append({
                    "name": rule["conclusion"],
                    "percent": percent,
                    "symptoms": [cond for cond in conditions if cond in facts],
                    "advice": rule.get("advice", "")
                })

        results = sorted(results, key=lambda x: x["percent"], reverse=True)
        return results

    def explain_disease(self, disease_name):
        for rule in self.rules:
            if rule["conclusion"].lower() == disease_name.lower():
                return {
                    "name": rule["conclusion"],
                    "symptoms": list(rule["conditions"].keys()),
                    "advice": rule.get("advice", "Không có lời khuyên cụ thể.")
                }
        return None

    def list_all_symptoms(self):
        if self.symptom_keywords:
            return list(self.symptom_keywords.keys())

        # Fallback: lấy từ conditions trong rules
        symptoms = set()
        for rule in self.rules:
            symptoms.update(rule.get("conditions", {}).keys())

        return sorted(symptoms)

    def list_all_diseases(self):
        return list(set(rule["conclusion"] for rule in self.rules))
