import pandas as pd
from datetime import datetime

def export_history_to_csv(chatbot_history, hybrid_history, compare_history, filename="lich_su_chan_doan.csv"):
    """
    Xuất lịch sử chẩn đoán ra file CSV.
    
    Parameters:
        chatbot_history (list): danh sách lịch sử hội thoại Chatbot
        hybrid_history (list): danh sách lịch sử chẩn đoán Hệ lai
        compare_history (list of dict): danh sách lịch sử so sánh mô hình
        filename (str): tên file CSV xuất ra
    
    Returns:
        csv (bytes): nội dung CSV dạng bytes để dùng với Streamlit download_button
    """
    data = []

    # Chatbot
    for msg in chatbot_history:
        data.append({
            "Loại": "Chatbot",
            "Thời gian": msg.split(" – ")[0] if "–" in msg else "",
            "Kết quả": msg
        })

    # Hệ lai
    for item in hybrid_history:
        data.append({
            "Loại": "Hệ lai",
            "Thời gian": item.split(" – ")[0] if "–" in item else "",
            "Kết quả": item
        })

    # So sánh mô hình
    for comp in compare_history:
        data.append({
            "Loại": "So sánh",
            "Thời gian": comp["time"],
            "Logistic": comp["logistic"],
            "Expert": comp["expert"]
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")
