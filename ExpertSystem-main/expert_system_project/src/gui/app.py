import streamlit as st
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from expert_system.expert_system import ExpertSystem
from logistic_regression.predictor import LogisticPredictor
from utils.hybrid import hybrid_predict

# Khởi tạo hệ chuyên gia và Logistic Regression
@st.cache_resource
def load_expert_system():
    return ExpertSystem(json_path="data/json/dataset_vi.json")

@st.cache_resource
def load_logistic_model():
    return LogisticPredictor(model_path="data/models/logistic_model.pkl")

expert = load_expert_system()
logistic = load_logistic_model()


# ==============================
# Sidebar
# ==============================
st.sidebar.title("🧑‍⚕️ Bác sĩ ảo")
st.sidebar.info("⚠️ Ứng dụng chỉ mang tính tham khảo, không thay thế bác sĩ.")

# Khởi tạo session_state cho lịch sử
if "history" not in st.session_state:
    st.session_state.history = []
if "hybrid_history" not in st.session_state:
    st.session_state.hybrid_history = []
if "compare_history" not in st.session_state:
    st.session_state.compare_history = []

# Hiển thị lịch sử trong sidebar
st.sidebar.markdown("### 💬 Lịch sử chẩn đoán (Expert System)")
for msg in st.session_state.history:
    st.sidebar.write(msg)
if st.sidebar.button("Xóa lịch sử Expert System"):
    st.session_state.history = []

st.sidebar.markdown("### 🕘 Lịch sử chẩn đoán (Hệ lai)")
for item in st.session_state.hybrid_history:
    st.sidebar.write(item)
if st.sidebar.button("Xóa lịch sử Hệ lai"):
    st.session_state.hybrid_history = []

st.sidebar.markdown("### 📊 Lịch sử so sánh mô hình")
for comp in st.session_state.compare_history:
    st.sidebar.write(f"{comp['time']} – Logistic: {comp['logistic']} | Expert: {comp['expert']}")
if st.sidebar.button("Xóa lịch sử So sánh"):
    st.session_state.compare_history = []

# Xuất toàn bộ lịch sử ra CSV
if st.sidebar.button("📤 Xuất lịch sử ra CSV"):
    data = []
    for msg in st.session_state.history:
        data.append({"Loại": "Expert System", "Kết quả": msg})
    for item in st.session_state.hybrid_history:
        data.append({"Loại": "Hệ lai", "Kết quả": item})
    for comp in st.session_state.compare_history:
        data.append({
            "Loại": "So sánh",
            "Thời gian": comp["time"],
            "Logistic": comp["logistic"],
            "Expert": comp["expert"]
        })
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Tải về CSV",
        data=csv,
        file_name="lich_su_chan_doan.csv",
        mime="text/csv"
    )

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3 = st.tabs([
    "🧠 Expert System", 
    "🧠 Hệ lai (Hybrid System)", 
    "📊 So sánh mô hình"
])

# ==============================
# Tab 1: Expert System – Nhập triệu chứng
# ==============================
with tab1:
    st.header("🧠 Hệ chuyên gia – Chẩn đoán theo triệu chứng")

    if "facts" not in st.session_state:
        st.session_state.facts = {}

    # Chọn triệu chứng từ danh sách
    available_symptoms = expert.list_all_symptoms()
    selected_symptoms = st.multiselect(
        "📝 Chọn triệu chứng bạn đang gặp phải:",
        options=available_symptoms,
        help="Bạn có thể chọn nhiều triệu chứng từ danh sách"
    )

    # Chọn ngưỡng độ tin cậy
    threshold = st.slider(
        "🔧 Chọn ngưỡng độ tin cậy tối thiểu (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Kết quả dưới ngưỡng này sẽ được cảnh báo là không đáng tin cậy"
    )

    if st.button("Chẩn đoán", key="expert_diag"):
        if not selected_symptoms:
            st.warning("Vui lòng chọn ít nhất một triệu chứng.")
            st.stop()

        # 🔥 RESET FACTS MỖI LẦN CHẨN ĐOÁN
        facts = {}

        symptom_text = ", ".join(selected_symptoms)

        # Trích xuất triệu chứng (KHÔNG dùng session_state)
        new_syms = expert.extract_symptoms(symptom_text, facts)

        # Nếu không trích được triệu chứng nào
        if not new_syms:
            st.warning("⚠️ Không nhận diện được triệu chứng hợp lệ.")
            st.stop()

        ranked = expert.forward_chaining(facts)

        if not ranked:
            st.error("❌ Không tìm thấy bệnh phù hợp với triệu chứng đã chọn.")
            st.stop()

        # 🔹 Lấy bệnh có độ phù hợp cao nhất
        top = ranked[0]

        st.subheader("📋 Kết luận sơ bộ")
        st.write(f"- Bệnh nghi ngờ: **{top['name']}** ({top['percent']:.1f}%)")

        # 🔥 Chỉ hiển thị triệu chứng THỰC SỰ khớp
        if top.get("symptoms"):
            st.write("Triệu chứng khớp:", ", ".join(top["symptoms"]))
        else:
            st.write("Triệu chứng khớp: Không xác định rõ")

        if top.get("advice"):
            st.info(top["advice"])

        # Cảnh báo nếu độ tin cậy thấp
        if top["percent"] < threshold:
            st.warning(
                "⚠️ Kết quả có độ tin cậy thấp do thiếu triệu chứng. "
                "Vui lòng bổ sung thêm."
            )

        # Lưu lịch sử
        st.session_state.history.append(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} – "
            f"{top['name']} ({top['percent']:.1f}%)"
        )


# ==============================
# Tab 2: Hệ lai (Hybrid System – tuần tự)
# ==============================
with tab2:
    st.header("🧠 Hệ lai – Chẩn đoán tuần tự")

    symptom_text = st.text_area("Nhập triệu chứng (ví dụ: sốt, ho, đau đầu...)", key="hybrid_text")

    if st.button("Chẩn đoán", key="form_diag"):
        result = hybrid_predict(symptom_text, expert, logistic)

        # Expert System
        top_expert = result.get("expert")
        if top_expert:
            st.subheader("📋 Expert System")
            st.write(f"- Dự đoán: **{top_expert['name']}**")
            st.progress(int(top_expert['percent']))
            if top_expert.get('symptoms'):
                st.write("Triệu chứng khớp:", ", ".join(top_expert['symptoms']))
            if top_expert.get('advice'):
                st.info(top_expert['advice'])

        # Logistic Regression
        logistic_pred = result.get("logistic")
        if logistic_pred:
            st.subheader("📋 Logistic Regression")
            st.write(f"- Dự đoán: **{logistic_pred['name']}**")
            st.progress(int(logistic_pred['percent']))

        # Kết luận hệ lai
        final = result.get("final")
        if final:
            st.subheader("✅ Kết luận hệ lai")
            st.success(f"Hệ lai dự đoán: **{final['name']}** (độ tin cậy trung bình {final['score']:.1f}%)")

            st.session_state.hybrid_history.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} – {final['name']} ({final['score']:.1f}%)"
            )

# ==============================
# Tab 3: So sánh mô hình
# ==============================
with tab3:
    st.header("📊 So sánh Expert System và Logistic Regression")

    symptom_text = st.text_area("Nhập triệu chứng để so sánh mô hình:", key="compare_text")

    if st.button("So sánh", key="compare_diag"):
        logistic_pred = logistic.predict(symptom_text)

        facts = {}
        new_syms = expert.extract_symptoms(symptom_text, facts)
        if isinstance(new_syms, set):
            for sym, val in new_syms:
                facts[sym] = val
        ranked = expert.forward_chaining(facts)

        if logistic_pred and ranked:
            top_expert = ranked[0] if ranked else None
            if top_expert:
                st.subheader("📈 Kết quả so sánh")
                st.table({
                    "Mô hình": ["Logistic Regression", "Expert System"],
                    "Bệnh dự đoán": [logistic_pred['name'], top_expert['name']],
                    "Xác suất (%)": [f"{logistic_pred['percent']:.1f}", f"{top_expert['percent']:.1f}"]
                })

                fig, ax = plt.subplots()
                models = ["Logistic Regression", "Expert System"]
                percents = [logistic_pred['percent'], top_expert['percent']]
                ax.pie(
                    percents,
                    labels=models,
                    autopct="%1.1f%%",
                    colors=["orange", "blue"],
                    startangle=90
                )
                ax.set_title("So sánh độ tin cậy giữa hai mô hình")
                st.pyplot(fig)

                st.session_state.compare_history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "logistic": f"{logistic_pred['name']} ({logistic_pred['percent']:.1f}%)",
                    "expert": f"{top_expert['name']} ({top_expert['percent']:.1f}%)"
                })

                st.info("📌 Nhận xét: Tab này giúp đánh giá sự khác biệt giữa hai mô hình. "
                        "Logistic Regression mạnh với dữ liệu lớn, trong khi Expert System "
                        "dựa trên luật và phù hợp với tri thức chuyên gia.")
