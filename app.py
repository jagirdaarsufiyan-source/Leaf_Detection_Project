import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI SmartLeaf Detection",
    page_icon="🌿",
    layout="centered"
)

# ---------- BACKGROUND IMAGE ----------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("image3.avif")

# ---------- MODERN UI ----------
st.markdown(f"""
<style>

.stApp {{
    background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
}}

.main-card {{
    background: rgba(255, 255, 255, 0.85);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}}

.title {{
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: black;
}}

.subtitle {{
    text-align: center;
    color: #333;
    margin-bottom: 20px;
}}

.result-card {{
    background: #ffffff;
    padding: 15px;
    border-radius: 12px;
    margin-top: 12px;
    color: black;
    border-left: 6px solid #00c853;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}}

.stButton>button {{
    background: linear-gradient(45deg, #00c853, #64dd17);
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
}}

</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
model = YOLO("best.pt")

# ---------- HEADER ----------
st.markdown('<div class="title">🌿 AI SmartLeaf Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Leaf Analysis with AI + Solutions</div>', unsafe_allow_html=True)

# ---------- LANGUAGE ----------
language = st.selectbox("🌐 Language", ["English", "Telugu", "Kannada"])

# ---------- SOLUTIONS ----------
solutions = {
    "Boron": {
        "English": "Boron deficiency causes weak leaf growth. Use boron fertilizers like borax in small amounts.",
        "Telugu": "బోరాన్ లోపం వల్ల ఆకులు బలహీనంగా పెరుగుతాయి. బోరాక్స్ ఎరువులు కొద్దిగా వాడండి.",
        "Kannada": "ಬೋರಾನ್ ಕೊರತೆಯಿಂದ ಎಲೆ ದುರ್ಬಲವಾಗುತ್ತದೆ. ಬೋರಾಕ್ಸ್ ಗೊಬ್ಬರವನ್ನು ಸ್ವಲ್ಪ ಪ್ರಮಾಣದಲ್ಲಿ ಬಳಸಿ."
    },
    "Nitrogen": {
        "English": "Nitrogen deficiency causes yellow leaves. Use urea or compost.",
        "Telugu": "నత్రజని లోపం వల్ల ఆకులు పసుపు రంగులోకి మారతాయి. యూరియా వాడండి.",
        "Kannada": "ನೈಟ್ರೋಜನ್ ಕೊರತೆಯಿಂದ ಎಲೆ ಹಳದಿ ಆಗುತ್ತದೆ. ಯೂರಿಯಾ ಬಳಸಿ."
    },
    "Potassium": {
        "English": "Potassium deficiency affects leaf edges. Use potash fertilizers.",
        "Telugu": "పొటాషియం లోపం వల్ల ఆకుల అంచులు దెబ్బతింటాయి.",
        "Kannada": "ಪೊಟ್ಯಾಸಿಯಂ ಕೊರತೆಯಿಂದ ಎಲೆ ಅಂಚು ಹಾನಿಯಾಗುತ್ತದೆ."
    }
}

# ---------- MAIN CARD ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)

    # Convert image
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Prediction
    results = model.predict(img, conf=0.25)

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(annotated, caption="Detected", use_column_width=True)

    st.markdown("### 📊 Results & Suggestions")

    names = model.names

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = names[cls_id]

            solution_text = solutions.get(label, {}).get(
                language,
                "Use balanced fertilizers and proper irrigation."
            )

            st.markdown(f"""
            <div class="result-card">
            ✅ <b>{label}</b><br>
            Confidence: {conf:.2f}<br><br>
            🌱 {solution_text}
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Upload an image to start")

st.markdown('</div>', unsafe_allow_html=True)