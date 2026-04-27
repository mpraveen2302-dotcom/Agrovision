# =========================
# IMPORTS
# =========================
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
import json
import requests
import plotly.graph_objects as go
import tensorflow as tf
import time
from deep_translator import GoogleTranslator

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AgroVision AI PRO", layout="wide")

# =========================
# 🌙 DARK MODE TOGGLE
# =========================
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# =========================
# 🎨 PRO CSS
# =========================
st.markdown(f"""
<style>
.stApp {{
    {"background: linear-gradient(-45deg,#0f2027,#203a43,#000000);" if dark_mode else "background: linear-gradient(-45deg,#ecfdf5,#d1fae5,#bbf7d0);"}
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}}

@keyframes gradientBG {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#2e7d32,#1b5e20);
}}
section[data-testid="stSidebar"] * {{
    color: white !important;
}}

.glass {{
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}}

.card {{
    padding:22px;
    border-radius:18px;
    text-align:center;
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.2);
}}

.card1 {{ border-left:6px solid #22c55e; }}
.card2 {{ border-left:6px solid #3b82f6; }}
.card3 {{ border-left:6px solid #f97316; }}

.stTextInput input,
.stNumberInput input {{
    background-color: {"#222" if dark_mode else "#ffffff"} !important;
    color: {"white" if dark_mode else "black"} !important;
}}

h1,h2,h3 {{
    color: {"#ffffff" if dark_mode else "#1b5e20"};
}}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🌱 AgroVision AI — Smart Agriculture System")

# =========================
# 🌾 DASHBOARD
# =========================
st.markdown('<div class="glass"><h2>🌾 Smart Farm Dashboard</h2></div>', unsafe_allow_html=True)

d = st.session_state.get("last_result", {})

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="card card1"><h3>{d.get("level","--")}</h3><p>Status</p></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card card2"><h3>{d.get("confidence",0):.2f}</h3><p>Confidence</p></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card card3"><h3>{d.get("temp","--")}°C</h3><p>Temperature</p></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="card card1"><h3>{d.get("humidity","--")}%</h3><p>Humidity</p></div>', unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# =========================
# LOAD CLASSES
# =========================
@st.cache_data
def load_classes():
    with open("class_names.json") as f:
        return json.load(f)

class_names = load_classes()

# =========================
# LOAD KNOWLEDGE BASE (FINAL FIX)
# =========================
@st.cache_data
def load_kb_safe():
    try:
        with open("knowledge_base.json") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Knowledge base not loaded: {e}")
        return {}

knowledge_base = load_kb_safe()


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess(image):
    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img


# =========================
# TRANSLATION SYSTEM
# =========================
def translate_text(text, lang):
    if lang == "English":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang.lower()).translate(text)
    except:
        return text


# =========================
# WEATHER API
# =========================
def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        data = requests.get(url, timeout=5).json()
        humidity = float(data["current_condition"][0]["humidity"])
        temp = float(data["current_condition"][0]["temp_C"])
    except:
        humidity, temp = 60.0, 25.0
    return humidity, temp


# =========================
# WEATHER UI
# =========================
def show_weather_ui(temp, humidity):

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🌡 Temperature", f"{temp}°C")

    with col2:
        st.metric("💧 Humidity", f"{humidity}%")

    if humidity > 80:
        st.warning("⚠ High humidity → Disease risk")

    if temp > 35:
        st.warning("🔥 High temperature → Stress risk")


# =========================
# SCIENTIFIC RULES
# =========================
def scientific_rules(humidity, temp):
    notes = []

    if humidity > 80:
        notes.append("High humidity → fungal disease risk")

    if temp > 35:
        notes.append("High temperature → plant stress")

    if temp < 15:
        notes.append("Low temperature → slow recovery")

    return notes


# =========================
# SEVERITY SYSTEM
# =========================
def get_severity(conf, humidity, temp):

    if conf >= 0.7 or humidity > 80:
        level = "HIGH"
        color = "card3"
        message = "Immediate action required!"
    elif conf >= 0.4 or humidity > 65:
        level = "MEDIUM"
        color = "card2"
        message = "Monitor closely"
    else:
        level = "LOW"
        color = "card1"
        message = "Safe condition"

    notes = scientific_rules(humidity, temp)

    return level, color, message, notes


# =========================
# SEVERITY CARD
# =========================
def show_severity_card(level, color, message):

    st.markdown(f"""
    <div class="card {color}">
    <h2>⚠️ {level}</h2>
    <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# RISK ALERT
# =========================
def show_risk_alert(level, confidence):

    if level == "HIGH":
        st.error(f"🚨 High Risk! Confidence: {confidence:.2f}")

    elif level == "MEDIUM":
        st.warning(f"⚠️ Moderate Risk! Confidence: {confidence:.2f}")

    else:
        st.success(f"✅ Low Risk. Confidence: {confidence:.2f}")


# =========================
# SPRAY SCHEDULING
# =========================
def spray_schedule(humidity, level):

    if level == "HIGH":
        return 5
    elif humidity > 80:
        return 7
    elif humidity > 60:
        return 10
    else:
        return 14


# =========================
# SMART AI ADVICE ENGINE
# =========================
def get_advice(label, language, humidity, temp, confidence):

    info = knowledge_base.get(label)

    if not info:
        base = f"""
General Advice:
- Monitor crop regularly
- Maintain irrigation
- Remove infected leaves
- Use proper pesticide

Detected: {label}
"""
    else:
        base = f"""
Symptoms: {info['Symptoms']}
Causes: {info['Causes']}
Prevention: {info['Prevention']}
Cure: {info['Cure']}
Impact: {info['Impact']}
Best Practices: {info['Best Practices']}
"""

    weather_note = ""
    if humidity > 80:
        weather_note += "\n⚠ High humidity → fungal risk"
    if temp > 35:
        weather_note += "\n🔥 Heat stress risk"

    confidence_note = ""
    if confidence > 0.8:
        confidence_note = "\n🚨 High confidence → Act immediately"
    elif confidence > 0.5:
        confidence_note = "\n⚠ Moderate confidence → Monitor closely"
    else:
        confidence_note = "\n✅ Low risk"

    final_advice = base + weather_note + confidence_note

    return translate_text(final_advice, language)


# =========================
# SESSION TRACKING
# =========================
if "session_conf" not in st.session_state:
    st.session_state.session_conf = []
    st.session_state.session_time = []

def update_session(conf):
    if len(st.session_state.session_conf) >= 20:
        st.session_state.session_conf.pop(0)
        st.session_state.session_time.pop(0)

    st.session_state.session_conf.append(conf)
    st.session_state.session_time.append(time.strftime("%H:%M:%S"))


# =========================
# TREND GRAPH
# =========================
def plot_trend():

    if len(st.session_state.session_conf) == 0:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=st.session_state.session_conf,
        mode='lines+markers'
    ))

    fig.update_layout(
        title="📈 Prediction Trend",
        template="plotly_white"
    )

    return fig


# =========================
# TOP PREDICTIONS
# =========================
def plot_top_predictions(output, labels):

    top_idx = np.argsort(output)[-5:][::-1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[output[i] for i in top_idx],
        y=[labels[i] for i in top_idx],
        orientation='h'
    ))

    fig.update_layout(title="📊 Top Predictions")

    return fig, top_idx


# =========================
# FARM CALCULATOR
# =========================
def farm_calculator(area, humidity, temp, soil_moisture=25):

    area = float(area)

    irrigation = 3000

    if temp > 32:
        irrigation += 800
    if humidity < 40:
        irrigation += 500
    if soil_moisture > 35:
        irrigation -= 700

    irrigation = max(1500, irrigation)

    spray = 7 if humidity > 80 else 10 if humidity > 60 else 14

    N, P, K = 60, 40, 40

    return f"""
🌾 Smart Farm Plan

Area: {area} acres  

💧 Irrigation: {int(irrigation * area)} L  

📆 Spray Interval: {spray} days  

🌱 Fertilizer:
N: {int(N * area)} kg  
P: {int(P * area)} kg  
K: {int(K * area)} kg  
"""


# =========================
# GAUGE CHART
# =========================
def show_gauge(confidence):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Level"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Controls")

language = st.sidebar.selectbox("🌐 Language", ["English", "Tamil", "Hindi"])
city = st.sidebar.text_input("📍 City", "Chennai")
area = st.sidebar.number_input("🌾 Farm Area (acres)", value=1.0)
farmer_mode = st.sidebar.checkbox("👨‍🌾 Farmer Mode", True)

# =========================
# IMAGE INPUT
# =========================
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("📸 Or Take Photo")

if camera_image is not None:
    uploaded_file = camera_image


# =========================
# PREDICTION FUNCTION
# =========================
def predict(image, city, area, language):

    if model is None:
        st.error("Model not loaded")
        return None

    try:
        if image is None:
            return None

        image.seek(0)
        img = preprocess(image)

        output = model.predict(img)[0]

        if len(output.shape) == 2:
            output = output[0]

        safe_classes = class_names[:len(output)]

        idx = int(np.argmax(output))
        confidence = float(output[idx])
        label = safe_classes[idx]

        humidity, temp = get_weather(city)

        level, color, message, notes = get_severity(confidence, humidity, temp)
        spray_days = spray_schedule(humidity, level)
        advice = get_advice(label, language, humidity, temp, confidence)
        farm_info = farm_calculator(area, humidity, temp)

        fig_bar, _ = plot_top_predictions(output, safe_classes)
        fig_trend = plot_trend()

        update_session(confidence)

        return {
            "label": label,
            "confidence": confidence,
            "level": level,
            "color": color,
            "message": message,
            "notes": notes,
            "spray": spray_days,
            "humidity": humidity,
            "temp": temp,
            "advice": advice,
            "farm": farm_info,
            "fig_bar": fig_bar,
            "fig_trend": fig_trend
        }

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# =========================
# MAIN EXECUTION
# =========================
if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Leaf", use_container_width=True)

    if st.button("🚀 Analyze Crop"):

        with st.spinner("Analyzing crop..."):
            result = predict(uploaded_file, city, area, language)

        if result:

            # 🔥 STORE FOR DASHBOARD
            st.session_state.last_result = result

            # SAFE EXTRACTION
            prediction_label = result.get("label", "Unknown")
            confidence = float(result.get("confidence", 0))
            severity = result.get("level", "Unknown")

            st.markdown("<br>", unsafe_allow_html=True)

            # =========================
            # 🎨 TOP RESULT CARDS
            # =========================
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="card card1">
                    <h2>{prediction_label}</h2>
                    <p>Disease</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card card2">
                    <h2>{confidence:.2f}</h2>
                    <p>Confidence</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="card card3">
                    <h2>{severity}</h2>
                    <p>Severity</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # =========================
            # 📊 CONFIDENCE + GAUGE
            # =========================
            st.subheader("📊 Confidence Level")
            st.progress(min(max(confidence, 0.0), 1.0))

            st.subheader("🔋 Confidence Meter")
            show_gauge(confidence)

            # =========================
            # 🌦 WEATHER + ALERT
            # =========================
            show_weather_ui(result["temp"], result["humidity"])
            show_risk_alert(result["level"], confidence)

            # =========================
            # 🧠 TABS (GLASS STYLE)
            # =========================
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔬 Prediction",
                "👨‍🌾 Advice",
                "📊 Analytics",
                "🧮 Farm Tools"
            ])

            # =========================
            # TAB 1 — Prediction
            # =========================
            with tab1:
                st.markdown('<div class="glass">', unsafe_allow_html=True)

                show_severity_card(result["level"], result["color"], result["message"])

                for note in result["notes"]:
                    st.info(note)

                st.write(f"💊 Spray Interval: {result['spray']} days")

                st.markdown('</div>', unsafe_allow_html=True)

            # =========================
            # TAB 2 — Advice
            # =========================
            with tab2:
                st.markdown('<div class="glass">', unsafe_allow_html=True)

                with st.expander("📖 Detailed Farmer Advice", expanded=True):
                    st.markdown(result["advice"])

                st.markdown('</div>', unsafe_allow_html=True)

            # =========================
            # TAB 3 — Analytics
            # =========================
            with tab3:
                st.markdown('<div class="glass">', unsafe_allow_html=True)

                if result["fig_bar"]:
                    st.plotly_chart(result["fig_bar"], use_container_width=True)

                if result["fig_trend"]:
                    st.plotly_chart(result["fig_trend"], use_container_width=True)

                st.subheader("📈 Confidence History")
                st.line_chart(st.session_state.session_conf)

                st.markdown('</div>', unsafe_allow_html=True)

            # =========================
            # TAB 4 — Farm Tools
            # =========================
            with tab4:
                st.markdown('<div class="glass">', unsafe_allow_html=True)

                st.markdown(result["farm"])

                st.markdown('</div>', unsafe_allow_html=True)

            # =========================
            # 📄 DOWNLOAD REPORT
            # =========================
            report = f"""
AgroVision Report

Disease: {result['label']}
Confidence: {confidence:.2f}
Severity: {result['level']}

Advice:
{result['advice']}
"""
            st.download_button(
                "📄 Download Report",
                report,
                file_name="agrovision_report.txt"
            )

            # =========================
            # 👨‍🌾 FARMER MODE
            # =========================
            if farmer_mode:
                st.success("👨‍🌾 Farmer Mode Enabled")
                st.info("📌 Tip: Spray during early morning or evening")

else:
    st.warning("Please upload or capture an image")

knowledge_base = load_kb()
