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

st.write("Files:", os.listdir())
st.write("Model size:", os.path.getsize("model.tflite"))
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AgroVision AI", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#2E8B57;'>
🌱 AgroVision AI — Smart Agriculture System
</h1>
""", unsafe_allow_html=True)

# =========================
# UI STYLE (UPGRADED)
# =========================
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg,#f0fff4,#ccf2e0);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.15);
    text-align: center;
}
.high {background-color:#ffdddd;}
.medium {background-color:#fff4cc;}
.low {background-color:#ddffdd;}
</style>
""", unsafe_allow_html=True)

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
# LOAD KNOWLEDGE BASE (JSON)
# =========================
@st.cache_data
def load_kb():
    with open("knowledge_base.json") as f:
        return json.load(f)

knowledge_base = load_kb()

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
# WEATHER UI (UPGRADED)
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
        color = "high"
        message = "Immediate action required!"
    elif conf >= 0.4 or humidity > 65:
        level = "MEDIUM"
        color = "medium"
        message = "Monitor closely"
    else:
        level = "LOW"
        color = "low"
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
# SMART AI ADVICE ENGINE (UPGRADED)
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

    # 🌦 Weather intelligence
    weather_note = ""
    if humidity > 80:
        weather_note += "\n⚠ High humidity → fungal risk"
    if temp > 35:
        weather_note += "\n🔥 Heat stress risk"

    # 🎯 Confidence logic
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
# SESSION TRACKING (HISTORY)
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
# PLOTLY TREND GRAPH
# =========================
def plot_trend():

    if len(st.session_state.session_conf) == 0:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=st.session_state.session_conf,
        mode='lines+markers',
        name='Confidence'
    ))

    fig.update_layout(
        title="📈 Prediction Trend",
        xaxis_title="Time",
        yaxis_title="Confidence",
        template="plotly_white"
    )

    return fig


# =========================
# TOP PREDICTIONS BAR CHART
# =========================
def plot_top_predictions(output, labels):

    top_idx = np.argsort(output)[-5:][::-1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[output[i] for i in top_idx],
        y=[labels[i] for i in top_idx],
        orientation='h'
    ))

    fig.update_layout(
        title="📊 Top Predictions",
        template="plotly_white"
    )

    return fig, top_idx


# =========================
# SMART FARM CALCULATOR
# =========================
def farm_calculator(area, humidity, temp, soil_moisture=25):

    try:
        area = float(area)
    except:
        area = 1.0

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

    total_N = N * area
    total_P = P * area
    total_K = K * area

    return f"""
🌾 Smart Farm Plan

Area: {area} acres  

💧 Irrigation: {int(irrigation * area)} L  

📆 Spray Interval: {spray} days  

🌱 Fertilizer:
N: {int(total_N)} kg  
P: {int(total_P)} kg  
K: {int(total_K)} kg  
"""


# =========================
# GAUGE CHART (CONFIDENCE)
# =========================
def show_gauge(confidence):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"}
        }
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
# IMAGE INPUT (UPLOAD + CAMERA)
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

    if interpreter is None:
        st.error("Model not loaded")
        return None

    try:
        if image is None:
            return None

        image.seek(0)
        img = preprocess(image)

        # MODEL INFERENCE
        output = model.predict(img)[0]

        if len(output.shape) == 2:
            output = output[0]

        safe_classes = class_names[:len(output)]

        # Prediction
        idx = int(np.argmax(output))
        confidence = float(output[idx])
        label = safe_classes[idx]

        # Weather
        humidity, temp = get_weather(city)

        # Severity
        level, color, message, notes = get_severity(confidence, humidity, temp)

        # Spray
        spray_days = spray_schedule(humidity, level)

        # Advice
        advice = get_advice(label, language, humidity, temp, confidence)

        # Farm plan
        farm_info = farm_calculator(area, humidity, temp)

        # Charts
        fig_bar, _ = plot_top_predictions(output, safe_classes)
        fig_trend = plot_trend()

        # Update session
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

    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)

    if st.button("🚀 Analyze Crop"):

        with st.spinner("Analyzing crop..."):

            result = predict(uploaded_file, city, area, language)

        if result:

            # =========================
            # DASHBOARD CARDS
            # =========================
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"<div class='card'><h3>{result['label']}</h3>Disease</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"<div class='card'><h3>{result['confidence']:.2f}</h3>Confidence</div>", unsafe_allow_html=True)

            with col3:
                st.markdown(f"<div class='card {result['color']}'><h3>{result['level']}</h3>Severity</div>", unsafe_allow_html=True)

            # =========================
            # CONFIDENCE PROGRESS BAR
            # =========================
            st.subheader("🔋 Confidence Level")
            st.progress(int(result["confidence"] * 100))

            # =========================
            # GAUGE CHART
            # =========================
            show_gauge(result["confidence"])

            # =========================
            # WEATHER + ALERT
            # =========================
            show_weather_ui(result["temp"], result["humidity"])
            show_risk_alert(result["level"], result["confidence"])

            # =========================
            # TABS
            # =========================
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔬 Prediction",
                "👨‍🌾 Advice",
                "📊 Analytics",
                "🧮 Farm Tools"
            ])

            # TAB 1 — Prediction
            with tab1:
                show_severity_card(result["level"], result["color"], result["message"])

                for note in result["notes"]:
                    st.info(note)

                st.write(f"💊 Spray Interval: {result['spray']} days")

            # TAB 2 — Expandable Advice
            with tab2:
                with st.expander("📖 Detailed Farmer Advice", expanded=True):
                    st.markdown(result["advice"])

            # TAB 3 — Analytics
            with tab3:
                if result["fig_bar"]:
                    st.plotly_chart(result["fig_bar"], use_container_width=True)

                if result["fig_trend"]:
                    st.plotly_chart(result["fig_trend"], use_container_width=True)

                st.subheader("📈 Confidence History")
                st.line_chart(st.session_state.session_conf)

            # TAB 4 — Farm Tools
            with tab4:
                st.markdown(result["farm"])

            # =========================
            # DOWNLOAD REPORT
            # =========================
            report = f"""
AgroVision Report

Disease: {result['label']}
Confidence: {result['confidence']:.2f}
Severity: {result['level']}

Advice:
{result['advice']}
"""
            st.download_button("📄 Download Report", report, file_name="agrovision_report.txt")

            # =========================
            # FARMER MODE
            # =========================
            if farmer_mode:
                st.success("👨‍🌾 Farmer Mode Enabled")
                st.info("📌 Tip: Spray during early morning or evening")

else:
    st.warning("Please upload or capture an image")
