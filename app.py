
import io
import streamlit as st
import numpy as np
from PIL import Image
import json
import requests
import plotly.graph_objects as go
import tflite_runtime.interpreter as tflite
import time



# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="AgroVision AI", layout="wide")

st.title("🌱 AgroVision AI — Smart Agriculture System")

# -----------------------
# CSS (CARDS)
# -----------------------
st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
.high {background-color:#ffdddd;}
.medium {background-color:#fff4cc;}
.low {background-color:#ddffdd;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------
# LOAD CLASSES
# -----------------------
@st.cache_data
def load_classes():
    with open("class_names.json") as f:
        return json.load(f)

class_names = load_classes()
# -----------------------
# IMAGE PREPROCESSING
# -----------------------
def preprocess(image):

    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img
# -----------------------
# KNOWLEDGE BASE (DETAILED FARMER ADVICE)
# -----------------------
knowledge_base = {
    "Leaf Blight": {
        "Symptoms": "Yellowing of leaf edges, brown patches, drying of leaves.",
        "Causes": "Fungal infection due to high humidity and poor airflow.",
        "Prevention": "Avoid overwatering, maintain spacing, use resistant seeds.",
        "Cure": "Apply fungicide like mancozeb every 7 days.",
        "Impact": "Can reduce yield significantly if untreated.",
        "Best Practices": "Remove infected leaves, improve field drainage."
    },

    "Healthy": {
        "Symptoms": "Green and healthy leaves with no visible damage.",
        "Causes": "Proper growth conditions.",
        "Prevention": "Maintain regular irrigation and fertilization.",
        "Cure": "No treatment required.",
        "Impact": "Optimal crop yield.",
        "Best Practices": "Continue standard farming practices."
    }
}

st.write(label)

# -----------------------
# SAFE TRANSLATION (HEADINGS ONLY)
# -----------------------
LANG = {
    "Tamil": {
        "Prediction Result": "முன்கணிப்பு முடிவு",
        "Disease": "நோய்",
        "Confidence": "நம்பிக்கை",
        "Severity": "ஆபத்து நிலை",
        "Instant Treatment": "உடனடி சிகிச்சை",
        "Environmental Conditions": "சுற்றுச்சூழல் நிலை",
        "Temperature": "வெப்பநிலை",
        "Humidity": "ஈரப்பதம்",
        "Spray Interval": "தெளிப்பு இடைவெளி",
        "Farmer Advice": "விவசாயி ஆலோசனை",
        "Top Predictions": "முன்னணி கணிப்புகள்",
        "Symptoms": "அறிகுறிகள்",
        "Causes": "காரணங்கள்",
        "Prevention": "தடுப்பு",
        "Cure": "சிகிச்சை",
        "Impact": "பாதிப்பு",
        "Best Practices": "சிறந்த நடைமுறைகள்"
    },

    "Hindi": {
        "Prediction Result": "पूर्वानुमान परिणाम",
        "Disease": "रोग",
        "Confidence": "विश्वास स्तर",
        "Severity": "गंभीरता",
        "Instant Treatment": "तुरंत उपचार",
        "Environmental Conditions": "पर्यावरण स्थिति",
        "Temperature": "तापमान",
        "Humidity": "नमी",
        "Spray Interval": "स्प्रे अंतराल",
        "Farmer Advice": "किसान सलाह",
        "Top Predictions": "शीर्ष पूर्वानुमान",
        "Symptoms": "लक्षण",
        "Causes": "कारण",
        "Prevention": "रोकथाम",
        "Cure": "उपचार",
        "Impact": "प्रभाव",
        "Best Practices": "सर्वोत्तम तरीके"
    }
}


def t(key, lang):
    if lang == "English":
        return key
    return LANG.get(lang, {}).get(key, key)


# -----------------------
# FARMER ADVICE FUNCTION (FIXED)
# -----------------------
def get_advice(label, language):

    # normalize label
    clean_label = label.replace("_", " ").lower()

    for key in knowledge_base:
        if key.lower() in clean_label:
            info = knowledge_base[key]
            break
    else:
        return "No detailed advice available."

    return f"""
### {t("Farmer Advice", language)}

🔍 {t("Symptoms", language)}:
{info['Symptoms']}

⚠️ {t("Causes", language)}:
{info['Causes']}

🛡️ {t("Prevention", language)}:
{info['Prevention']}

💊 {t("Cure", language)}:
{info['Cure']}

📉 {t("Impact", language)}:
{info['Impact']}

🌱 {t("Best Practices", language)}:
{info['Best Practices']}
"""
# -----------------------
# WEATHER API
# -----------------------
def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        data = requests.get(url, timeout=5).json()
        humidity = float(data["current_condition"][0]["humidity"])
        temp = float(data["current_condition"][0]["temp_C"])
    except:
        humidity, temp = 60.0, 25.0
    return humidity, temp


# -----------------------
# WEATHER UI (CARDS)
# -----------------------
def show_weather_ui(temp, humidity):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
        🌡️ <h3>{temp}°C</h3>
        Temperature
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        💧 <h3>{humidity}%</h3>
        Humidity
        </div>
        """, unsafe_allow_html=True)


# -----------------------
# SCIENTIFIC RULES
# -----------------------
def scientific_rules(humidity, temp):
    notes = []

    if humidity > 80:
        notes.append("High humidity → fungal disease risk")

    if temp > 35:
        notes.append("High temperature → plant stress")

    if temp < 15:
        notes.append("Low temperature → slow recovery")

    return notes


# -----------------------
# SEVERITY SYSTEM
# -----------------------
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


# -----------------------
# SEVERITY CARD
# -----------------------
def show_severity_card(level, color, message):

    st.markdown(f"""
    <div class="card {color}">
    <h2>⚠️ {level}</h2>
    <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------
# RISK ALERT
# -----------------------
def show_risk_alert(level, confidence):

    if level == "HIGH":
        st.error(f"🚨 High Risk! Confidence: {confidence:.2f}")

    elif level == "MEDIUM":
        st.warning(f"⚠️ Moderate Risk! Confidence: {confidence:.2f}")

    else:
        st.success(f"✅ Low Risk. Confidence: {confidence:.2f}")


# -----------------------
# SPRAY SCHEDULING
# -----------------------
def spray_schedule(humidity, level):

    if level == "HIGH":
        return 5
    elif humidity > 80:
        return 7
    elif humidity > 60:
        return 10
    else:
        return 14
# -----------------------
# SESSION TRACKING
# -----------------------
if "session_conf" not in st.session_state:
    st.session_state.session_conf = []
    st.session_state.session_time = []


def update_session(conf):
    if len(st.session_state.session_conf) >= 20:
        st.session_state.session_conf.pop(0)
        st.session_state.session_time.pop(0)

    st.session_state.session_conf.append(conf)
    st.session_state.session_time.append(time.strftime("%H:%M:%S"))


# -----------------------
# PLOTLY TREND GRAPH
# -----------------------
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


# -----------------------
# PLOTLY TOP PREDICTIONS
# -----------------------
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


# -----------------------
# FORMAT TOP TEXT
# -----------------------
def format_top_predictions(output, labels, top_idx):
    return "\n".join([
        f"{labels[i]} → {output[i]:.2f}" for i in top_idx
    ])


# -----------------------
# SMART FARM CALCULATOR
# -----------------------
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
# -----------------------
# FINAL PREDICTION FUNCTION
# -----------------------
def predict(image, city, area, language):

    try:
        if image is None:
            return "❌ No image uploaded", None

        # Reset file pointer
        image.seek(0)

        # Preprocess
        img = preprocess(image)

        # MODEL INFERENCE
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if len(output.shape) == 2:
            output = output[0]

        safe_classes = class_names[:len(output)]

        # Prediction
        top_idx_main = int(np.argmax(output))
        confidence = float(output[top_idx_main])
        label = safe_classes[top_idx_main]

        # Weather
        humidity, temp = get_weather(city)

        # Severity
        level, color, message, notes = get_severity(confidence, humidity, temp)

        # Spray
        spray_days = spray_schedule(humidity, level)

        # Advice
        advice = get_advice(label, language)

        # Farm plan
        farm_info = farm_calculator(area, humidity, temp)

        # Charts
        fig_bar, top_idx = plot_top_predictions(output, safe_classes)
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
        return {
            "label": "Error",
            "confidence": 0,
            "level": "Unknown",
            "color": "gray",
            "message": str(e),
            "notes": [],
            "spray": "-",
            "humidity": "-",
            "temp": "-",
            "advice": "Something went wrong",
            "farm": "",
            "fig_bar": None,
            "fig_trend": None
        }
# -----------------------
# SIDEBAR CONTROLS
# -----------------------
st.sidebar.header("⚙️ Controls")

language = st.sidebar.selectbox("🌐 Language", ["English", "Tamil", "Hindi"])
city = st.sidebar.text_input("📍 City", "Chennai")
area = st.sidebar.number_input("🌾 Farm Area (acres)", value=1.0)

farmer_mode = st.sidebar.checkbox("👨‍🌾 Farmer Mode", True)

# -----------------------
# IMAGE UPLOAD
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)
camera_image = st.camera_input("📸 Or Take Photo")

# Priority logic
if camera_image is not None:
    uploaded_file = camera_image
if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)

    if st.button("🚀 Analyze Crop"):

        with st.spinner("Analyzing crop..."):

            result = predict(uploaded_file, city, area, language)

        # DASHBOARD
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='card'><h3>{result['label']}</h3>Disease</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card'><h3>{result['confidence']:.2f}</h3>Confidence</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='card {result['color']}'><h3>{result['level']}</h3>Severity</div>", unsafe_allow_html=True)

        show_weather_ui(result["temp"], result["humidity"])
        show_risk_alert(result["level"], result["confidence"])

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Prediction",
            "👨‍🌾 Advice",
            "📊 Analytics",
            "🧮 Farm Tools"
        ])

        with tab1:
            show_severity_card(result["level"], result["color"], result["message"])
            for note in result["notes"]:
                st.info(note)
            st.write(f"💊 Spray Interval: {result['spray']} days")

        with tab2:
            st.markdown(result["advice"])

        with tab3:
            if result["fig_bar"]:
                st.plotly_chart(result["fig_bar"], use_container_width=True)
            if result["fig_trend"]:
                st.plotly_chart(result["fig_trend"], use_container_width=True)

        with tab4:
            st.markdown(result["farm"])

        if farmer_mode:
            st.success("👨‍🌾 Farmer Mode Enabled")

else:
    st.warning("Please upload an image")
