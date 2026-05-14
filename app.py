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
import plotly.graph_objects as go
import tensorflow as tf
import time
import requests
from datetime import datetime, timedelta

# =========================
# 🌐 FULL MULTILINGUAL TRANSLATIONS
# =========================
translations = {
    "English": {
        "title": "🌱 AgroVision AI — Smart Agriculture System",
        "dashboard": "🌾 Smart Farm Dashboard",
        "controls": "⚙️ Controls",
        "language": "🌐 Language",
        "city": "📍 City",
        "area": "🌾 Farm Area (acres)",
        "farmer_mode": "👨‍🌾 Farmer Mode",
        "upload": "Upload Leaf Image",
        "camera": "📸 Or Take Photo",
        "analyze": "🚀 Analyze Crop",
        "prediction": "🔬 Prediction",
        "advice": "👨‍🌾 Advice",
        "analytics": "📊 Analytics",
        "farm_tools": "🧮 Farm Tools",
        "confidence": "Confidence",
        "severity": "Severity",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "download": "📄 Download Report",
        "status": "Status",
        "no_image": "Please upload or capture an image",
        "processing": "Analyzing crop...",
        "trend": "📈 Prediction Trend",
        "top_pred": "📊 Top Predictions",
        "confidence_level": "📊 Confidence Level",
        "confidence_meter": "🔋 Confidence Meter",
        "history": "📈 Confidence History",
        "farmer_enabled": "👨‍🌾 Farmer Mode Enabled",
        "disease": "Disease",
        "spray_interval": "💊 Spray Interval",
        "days": "days",
        "detailed_advice": "📖 Detailed Farmer Advice",
        "high_humidity_warn": "⚠ High humidity → Disease risk",
        "high_temp_warn": "🔥 High temperature → Stress risk",
        "fungal_risk": "High humidity → fungal disease risk",
        "heat_stress": "High temperature → plant stress",
        "slow_recovery": "Low temperature → slow recovery",
        "immediate_action": "Immediate action required!",
        "monitor_closely": "Monitor closely",
        "safe_condition": "Safe condition",
        "high_risk_msg": "🚨 High Risk! Confidence",
        "moderate_risk_msg": "⚠️ Moderate Risk! Confidence",
        "low_risk_msg": "✅ Low Risk. Confidence",
        "spray_tip": "📌 Tip: Spray during early morning or evening",
        "offline_weather": "🌐 Offline Mode: Using simulated weather for",
        "report_title": "AgroVision Report",
        "report_disease": "Disease",
        "report_confidence": "Confidence",
        "report_severity": "Severity",
        "report_advice": "Advice",
        "smart_farm_plan": "🌾 Smart Farm Plan",
        "crop_label": "Crop",
        "area_label": "Area",
        "irrigation_label": "💧 Irrigation",
        "spray_label": "📆 Spray Interval",
        "fertilizer_label": "🌱 Fertilizer",
        "upload_warning": "Please upload or capture an image",
        "general_advice": "General Advice",
        "monitor": "Monitor crop regularly",
        "maintain_irr": "Maintain irrigation",
        "remove_infected": "Remove infected leaves",
        "use_pesticide": "Use proper pesticide",
        "detected": "Detected",
        "high_conf_note": "🚨 High confidence → Act immediately",
        "moderate_conf_note": "⚠ Moderate confidence → Monitor closely",
        "low_conf_note": "✅ Low risk",
        "humidity_fungal": "⚠ High humidity → fungal risk",
        "heat_risk": "🔥 Heat stress risk",
        "enter_crop": "🌱 Enter Crop Name",
        "acres": "acres",
        "liters": "L",
        "kg": "kg",
    },

    "Tamil": {
        "title": "🌱 அக்ரோவிஷன் AI — ஸ்மார்ட் வேளாண்மை",
        "dashboard": "🌾 ஸ்மார்ட் பண்ணை டாஷ்போர்டு",
        "controls": "⚙️ கட்டுப்பாடுகள்",
        "language": "🌐 மொழி",
        "city": "📍 நகரம்",
        "area": "🌾 பண்ணை பரப்பு (ஏக்கர்)",
        "farmer_mode": "👨‍🌾 விவசாயி முறை",
        "upload": "இலை படத்தை பதிவேற்றவும்",
        "camera": "📸 படம் எடுக்கவும்",
        "analyze": "🚀 பயிர் பகுப்பாய்வு",
        "prediction": "🔬 கணிப்பு",
        "advice": "👨‍🌾 ஆலோசனை",
        "analytics": "📊 பகுப்பாய்வு",
        "farm_tools": "🧮 பண்ணை கருவிகள்",
        "confidence": "நம்பிக்கை",
        "severity": "தீவிரம்",
        "temperature": "வெப்பநிலை",
        "humidity": "ஈரப்பதம்",
        "download": "📄 அறிக்கை பதிவிறக்கம்",
        "status": "நிலை",
        "no_image": "படம் பதிவேற்றவும்",
        "processing": "பகுப்பாய்வு செய்யப்படுகிறது...",
        "trend": "📈 கணிப்பு வரலாறு",
        "top_pred": "📊 முன்னணி கணிப்புகள்",
        "confidence_level": "📊 நம்பிக்கை நிலை",
        "confidence_meter": "🔋 நம்பிக்கை மீட்டர்",
        "history": "📈 வரலாறு",
        "farmer_enabled": "👨‍🌾 விவசாயி முறை செயல்படுத்தப்பட்டது",
        "disease": "நோய்",
        "spray_interval": "💊 தெளிப்பு இடைவெளி",
        "days": "நாட்கள்",
        "detailed_advice": "📖 விரிவான விவசாயி ஆலோசனை",
        "high_humidity_warn": "⚠ அதிக ஈரப்பதம் → நோய் அபாயம்",
        "high_temp_warn": "🔥 அதிக வெப்பநிலை → அழுத்த அபாயம்",
        "fungal_risk": "அதிக ஈரப்பதம் → பூஞ்சை நோய் அபாயம்",
        "heat_stress": "அதிக வெப்பநிலை → தாவர அழுத்தம்",
        "slow_recovery": "குறைந்த வெப்பநிலை → மெதுவான குணமடைதல்",
        "immediate_action": "உடனடி நடவடிக்கை தேவை!",
        "monitor_closely": "நெருக்கமாக கண்காணிக்கவும்",
        "safe_condition": "பாதுகாப்பான நிலை",
        "high_risk_msg": "🚨 அதிக ஆபத்து! நம்பிக்கை",
        "moderate_risk_msg": "⚠️ மிதமான ஆபத்து! நம்பிக்கை",
        "low_risk_msg": "✅ குறைந்த ஆபத்து. நம்பிக்கை",
        "spray_tip": "📌 குறிப்பு: அதிகாலை அல்லது மாலையில் தெளிக்கவும்",
        "offline_weather": "🌐 ஆஃப்லைன் முறை: உருவகப்படுத்தப்பட்ட வானிலை பயன்படுத்தப்படுகிறது",
        "report_title": "அக்ரோவிஷன் அறிக்கை",
        "report_disease": "நோய்",
        "report_confidence": "நம்பிக்கை",
        "report_severity": "தீவிரம்",
        "report_advice": "ஆலோசனை",
        "smart_farm_plan": "🌾 ஸ்மார்ட் பண்ணை திட்டம்",
        "crop_label": "பயிர்",
        "area_label": "பரப்பு",
        "irrigation_label": "💧 நீர்ப்பாசனம்",
        "spray_label": "📆 தெளிப்பு இடைவெளி",
        "fertilizer_label": "🌱 உரம்",
        "upload_warning": "படம் பதிவேற்றவும் அல்லது எடுக்கவும்",
        "general_advice": "பொது ஆலோசனை",
        "monitor": "பயிரை தொடர்ந்து கண்காணிக்கவும்",
        "maintain_irr": "நீர்ப்பாசனத்தை பராமரிக்கவும்",
        "remove_infected": "பாதிக்கப்பட்ட இலைகளை அகற்றவும்",
        "use_pesticide": "சரியான பூச்சிக்கொல்லி பயன்படுத்தவும்",
        "detected": "கண்டறியப்பட்டது",
        "high_conf_note": "🚨 அதிக நம்பிக்கை → உடனடியாக நடவடிக்கை எடுக்கவும்",
        "moderate_conf_note": "⚠ மிதமான நம்பிக்கை → நெருக்கமாக கண்காணிக்கவும்",
        "low_conf_note": "✅ குறைந்த ஆபத்து",
        "humidity_fungal": "⚠ அதிக ஈரப்பதம் → பூஞ்சை அபாயம்",
        "heat_risk": "🔥 வெப்ப அழுத்த அபாயம்",
        "enter_crop": "🌱 பயிர் பெயர் உள்ளிடவும்",
        "acres": "ஏக்கர்",
        "liters": "லி",
        "kg": "கி.கி",
    },

    "Hindi": {
        "title": "🌱 एग्रोविजन AI — स्मार्ट कृषि",
        "dashboard": "🌾 स्मार्ट फार्म डैशबोर्ड",
        "controls": "⚙️ नियंत्रण",
        "language": "🌐 भाषा",
        "city": "📍 शहर",
        "area": "🌾 खेत क्षेत्र (एकड़)",
        "farmer_mode": "👨‍🌾 किसान मोड",
        "upload": "पत्ती की छवि अपलोड करें",
        "camera": "📸 फोटो लें",
        "analyze": "🚀 फसल विश्लेषण",
        "prediction": "🔬 भविष्यवाणी",
        "advice": "👨‍🌾 सलाह",
        "analytics": "📊 विश्लेषण",
        "farm_tools": "🧮 फार्म टूल्स",
        "confidence": "विश्वास",
        "severity": "गंभीरता",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "download": "📄 रिपोर्ट डाउनलोड",
        "status": "स्थिति",
        "no_image": "कृपया छवि अपलोड करें",
        "processing": "विश्लेषण हो रहा है...",
        "trend": "📈 भविष्यवाणी इतिहास",
        "top_pred": "📊 शीर्ष भविष्यवाणियाँ",
        "confidence_level": "📊 विश्वास स्तर",
        "confidence_meter": "🔋 विश्वास मीटर",
        "history": "📈 इतिहास",
        "farmer_enabled": "👨‍🌾 किसान मोड सक्रिय",
        "disease": "रोग",
        "spray_interval": "💊 छिड़काव अंतराल",
        "days": "दिन",
        "detailed_advice": "📖 विस्तृत किसान सलाह",
        "high_humidity_warn": "⚠ उच्च आर्द्रता → रोग का खतरा",
        "high_temp_warn": "🔥 उच्च तापमान → तनाव का खतरा",
        "fungal_risk": "उच्च आर्द्रता → फफूंद रोग का खतरा",
        "heat_stress": "उच्च तापमान → पौधे का तनाव",
        "slow_recovery": "कम तापमान → धीमी रिकवरी",
        "immediate_action": "तत्काल कार्रवाई आवश्यक!",
        "monitor_closely": "ध्यान से निगरानी करें",
        "safe_condition": "सुरक्षित स्थिति",
        "high_risk_msg": "🚨 उच्च जोखिम! विश्वास",
        "moderate_risk_msg": "⚠️ मध्यम जोखिम! विश्वास",
        "low_risk_msg": "✅ कम जोखिम। विश्वास",
        "spray_tip": "📌 सुझाव: सुबह जल्दी या शाम को छिड़काव करें",
        "offline_weather": "🌐 ऑफलाइन मोड: अनुकरण मौसम उपयोग किया जा रहा है",
        "report_title": "एग्रोविजन रिपोर्ट",
        "report_disease": "रोग",
        "report_confidence": "विश्वास",
        "report_severity": "गंभीरता",
        "report_advice": "सलाह",
        "smart_farm_plan": "🌾 स्मार्ट फार्म योजना",
        "crop_label": "फसल",
        "area_label": "क्षेत्र",
        "irrigation_label": "💧 सिंचाई",
        "spray_label": "📆 छिड़काव अंतराल",
        "fertilizer_label": "🌱 उर्वरक",
        "upload_warning": "कृपया छवि अपलोड या कैप्चर करें",
        "general_advice": "सामान्य सलाह",
        "monitor": "फसल की नियमित निगरानी करें",
        "maintain_irr": "सिंचाई बनाए रखें",
        "remove_infected": "संक्रमित पत्तियाँ हटाएं",
        "use_pesticide": "उचित कीटनाशक का उपयोग करें",
        "detected": "पहचाना गया",
        "high_conf_note": "🚨 उच्च विश्वास → तुरंत कार्रवाई करें",
        "moderate_conf_note": "⚠ मध्यम विश्वास → ध्यान से निगरानी करें",
        "low_conf_note": "✅ कम जोखिम",
        "humidity_fungal": "⚠ उच्च आर्द्रता → फफूंद का खतरा",
        "heat_risk": "🔥 गर्मी तनाव का खतरा",
        "enter_crop": "🌱 फसल का नाम दर्ज करें",
        "acres": "एकड़",
        "liters": "लीटर",
        "kg": "किग्रा",
    }
}

# =========================
# KNOWLEDGE BASE (MULTILINGUAL)
# =========================
knowledge_base_ml = {
    "English": {},   # loaded from file
    "Tamil": {
        "_default": {
            "Symptoms": "இலைகளில் மஞ்சள் அல்லது பழுப்பு நிற புள்ளிகள், வாடுதல், அழுகல்",
            "Causes": "பூஞ்சை, பாக்டீரியா அல்லது வைரஸ் தொற்று",
            "Prevention": "சரியான நீர்ப்பாசனம், நல்ல காற்றோட்டம், சுத்தமான கருவிகள்",
            "Cure": "பூஞ்சை எதிர்ப்பு மருந்து அல்லது பூச்சிக்கொல்லி தெளிக்கவும்",
            "Impact": "அதிக மகசூல் இழப்பு ஏற்படலாம்",
            "Best Practices": "அதிகாலையில் தெளிக்கவும், பாதிக்கப்பட்ட இலைகளை எரிக்கவும்"
        }
    },
    "Hindi": {
        "_default": {
            "Symptoms": "पत्तियों पर पीले या भूरे धब्बे, मुरझाना, सड़न",
            "Causes": "फफूंद, बैक्टीरिया या वायरस संक्रमण",
            "Prevention": "उचित सिंचाई, अच्छा वायु संचार, साफ उपकरण",
            "Cure": "एंटीफंगल या कीटनाशक का छिड़काव करें",
            "Impact": "उपज में भारी नुकसान हो सकता है",
            "Best Practices": "सुबह जल्दी छिड़काव करें, संक्रमित पत्तियाँ जलाएं"
        }
    }
}

def t(key):
    return translations[language].get(key, key)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AgroVision AI PRO", layout="wide")
language = st.sidebar.selectbox("🌐 Language", ["English", "Tamil", "Hindi"])

# =========================
# 🌙 DARK MODE TOGGLE
# =========================
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# =========================
# 🎨 PRO CSS
# =========================
st.markdown(f"""
<style>

/* ===== BACKGROUND ===== */
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

/* ===== SIDEBAR TEXT ===== */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: white !important;
}}

/* ===== DROPDOWN FIX ===== */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
    color: black !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] {{
    background-color: white !important;
    border-radius: 8px;
}}

section[data-testid="stSidebar"] ul[role="listbox"] li {{
    color: black !important;
    background-color: white !important;
}}

section[data-testid="stSidebar"] ul[role="listbox"] li:hover {{
    background-color: #e8f5e9 !important;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
    color: black !important;
}}

/* ===== SIDEBAR BASE ===== */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#1b5e20,#2e7d32);
    position: relative;
    overflow: hidden;
}}

/* ===== INPUT FIX ===== */
.stTextInput input,
.stNumberInput input,
textarea {{
    background-color: {"#222" if dark_mode else "#ffffff"} !important;
    color: {"white" if dark_mode else "black"} !important;
    border-radius: 10px !important;
}}

/* ===== GLASS UI ===== */
.glass {{
    background: {"rgba(30,30,30,0.7)" if dark_mode else "rgba(255,255,255,0.15)"};
    color: {"white" if dark_mode else "black"};
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    transition: 0.3s;
}}

.glass:hover {{
    transform: translateY(-8px);
}}

/* ===== CARDS ===== */
.card {{
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    backdrop-filter: blur(12px);
    background: {"rgba(40,40,40,0.85)" if dark_mode else "rgba(255,255,255,0.2)"};
    color: {"white" if dark_mode else "black"};
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}}

.card p {{
    color: white !important;
}}

.card h3 {{
    color: white !important;
}}

.card1 {{ border-left:6px solid #22c55e; }}
.card2 {{ border-left:6px solid #3b82f6; }}
.card3 {{ border-left:6px solid #f97316; }}

/* ===== TEXT VISIBILITY ===== */
body, div, span {{
    color: {"white" if dark_mode else "black"} !important;
}}

/* ===== HEADINGS ===== */
h1, h2, h3 {{
    color: {"#ffffff" if dark_mode else "#1b5e20"};
}}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title(t("title"))

# =========================
# 🌾 DASHBOARD
# =========================
st.markdown(f'<div class="glass"><h2>{t("dashboard")}</h2></div>', unsafe_allow_html=True)

d = st.session_state.get("last_result", {})

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="card card1"><h3>{d.get("level","--")}</h3><p>{t("status")}</p></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card card2"><h3>{d.get("confidence",0):.2f}</h3><p>{t("confidence")}</p></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card card3"><h3>{d.get("temp","--")}°C</h3><p>{t("temperature")}</p></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="card card1"><h3>{d.get("humidity","--")}%</h3><p>{t("humidity")}</p></div>', unsafe_allow_html=True)

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
# LOAD KNOWLEDGE BASE
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
# 🌍 CITY COORDINATES LOOKUP
# Common cities — extendable
# =========================
CITY_COORDS = {
    "chennai":   (13.0827, 80.2707),
    "delhi":     (28.6139, 77.2090),
    "mumbai":    (19.0760, 72.8777),
    "kolkata":   (22.5726, 88.3639),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "pune":      (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur":    (26.9124, 75.7873),
    "lucknow":   (26.8467, 80.9462),
    "coimbatore":(11.0168, 76.9558),
    "madurai":   (9.9252,  78.1198),
    "trichy":    (10.7905, 78.7047),
    "salem":     (11.6643, 78.1460),
    "vellore":   (12.9165, 79.1325),
}

WEATHER_CACHE_FILE = "weather_cache.json"
CACHE_EXPIRY_HOURS = 6  # Refresh every 6 hours when online

# =========================
# 📦 CACHE HELPERS
# =========================
def load_weather_cache():
    try:
        with open(WEATHER_CACHE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_weather_cache(cache):
    try:
        with open(WEATHER_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except:
        pass

def is_cache_fresh(entry):
    try:
        cached_time = datetime.fromisoformat(entry["timestamp"])
        return datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRY_HOURS)
    except:
        return False

# =========================
# 🌐 OPEN-METEO FETCH (online)
# Free, no API key required
# =========================
def fetch_openmeteo(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m"
            f"&forecast_days=1"
        )
        resp = requests.get(url, timeout=5)
        data = resp.json()
        temp = float(data["current"]["temperature_2m"])
        humidity = float(data["current"]["relative_humidity_2m"])
        return temp, humidity
    except:
        return None, None

# =========================
# 🌤 MAIN WEATHER FUNCTION
# Online → fetches + caches | Offline → serves from cache
# =========================
def get_weather(city):
    city_key = city.strip().lower()
    cache = load_weather_cache()

    # ✅ Return fresh cache if available
    if city_key in cache and is_cache_fresh(cache[city_key]):
        entry = cache[city_key]
        return entry["humidity"], entry["temp"], "cache"

    # 🌍 Get coordinates
    coords = CITY_COORDS.get(city_key)
    if coords is None:
        # Try geocoding via Open-Meteo's geocoding API
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            geo = requests.get(geo_url, timeout=5).json()
            result = geo["results"][0]
            coords = (result["latitude"], result["longitude"])
        except:
            coords = None

    if coords:
        lat, lon = coords
        temp, humidity = fetch_openmeteo(lat, lon)

        if temp is not None:
            # ✅ Online fetch succeeded → update cache
            cache[city_key] = {
                "temp": round(temp, 1),
                "humidity": round(humidity, 1),
                "timestamp": datetime.now().isoformat(),
                "source": "live"
            }
            save_weather_cache(cache)
            return round(humidity, 1), round(temp, 1), "live"

    # 📦 Offline fallback → use stale cache if exists
    if city_key in cache:
        entry = cache[city_key]
        return entry["humidity"], entry["temp"], "stale"

    # 🔴 No cache, no internet → regional default
    return 65.0, 28.0, "default"

# =========================
# WEATHER UI
# =========================
def show_weather_ui(temp, humidity, source="live"):
    source_labels = {
        "live":    "🟢 Live weather",
        "cache":   "🟡 Cached weather (fresh)",
        "stale":   "🟠 Cached weather (last known)",
        "default": "🔴 Default estimate (city not found)"
    }
    st.caption(source_labels.get(source, ""))
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"🌡 {t('temperature')}", f"{temp}°C")
    with col2:
        st.metric(f"💧 {t('humidity')}", f"{humidity}%")
    if humidity > 80:
        st.warning(t("high_humidity_warn"))
    if temp > 35:
        st.warning(t("high_temp_warn"))

# =========================
# SCIENTIFIC RULES
# =========================
def scientific_rules(humidity, temp):
    notes = []

    if humidity > 80:
        notes.append(t("fungal_risk"))

    if temp > 35:
        notes.append(t("heat_stress"))

    if temp < 15:
        notes.append(t("slow_recovery"))

    return notes

# =========================
# SEVERITY SYSTEM
# =========================
def get_severity(conf, humidity, temp):
    if conf >= 0.7 or humidity > 80:
        level = t("immediate_action").split("!")[0].strip() if language != "English" else "HIGH"
        color = "card3"
        message = t("immediate_action")
    elif conf >= 0.4 or humidity > 65:
        level = t("monitor_closely") if language != "English" else "MEDIUM"
        color = "card2"
        message = t("monitor_closely")
    else:
        level = t("safe_condition") if language != "English" else "LOW"
        color = "card1"
        message = t("safe_condition")

    # Use simple English level keys for dashboard display consistency
    if conf >= 0.7 or humidity > 80:
        level_key = "HIGH"
    elif conf >= 0.4 or humidity > 65:
        level_key = "MEDIUM"
    else:
        level_key = "LOW"

    notes = scientific_rules(humidity, temp)
    return level_key, color, message, notes

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
        st.error(f"{t('high_risk_msg')}: {confidence:.2f}")
    elif level == "MEDIUM":
        st.warning(f"{t('moderate_risk_msg')}: {confidence:.2f}")
    else:
        st.success(f"{t('low_risk_msg')}: {confidence:.2f}")

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
# SMART ADVICE ENGINE (OFFLINE + MULTILINGUAL)
# =========================
def get_advice(label, lang, humidity, temp, confidence):
    # Try English knowledge base first
    info = knowledge_base.get(label)

    if not info:
        # No entry in knowledge_base → use generic multilingual advice
        base = f"""
{t('general_advice')}:
- {t('monitor')}
- {t('maintain_irr')}
- {t('remove_infected')}
- {t('use_pesticide')}

{t('detected')}: {label}
"""
    else:
        # English info — present directly for English, or use built-in field translations for other langs
        if lang == "English":
            base = f"""
Symptoms: {info.get('Symptoms', 'N/A')}
Causes: {info.get('Causes', 'N/A')}
Prevention: {info.get('Prevention', 'N/A')}
Cure: {info.get('Cure', 'N/A')}
Impact: {info.get('Impact', 'N/A')}
Best Practices: {info.get('Best Practices', 'N/A')}
"""
        elif lang == "Tamil":
            default = knowledge_base_ml["Tamil"].get("_default", {})
            base = f"""
அறிகுறிகள்: {info.get('Symptoms', default.get('Symptoms', 'N/A'))}
காரணங்கள்: {info.get('Causes', default.get('Causes', 'N/A'))}
தடுப்பு: {info.get('Prevention', default.get('Prevention', 'N/A'))}
சிகிச்சை: {info.get('Cure', default.get('Cure', 'N/A'))}
தாக்கம்: {info.get('Impact', default.get('Impact', 'N/A'))}
சிறந்த நடைமுறைகள்: {info.get('Best Practices', default.get('Best Practices', 'N/A'))}
"""
        elif lang == "Hindi":
            default = knowledge_base_ml["Hindi"].get("_default", {})
            base = f"""
लक्षण: {info.get('Symptoms', default.get('Symptoms', 'N/A'))}
कारण: {info.get('Causes', default.get('Causes', 'N/A'))}
रोकथाम: {info.get('Prevention', default.get('Prevention', 'N/A'))}
उपचार: {info.get('Cure', default.get('Cure', 'N/A'))}
प्रभाव: {info.get('Impact', default.get('Impact', 'N/A'))}
सर्वोत्तम अभ्यास: {info.get('Best Practices', default.get('Best Practices', 'N/A'))}
"""
        else:
            base = f"""
Symptoms: {info.get('Symptoms', 'N/A')}
"""

    # Multilingual weather & confidence notes
    weather_note = ""
    if humidity > 80:
        weather_note += f"\n{t('humidity_fungal')}"
    if temp > 35:
        weather_note += f"\n{t('heat_risk')}"

    confidence_note = ""
    if confidence > 0.8:
        confidence_note = f"\n{t('high_conf_note')}"
    elif confidence > 0.5:
        confidence_note = f"\n{t('moderate_conf_note')}"
    else:
        confidence_note = f"\n{t('low_conf_note')}"

    return base + weather_note + confidence_note

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
        title=t("trend"),
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
    fig.update_layout(title=t("top_pred"))
    return fig, top_idx

# =========================
# NPK & WATER DATA
# =========================
crop_npk = {
    "Tomato": (100, 50, 50),
    "Potato": (180, 80, 100),
    "Corn": (150, 70, 50),
    "Maize": (150, 70, 50),
    "Apple": (70, 40, 60),
    "Grape": (100, 50, 80),
    "Pepper": (120, 60, 60),
    "Bell Pepper": (120, 60, 60),
    "Cherry": (80, 40, 60),
    "Peach": (90, 50, 70),
    "Strawberry": (100, 50, 80),
    "Soybean": (20, 60, 40),
    "Wheat": (100, 50, 40),
    "Rice": (120, 60, 40)
}
crop_water = {
    "Tomato": 3000,
    "Potato": 3200,
    "Corn": 3500,
    "Maize": 3500,
    "Apple": 2800,
    "Grape": 2500,
    "Pepper": 3000,
    "Bell Pepper": 3000,
    "Cherry": 2800,
    "Peach": 2900,
    "Strawberry": 2700,
    "Soybean": 2200,
    "Wheat": 3000,
    "Rice": 5000
}

# =========================
# FARM CALCULATOR
# =========================
def farm_calculator(area, humidity, temp, crop, soil_moisture=25):
    area = float(area)
    N, P, K = crop_npk.get(crop, (60, 40, 40))
    irrigation = crop_water.get(crop, 3000)

    if temp > 32:
        irrigation += 800
    if humidity < 40:
        irrigation += 500
    if soil_moisture > 35:
        irrigation -= 700

    irrigation = max(1500, irrigation)
    spray = 7 if humidity > 80 else 10 if humidity > 60 else 14

    return f"""
{t('smart_farm_plan')}

{t('crop_label')}: {crop}

{t('area_label')}: {area} {t('acres')}

{t('irrigation_label')}: {int(irrigation * area)} {t('liters')}

{t('spray_label')}: {spray} {t('days')}

{t('fertilizer_label')}:
N: {int(N * area)} {t('kg')}
P: {int(P * area)} {t('kg')}
K: {int(K * area)} {t('kg')}
"""

# =========================
# GAUGE CHART
# =========================
def show_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': t("confidence_meter")},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header(t("controls"))
city = st.sidebar.text_input(t("city"), "Chennai")
area = st.sidebar.number_input(t("area"), value=1.0)
crop = st.sidebar.text_input(t("enter_crop"), "Rice")
farmer_mode = st.sidebar.checkbox(t("farmer_mode"), True)

# =========================
# IMAGE INPUT
# =========================
uploaded_file = st.file_uploader(
    t("upload"),
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input(t("camera"))

if camera_image is not None:
    uploaded_file = camera_image

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image, city, area, lang, crop):
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

        # 🌤 SMART WEATHER — live fetch with offline cache fallback
        humidity, temp, weather_source = get_weather(city)

        level, color, message, notes = get_severity(confidence, humidity, temp)
        spray_days = spray_schedule(humidity, level)
        advice = get_advice(label, lang, humidity, temp, confidence)
        farm_info = farm_calculator(area, humidity, temp, crop)

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
            "fig_trend": fig_trend,
            "weather_source": weather_source
        }

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# =========================
# MAIN EXECUTION
# =========================
if uploaded_file is not None:

    st.image(uploaded_file, caption=t("upload"), use_container_width=True)

    if st.button(t("analyze")):
        with st.spinner(t("processing")):
            result = predict(uploaded_file, city, area, language, crop)

        if result:

            # 🔥 STORE FOR DASHBOARD
            st.session_state.last_result = result

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
                    <p>{t('disease')}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card card2">
                    <h2>{confidence:.2f}</h2>
                    <p>{t('confidence')}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="card card3">
                    <h2>{severity}</h2>
                    <p>{t('severity')}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # =========================
            # 📊 CONFIDENCE + GAUGE
            # =========================
            st.subheader(t("confidence_level"))
            st.progress(min(max(confidence, 0.0), 1.0))

            st.subheader(t("confidence_meter"))
            show_gauge(confidence)

            # =========================
            # 🌦 WEATHER + ALERT
            # =========================
            show_weather_ui(result["temp"], result["humidity"], result.get("weather_source", "live"))
            show_risk_alert(result["level"], confidence)

            # =========================
            # 🧠 TABS
            # =========================
            tab1, tab2, tab3, tab4 = st.tabs([
                t("prediction"),
                t("advice"),
                t("analytics"),
                t("farm_tools")
            ])

            # TAB 1 — Prediction
            with tab1:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                show_severity_card(result["level"], result["color"], result["message"])
                for note in result["notes"]:
                    st.info(note)
                st.write(f"{t('spray_interval')}: {result['spray']} {t('days')}")
                st.markdown('</div>', unsafe_allow_html=True)

            # TAB 2 — Advice
            with tab2:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                with st.expander(t("detailed_advice"), expanded=True):
                    st.markdown(result["advice"])
                st.markdown('</div>', unsafe_allow_html=True)

            # TAB 3 — Analytics
            with tab3:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                if result["fig_bar"]:
                    st.plotly_chart(result["fig_bar"], use_container_width=True)
                if result["fig_trend"]:
                    st.plotly_chart(result["fig_trend"], use_container_width=True)
                st.subheader(t("history"))
                st.line_chart(st.session_state.session_conf)
                st.markdown('</div>', unsafe_allow_html=True)

            # TAB 4 — Farm Tools
            with tab4:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.markdown(result["farm"])
                st.markdown('</div>', unsafe_allow_html=True)

            # =========================
            # 📄 DOWNLOAD REPORT
            # =========================
            report = f"""
{t('report_title')}

{t('report_disease')}: {result['label']}
{t('report_confidence')}: {confidence:.2f}
{t('report_severity')}: {result['level']}

{t('report_advice')}:
{result['advice']}
"""
            st.download_button(
                t("download"),
                report,
                file_name="agrovision_report.txt"
            )

            # =========================
            # 👨‍🌾 FARMER MODE
            # =========================
            if farmer_mode:
                st.success(t("farmer_enabled"))
                st.info(t("spray_tip"))

else:
    st.warning(t("upload_warning"))
