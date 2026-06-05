"""
app.py — AgroVision Pro
Production-Grade Agricultural AI Diagnostic Platform
All 17 upgrade phases integrated.
"""

# ─── stdlib + env ────────────────────────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import time
import requests
from datetime import datetime, timedelta

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
st.write("TensorFlow Version:", tf.__version__)
st.write("Keras Version:", tf.keras.__version__)
from PIL import Image

# ─── local modules ────────────────────────────────────────────────────────────
from image_validation  import validate_image
from leaf_segmentation import segment_leaf
from gradcam           import generate_gradcam
from utils             import (
    preprocess_image,
    estimate_severity_from_mask,
    generate_txt_report,
    generate_csv_report,
    generate_json_report,
    top_k_predictions,
    pil_to_bytes,
)

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════
translations = {
    "English": {
        "title": "🌱 AgroVision Pro — Smart Agriculture System",
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
        "top_pred": "📊 Top 5 Predictions",
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
        "gradcam_tab": "🔬 Explainable AI",
        "image_quality": "🖼 Image Quality",
        "leaf_coverage": "🌿 Leaf Coverage",
        "image_quality_score": "Image Quality Score",
        "leaf_coverage_pct": "Leaf Coverage",
        "low_confidence_warning": "⚠ Prediction uncertain. Please upload a clearer image closer to the leaf.",
        "image_rejected": "Image rejected",
        "hotspot_analysis": "🔥 Disease Hotspot Analysis",
        "hotspot_pct": "High-activation region",
        "exports": "📤 Export Reports",
        "export_txt": "📄 Download TXT Report",
        "export_csv": "📊 Download CSV Report",
        "export_json": "🗂 Download JSON Report",
    },
    "Tamil": {
        "title": "🌱 அக்ரோவிஷன் Pro — ஸ்மார்ட் வேளாண்மை",
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
        "top_pred": "📊 முன்னணி 5 கணிப்புகள்",
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
        "offline_weather": "🌐 ஆஃப்லைன் முறை",
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
        "gradcam_tab": "🔬 விளக்க AI",
        "image_quality": "🖼 படத்தின் தரம்",
        "leaf_coverage": "🌿 இலை அளவு",
        "image_quality_score": "படத்தின் தர மதிப்பெண்",
        "leaf_coverage_pct": "இலை அளவு",
        "low_confidence_warning": "⚠ கணிப்பு நிச்சயமற்றது. தெளிவான படத்தை பதிவேற்றவும்.",
        "image_rejected": "படம் நிராகரிக்கப்பட்டது",
        "hotspot_analysis": "🔥 நோய் ஹாட்ஸ்பாட் பகுப்பாய்வு",
        "hotspot_pct": "அதிக செயல்பாட்டு பகுதி",
        "exports": "📤 அறிக்கைகள் ஏற்றுமதி",
        "export_txt": "📄 TXT அறிக்கை பதிவிறக்கம்",
        "export_csv": "📊 CSV அறிக்கை பதிவிறக்கம்",
        "export_json": "🗂 JSON அறிக்கை பதிவிறக்கம்",
    },
    "Hindi": {
        "title": "🌱 एग्रोविजन Pro — स्मार्ट कृषि",
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
        "top_pred": "📊 शीर्ष 5 भविष्यवाणियाँ",
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
        "offline_weather": "🌐 ऑफलाइन मोड",
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
        "gradcam_tab": "🔬 व्याख्यात्मक AI",
        "image_quality": "🖼 छवि गुणवत्ता",
        "leaf_coverage": "🌿 पत्ती कवरेज",
        "image_quality_score": "छवि गुणवत्ता स्कोर",
        "leaf_coverage_pct": "पत्ती कवरेज",
        "low_confidence_warning": "⚠ भविष्यवाणी अनिश्चित। कृपया स्पष्ट छवि अपलोड करें।",
        "image_rejected": "छवि अस्वीकृत",
        "hotspot_analysis": "🔥 रोग हॉटस्पॉट विश्लेषण",
        "hotspot_pct": "उच्च सक्रियण क्षेत्र",
        "exports": "📤 रिपोर्ट निर्यात",
        "export_txt": "📄 TXT रिपोर्ट डाउनलोड",
        "export_csv": "📊 CSV रिपोर्ट डाउनलोड",
        "export_json": "🗂 JSON रिपोर्ट डाउनलोड",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# MULTILINGUAL KNOWLEDGE BASE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
knowledge_base_ml = {
    "Tamil": {
        "_default": {
            "Symptoms": "இலைகளில் மஞ்சள் அல்லது பழுப்பு நிற புள்ளிகள், வாடுதல், அழுகல்",
            "Causes": "பூஞ்சை, பாக்டீரியா அல்லது வைரஸ் தொற்று",
            "Prevention": "சரியான நீர்ப்பாசனம், நல்ல காற்றோட்டம், சுத்தமான கருவிகள்",
            "Cure": "பூஞ்சை எதிர்ப்பு மருந்து அல்லது பூச்சிக்கொல்லி தெளிக்கவும்",
            "Impact": "அதிக மகசூல் இழப்பு ஏற்படலாம்",
            "Best Practices": "அதிகாலையில் தெளிக்கவும், பாதிக்கப்பட்ட இலைகளை எரிக்கவும்",
        }
    },
    "Hindi": {
        "_default": {
            "Symptoms": "पत्तियों पर पीले या भूरे धब्बे, मुरझाना, सड़न",
            "Causes": "फफूंद, बैक्टीरिया या वायरस संक्रमण",
            "Prevention": "उचित सिंचाई, अच्छा वायु संचार, साफ उपकरण",
            "Cure": "एंटीफंगल या कीटनाशक का छिड़काव करें",
            "Impact": "उपज में भारी नुकसान हो सकता है",
            "Best Practices": "सुबह जल्दी छिड़काव करें, संक्रमित पत्तियाँ जलाएं",
        }
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AgroVision Pro", layout="wide", page_icon="🌱")

# ─── language + dark-mode (must come before any t() calls) ───────────────────
language  = st.sidebar.selectbox("🌐 Language", ["English", "Tamil", "Hindi"])
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)


def t(key: str) -> str:
    return translations[language].get(key, key)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
.stApp {{
    {"background: linear-gradient(-45deg,#0f2027,#203a43,#000000);" if dark_mode else "background: linear-gradient(-45deg,#ecfdf5,#d1fae5,#bbf7d0);"}
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}}
@keyframes gradientBG {{
    0%{{background-position:0% 50%;}}
    50%{{background-position:100% 50%;}}
    100%{{background-position:0% 50%;}}
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color:white!important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] span {{ color:black!important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] {{
    background-color:white!important; border-radius:8px;}}
section[data-testid="stSidebar"] ul[role="listbox"] li {{
    color:black!important; background-color:white!important;}}
section[data-testid="stSidebar"] ul[role="listbox"] li:hover {{
    background-color:#e8f5e9!important;}}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {{ color:black!important; }}
section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#1b5e20,#2e7d32);
    position:relative; overflow:hidden;}}
.stTextInput input,.stNumberInput input,textarea {{
    background-color:{"#222" if dark_mode else "#ffffff"}!important;
    color:{"white" if dark_mode else "black"}!important; border-radius:10px!important;}}
.glass {{
    background:{"rgba(30,30,30,0.7)" if dark_mode else "rgba(255,255,255,0.15)"};
    color:{"white" if dark_mode else "black"};
    border-radius:20px; padding:20px;
    backdrop-filter:blur(14px);
    box-shadow:0 10px 40px rgba(0,0,0,0.3);
    margin-bottom:20px; transition:0.3s;}}
.glass:hover {{ transform:translateY(-8px); }}
.card {{
    padding:22px; border-radius:18px; text-align:center;
    backdrop-filter:blur(12px);
    background:{"rgba(40,40,40,0.85)" if dark_mode else "rgba(255,255,255,0.2)"};
    color:{"white" if dark_mode else "black"};
    box-shadow:0 8px 25px rgba(0,0,0,0.15);}}
.card p {{ color:white!important; }}
.card h3 {{ color:white!important; }}
.card1 {{ border-left:6px solid #22c55e; }}
.card2 {{ border-left:6px solid #3b82f6; }}
.card3 {{ border-left:6px solid #f97316; }}
.card4 {{ border-left:6px solid #a855f7; }}
body,div,span {{ color:{"white" if dark_mode else "black"}!important; }}
h1,h2,h3 {{ color:{"#ffffff" if dark_mode else "#1b5e20"}; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE + DASHBOARD HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title(t("title"))
st.markdown(f'<div class="glass"><h2>{t("dashboard")}</h2></div>', unsafe_allow_html=True)

# ─── Dashboard KPI cards (populated after prediction) ────────────────────────
d    = st.session_state.get("last_result", {})
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card card1"><h3>{d.get("level","--")}</h3><p>{t("status")}</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card card2"><h3>{d.get("confidence",0):.2f}</h3><p>{t("confidence")}</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card card3"><h3>{d.get("temp","--")}°C</h3><p>{t("temperature")}</p></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card card1"><h3>{d.get("humidity","--")}%</h3><p>{t("humidity")}</p></div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    try:
        model = tf.keras.models.load_model(
            "model.h5",
            compile=False
        )

        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)

        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None
@st.cache_data(show_spinner=False)
def load_classes() -> list:
    with open("class_names.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_knowledge_base() -> dict:
    try:
        with open("knowledge_base.json") as f:
            return json.load(f)
    except Exception:
        return {}


model        = load_model()
class_names  = load_classes()
knowledge_base = load_knowledge_base()

if model is None:
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER
# ═══════════════════════════════════════════════════════════════════════════════
CITY_COORDS = {
    "chennai":   (13.0827, 80.2707), "delhi":     (28.6139, 77.2090),
    "mumbai":    (19.0760, 72.8777), "kolkata":   (22.5726, 88.3639),
    "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
    "pune":      (18.5204, 73.8567), "ahmedabad": (23.0225, 72.5714),
    "jaipur":    (26.9124, 75.7873), "lucknow":   (26.8467, 80.9462),
    "coimbatore":(11.0168, 76.9558), "madurai":   ( 9.9252, 78.1198),
    "trichy":    (10.7905, 78.7047), "salem":     (11.6643, 78.1460),
    "vellore":   (12.9165, 79.1325),
}
WEATHER_CACHE_FILE  = "weather_cache.json"
CACHE_EXPIRY_HOURS  = 6


def _load_cache() -> dict:
    try:
        with open(WEATHER_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(WEATHER_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _is_fresh(entry: dict) -> bool:
    try:
        return datetime.now() - datetime.fromisoformat(entry["timestamp"]) \
               < timedelta(hours=CACHE_EXPIRY_HOURS)
    except Exception:
        return False


def _fetch_openmeteo(lat: float, lon: float):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m&forecast_days=1")
        data = requests.get(url, timeout=5).json()
        return float(data["current"]["temperature_2m"]), \
               float(data["current"]["relative_humidity_2m"])
    except Exception:
        return None, None


def get_weather(city: str) -> tuple:
    key   = city.strip().lower()
    cache = _load_cache()

    if key in cache and _is_fresh(cache[key]):
        e = cache[key]
        return e["humidity"], e["temp"], "cache"

    coords = CITY_COORDS.get(key)
    if coords is None:
        try:
            geo = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
                timeout=5).json()
            r       = geo["results"][0]
            coords  = (r["latitude"], r["longitude"])
        except Exception:
            coords = None

    if coords:
        temp, humidity = _fetch_openmeteo(*coords)
        if temp is not None:
            cache[key] = dict(temp=round(temp, 1), humidity=round(humidity, 1),
                              timestamp=datetime.now().isoformat(), source="live")
            _save_cache(cache)
            return round(humidity, 1), round(temp, 1), "live"

    if key in cache:
        e = cache[key]
        return e["humidity"], e["temp"], "stale"

    return 65.0, 28.0, "default"


def show_weather_ui(temp, humidity, source="live"):
    labels = {
        "live":    "🟢 Live weather",
        "cache":   "🟡 Cached weather (fresh)",
        "stale":   "🟠 Cached weather (last known)",
        "default": "🔴 Default estimate",
    }
    st.caption(labels.get(source, ""))
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"🌡 {t('temperature')}", f"{temp}°C")
    with col2:
        st.metric(f"💧 {t('humidity')}", f"{humidity}%")
    if humidity > 80:
        st.warning(t("high_humidity_warn"))
    if temp > 35:
        st.warning(t("high_temp_warn"))


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY
# ═══════════════════════════════════════════════════════════════════════════════
def scientific_rules(humidity, temp):
    notes = []
    if humidity > 80: notes.append(t("fungal_risk"))
    if temp > 35:     notes.append(t("heat_stress"))
    if temp < 15:     notes.append(t("slow_recovery"))
    return notes


def get_severity(conf, humidity, temp):
    if conf >= 0.7 or humidity > 80:
        level_key, color = "HIGH",   "card3"
        message          = t("immediate_action")
    elif conf >= 0.4 or humidity > 65:
        level_key, color = "MEDIUM", "card2"
        message          = t("monitor_closely")
    else:
        level_key, color = "LOW",    "card1"
        message          = t("safe_condition")
    notes = scientific_rules(humidity, temp)
    return level_key, color, message, notes


def show_severity_card(level, color, message):
    st.markdown(f"""
    <div class="card {color}">
    <h2>⚠️ {level}</h2>
    <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


def show_risk_alert(level, confidence):
    if level == "HIGH":
        st.error(f"{t('high_risk_msg')}: {confidence:.2f}")
    elif level == "MEDIUM":
        st.warning(f"{t('moderate_risk_msg')}: {confidence:.2f}")
    else:
        st.success(f"{t('low_risk_msg')}: {confidence:.2f}")


def spray_schedule(humidity, level):
    if level == "HIGH":     return 5
    if humidity > 80:       return 7
    if humidity > 60:       return 10
    return 14


# ═══════════════════════════════════════════════════════════════════════════════
# ADVICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def get_advice(label, lang, humidity, temp, confidence):
    info = knowledge_base.get(label)

    if not info:
        base = (f"{t('general_advice')}:\n"
                f"- {t('monitor')}\n- {t('maintain_irr')}\n"
                f"- {t('remove_infected')}\n- {t('use_pesticide')}\n\n"
                f"{t('detected')}: {label}\n")
    elif lang == "English":
        base = (f"Symptoms: {info.get('Symptoms','N/A')}\n"
                f"Causes: {info.get('Causes','N/A')}\n"
                f"Prevention: {info.get('Prevention','N/A')}\n"
                f"Cure: {info.get('Cure','N/A')}\n"
                f"Impact: {info.get('Impact','N/A')}\n"
                f"Best Practices: {info.get('Best Practices','N/A')}\n")
    elif lang == "Tamil":
        d = knowledge_base_ml["Tamil"]["_default"]
        base = (f"அறிகுறிகள்: {info.get('Symptoms', d['Symptoms'])}\n"
                f"காரணங்கள்: {info.get('Causes', d['Causes'])}\n"
                f"தடுப்பு: {info.get('Prevention', d['Prevention'])}\n"
                f"சிகிச்சை: {info.get('Cure', d['Cure'])}\n"
                f"தாக்கம்: {info.get('Impact', d['Impact'])}\n"
                f"சிறந்த நடைமுறைகள்: {info.get('Best Practices', d['Best Practices'])}\n")
    elif lang == "Hindi":
        d = knowledge_base_ml["Hindi"]["_default"]
        base = (f"लक्षण: {info.get('Symptoms', d['Symptoms'])}\n"
                f"कारण: {info.get('Causes', d['Causes'])}\n"
                f"रोकथाम: {info.get('Prevention', d['Prevention'])}\n"
                f"उपचार: {info.get('Cure', d['Cure'])}\n"
                f"प्रभाव: {info.get('Impact', d['Impact'])}\n"
                f"सर्वोत्तम अभ्यास: {info.get('Best Practices', d['Best Practices'])}\n")
    else:
        base = f"Symptoms: {info.get('Symptoms','N/A')}\n"

    weather_note = ""
    if humidity > 80: weather_note += f"\n{t('humidity_fungal')}"
    if temp > 35:     weather_note += f"\n{t('heat_risk')}"

    if confidence > 0.8:       conf_note = f"\n{t('high_conf_note')}"
    elif confidence > 0.5:     conf_note = f"\n{t('moderate_conf_note')}"
    else:                      conf_note = f"\n{t('low_conf_note')}"

    return base + weather_note + conf_note


# ═══════════════════════════════════════════════════════════════════════════════
# CROP DATABASE + FARM CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
CROP_DB = {
    "Tomato":       dict(N=100,P=50, K=50, water=3000,pH="6.0–6.8",soil="Sandy loam",      temp="20–27°C",season="60–80 days",  yield_="25–35 t/acre",spacing="45×60 cm",  depth="0.5–1.5 m",method="Drip / Furrow",     spray=7),
    "Potato":       dict(N=180,P=80, K=100,water=3200,pH="5.5–6.5",soil="Sandy loam",       temp="15–20°C",season="70–120 days", yield_="15–25 t/acre",spacing="30×60 cm",  depth="0.5–1.0 m",method="Furrow / Sprinkler", spray=7),
    "Brinjal":      dict(N=100,P=50, K=50, water=2800,pH="5.5–6.8",soil="Loam",             temp="22–32°C",season="80–100 days", yield_="15–25 t/acre",spacing="60×75 cm",  depth="0.5–1.2 m",method="Drip / Furrow",     spray=10),
    "Cabbage":      dict(N=120,P=60, K=60, water=2500,pH="6.0–7.0",soil="Clay loam",        temp="15–20°C",season="60–90 days",  yield_="20–30 t/acre",spacing="45×60 cm",  depth="0.4–0.8 m",method="Sprinkler",         spray=10),
    "Cauliflower":  dict(N=120,P=60, K=60, water=2400,pH="6.0–7.0",soil="Loam",             temp="15–20°C",season="65–85 days",  yield_="10–20 t/acre",spacing="45×60 cm",  depth="0.4–0.8 m",method="Sprinkler",         spray=10),
    "Onion":        dict(N=100,P=50, K=80, water=3000,pH="6.0–7.5",soil="Sandy loam",       temp="13–24°C",season="120–150 days",yield_="10–20 t/acre",spacing="10×15 cm",  depth="0.3–0.5 m",method="Drip / Furrow",     spray=10),
    "Garlic":       dict(N=80, P=50, K=60, water=2200,pH="6.0–7.0",soil="Sandy loam",       temp="12–24°C",season="120–150 days",yield_="6–10 t/acre", spacing="10×15 cm",  depth="0.3–0.5 m",method="Drip / Furrow",     spray=14),
    "Carrot":       dict(N=60, P=80, K=80, water=2000,pH="6.0–6.8",soil="Deep loam",        temp="15–20°C",season="70–90 days",  yield_="8–15 t/acre", spacing="5×30 cm",   depth="0.5–1.0 m",method="Sprinkler / Drip",  spray=14),
    "Spinach":      dict(N=80, P=40, K=60, water=1800,pH="6.0–7.0",soil="Loam",             temp="10–18°C",season="40–50 days",  yield_="5–10 t/acre", spacing="20×30 cm",  depth="0.3–0.5 m",method="Sprinkler",         spray=14),
    "Cucumber":     dict(N=100,P=50, K=80, water=3000,pH="6.0–7.0",soil="Sandy loam",       temp="18–30°C",season="50–70 days",  yield_="10–20 t/acre",spacing="45×90 cm",  depth="0.5–1.0 m",method="Drip / Furrow",     spray=7),
    "Rice":         dict(N=120,P=60, K=40, water=5000,pH="5.5–6.5",soil="Clay / Clay loam", temp="20–35°C",season="110–150 days",yield_="3–6 t/acre",  spacing="20×15 cm",  depth="0.3–0.6 m",method="Flood / Furrow",    spray=10),
    "Wheat":        dict(N=100,P=50, K=40, water=3000,pH="6.0–7.0",soil="Loam",             temp="15–22°C",season="100–130 days",yield_="2–4 t/acre",  spacing="20 cm rows",depth="1.0–1.5 m",method="Furrow / Sprinkler", spray=14),
    "Maize":        dict(N=150,P=70, K=50, water=3500,pH="5.8–7.0",soil="Sandy loam",       temp="18–32°C",season="80–110 days", yield_="3–6 t/acre",  spacing="25×75 cm",  depth="1.0–1.5 m",method="Furrow / Sprinkler", spray=14),
    "Corn":         dict(N=150,P=70, K=50, water=3500,pH="5.8–7.0",soil="Sandy loam",       temp="18–32°C",season="80–110 days", yield_="3–6 t/acre",  spacing="25×75 cm",  depth="1.0–1.5 m",method="Furrow / Sprinkler", spray=14),
    "Soybean":      dict(N=20, P=60, K=40, water=2200,pH="6.0–7.0",soil="Loam",             temp="20–30°C",season="90–120 days", yield_="1–2 t/acre",  spacing="5×45 cm",   depth="0.8–1.2 m",method="Furrow / Sprinkler", spray=14),
    "Banana":       dict(N=200,P=60, K=300,water=6000,pH="6.0–7.5",soil="Clay loam",        temp="26–35°C",season="9–12 months", yield_="25–40 t/acre",spacing="180×180 cm",depth="0.5–1.5 m",method="Drip / Flood",      spray=10),
    "Mango":        dict(N=100,P=50, K=100,water=3000,pH="5.5–7.5",soil="Sandy loam",       temp="24–35°C",season="3–5 months",  yield_="5–15 t/acre", spacing="10×10 m",   depth="1.5–3.0 m",method="Drip / Furrow",     spray=14),
    "Cotton":       dict(N=120,P=60, K=60, water=4000,pH="6.0–7.5",soil="Black soil",       temp="21–30°C",season="150–180 days",yield_="0.5–1 t/acre",spacing="60×90 cm",  depth="1.0–2.0 m",method="Furrow / Drip",     spray=10),
    "Sugarcane":    dict(N=200,P=80, K=120,water=7000,pH="6.0–8.0",soil="Clay loam",        temp="25–35°C",season="10–14 months",yield_="40–60 t/acre",spacing="90 cm rows",depth="1.0–2.0 m",method="Furrow / Drip",     spray=14),
    "Grapes":       dict(N=100,P=50, K=80, water=2500,pH="6.0–7.0",soil="Sandy loam",       temp="15–30°C",season="6–8 months",  yield_="8–15 t/acre", spacing="3×2 m",     depth="0.8–1.5 m",method="Drip",              spray=7),
    "Apple":        dict(N=70, P=40, K=60, water=2800,pH="5.5–6.5",soil="Sandy loam",       temp="15–25°C",season="5–7 months",  yield_="10–20 t/acre",spacing="6×6 m",     depth="1.0–2.0 m",method="Drip / Sprinkler",  spray=10),
    "Strawberry":   dict(N=100,P=50, K=80, water=2700,pH="5.5–6.5",soil="Sandy loam",       temp="15–22°C",season="90–120 days", yield_="3–8 t/acre",  spacing="30×60 cm",  depth="0.3–0.5 m",method="Drip",              spray=7),
    "Chilli":       dict(N=120,P=60, K=60, water=2800,pH="6.0–7.0",soil="Loam",             temp="20–30°C",season="90–120 days", yield_="3–6 t/acre",  spacing="45×60 cm",  depth="0.5–0.8 m",method="Drip / Furrow",     spray=7),
    "Turmeric":     dict(N=80, P=40, K=80, water=3000,pH="5.5–7.0",soil="Sandy loam",       temp="20–30°C",season="8–9 months",  yield_="2–4 t/acre",  spacing="25×45 cm",  depth="0.3–0.6 m",method="Furrow / Drip",     spray=10),
    "Sunflower":    dict(N=80, P=40, K=40, water=2500,pH="6.0–7.5",soil="Loam",             temp="20–30°C",season="80–100 days", yield_="0.8–1.5 t/acre",spacing="30×60 cm",depth="0.8–1.5 m",method="Furrow / Sprinkler", spray=14),
}


def farm_calculator(area, humidity, temp, crop):
    area     = float(area)
    crop_key = next((k for k in CROP_DB if k.lower() == crop.strip().lower()), None)
    info     = CROP_DB.get(crop_key) if crop_key else None

    if info:
        N, P, K      = info["N"], info["P"], info["K"]
        irrigation   = info["water"]
        spray_base   = info["spray"]
        ph           = info["pH"]
        soil         = info["soil"]
        opt_temp     = info["temp"]
        season       = info["season"]
        exp_yield    = info["yield_"]
        spacing      = info["spacing"]
        depth        = info["depth"]
        method       = info["method"]
        display_crop = crop_key
    else:
        N = P = K = 60; irrigation = 3000; spray_base = 10
        ph = soil = opt_temp = season = exp_yield = spacing = depth = method = "N/A"
        display_crop = crop

    if temp > 32:           irrigation += 800
    if humidity < 40:       irrigation += 500
    if humidity > 35:       irrigation = max(irrigation - 200, 1500)
    irrigation = max(1500, irrigation)
    spray = 7 if humidity > 80 else (spray_base if humidity > 60 else min(spray_base + 4, 14))

    return f"""
{t('smart_farm_plan')}

{t('crop_label')}: {display_crop}
{t('area_label')}: {area} {t('acres')}

── {t('fertilizer_label')} ──
N : {int(N * area)} {t('kg')}
P : {int(P * area)} {t('kg')}
K : {int(K * area)} {t('kg')}

── {t('irrigation_label')} ──
{int(irrigation * area)} {t('liters')}  ({method})

── {t('spray_label')} ──
{spray} {t('days')}

── Agronomic Details ──
Soil pH       : {ph}
Soil Type     : {soil}
Optimal Temp  : {opt_temp}
Season        : {season}
Expected Yield: {exp_yield}
Spacing       : {spacing}
Root Depth    : {depth}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
if "session_conf" not in st.session_state:
    st.session_state.session_conf  = []
    st.session_state.session_time  = []
    st.session_state.session_labels = []


def update_session(conf, label=""):
    sess = st.session_state
    if len(sess.session_conf) >= 20:
        sess.session_conf.pop(0)
        sess.session_time.pop(0)
        sess.session_labels.pop(0)
    sess.session_conf.append(conf)
    sess.session_time.append(time.strftime("%H:%M:%S"))
    sess.session_labels.append(label)


def plot_trend():
    if not st.session_state.session_conf:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.session_conf,
        x=st.session_state.session_time,
        mode="lines+markers",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=8),
    ))
    fig.update_layout(title=t("trend"), template="plotly_white",
                      xaxis_title="Time", yaxis_title="Confidence")
    return fig


def plot_top5(top_preds: list):
    labels = [p["label"]       for p in top_preds]
    values = [p["probability"] for p in top_preds]
    colors = ["#22c55e" if i == 0 else "#3b82f6" for i in range(len(labels))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        title=t("top_pred"),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        template="plotly_white",
    )
    return fig


def show_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": t("confidence_meter")},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#22c55e"},
            "steps": [
                {"range": [0,  40], "color": "#bbf7d0"},
                {"range": [40, 70], "color": "#fef08a"},
                {"range": [70,100], "color": "#fca5a5"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.header(t("controls"))
city        = st.sidebar.text_input(t("city"), "Chennai")
area        = st.sidebar.number_input(t("area"), value=1.0, min_value=0.1, step=0.5)
crop        = st.sidebar.text_input(t("enter_crop"), "Rice")
farmer_mode = st.sidebar.checkbox(t("farmer_mode"), True)

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE INPUT
# ═══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(t("upload"), type=["jpg", "jpeg", "png"])
camera_image  = st.camera_input(t("camera"))

if camera_image is not None:
    uploaded_file = camera_image

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION + FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.85   # Phase 6: reject below this


def run_full_pipeline(image_file, city, area, lang, crop):
    """
    Complete AgroVision Pro pipeline:
    1. Image validation
    2. Leaf coverage check + background removal
    3. Advanced preprocessing
    4. Inference + confidence gate
    5. Top-5 predictions
    6. Severity estimation
    7. Grad-CAM
    8. Advisory + farm plan
    """
    # ── 1. Load PIL ───────────────────────────────────────────────────────────
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    pil_original = Image.open(image_file).convert("RGB")

    # ── 2. Image validation ────────────────────────────────────────────────────
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    val = validate_image(pil_original)
    if not val["valid"]:
        return dict(error=True, error_type="image_quality",
                    errors=val["errors"], quality=val["quality"],
                    resolution=val["resolution"],
                    blur_score=val["blur_score"],
                    brightness=val["brightness"],
                    contrast=val["contrast"],
                    warnings=val["warnings"])

    # ── 3. Leaf coverage + background removal ──────────────────────────────────
    seg = segment_leaf(pil_original, use_grabcut=True)
    if not seg["ok"]:
        return dict(error=True, error_type="leaf_coverage",
                    errors=[seg["message"]],
                    quality=val["quality"],
                    coverage_pct=seg["coverage_pct"])

    cleaned_image = seg["cleaned_image"]
    leaf_mask     = seg["mask"]

    # ── 4. Advanced preprocessing ──────────────────────────────────────────────
    img_array = preprocess_image(cleaned_image, target_size=(224, 224))

    # ── 5. Inference ───────────────────────────────────────────────────────────
    raw_output = model.predict(img_array, verbose=0)[0]
    idx        = int(np.argmax(raw_output))
    confidence = float(raw_output[idx])
    safe_names = class_names[:len(raw_output)]
    label      = safe_names[idx] if idx < len(safe_names) else f"Class_{idx}"

    # ── 6. Confidence gate ─────────────────────────────────────────────────────
    uncertain = confidence < CONFIDENCE_THRESHOLD

    # ── 7. Top-5 predictions ───────────────────────────────────────────────────
    top5 = top_k_predictions(raw_output, safe_names, k=5)

    # ── 8. Weather ────────────────────────────────────────────────────────────
    humidity, temp, weather_source = get_weather(city)

    # ── 9. Severity ───────────────────────────────────────────────────────────
    level, color, message, notes = get_severity(confidence, humidity, temp)
    sev_info = estimate_severity_from_mask(leaf_mask, confidence)
    spray    = spray_schedule(humidity, level)

    # ── 10. Advisory + farm plan ──────────────────────────────────────────────
    advice    = get_advice(label, lang, humidity, temp, confidence)
    farm_info = farm_calculator(area, humidity, temp, crop)

    # ── 11. Grad-CAM ──────────────────────────────────────────────────────────
    gc = generate_gradcam(cleaned_image, model, pred_index=idx)

    # ── 12. Analytics ─────────────────────────────────────────────────────────
    update_session(confidence, label)
    fig_top5  = plot_top5(top5)
    fig_trend = plot_trend()

    return dict(
        error          = False,
        uncertain      = uncertain,
        label          = label,
        confidence     = confidence,
        level          = level,
        color          = color,
        message        = message,
        notes          = notes,
        spray          = spray,
        humidity       = humidity,
        temp           = temp,
        weather_source = weather_source,
        advice         = advice,
        farm           = farm_info,
        top5           = top5,
        fig_top5       = fig_top5,
        fig_trend      = fig_trend,
        # image quality
        quality        = val["quality"],
        resolution     = val["resolution"],
        blur_score     = val["blur_score"],
        brightness     = val["brightness"],
        contrast       = val["contrast"],
        quality_warnings = val["warnings"],
        # leaf
        coverage_pct   = seg["coverage_pct"],
        cleaned_image  = cleaned_image,
        # severity extended
        sev_infected   = sev_info["infected_pct"],
        sev_desc       = sev_info["description"],
        sev_color      = sev_info["color"],
        # grad-cam
        gradcam_overlay = gc["overlay_image"],
        gradcam_heatmap = gc["heatmap_image"],
        hotspot_pct     = gc["hotspot_pct"],
        gradcam_error   = gc["error"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI — TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    st.image(uploaded_file, caption=t("upload"), use_container_width=True)

    if st.button(t("analyze"), type="primary"):
        with st.spinner(t("processing")):
            result = run_full_pipeline(uploaded_file, city, area, language, crop)

        # ── Errors ────────────────────────────────────────────────────────────
        if result.get("error"):
            st.error(f"🚫 {t('image_rejected')}")
            for err in result.get("errors", []):
                st.error(err)
            if result.get("quality") is not None:
                st.info(f"{t('image_quality_score')}: {result['quality']}%")
            st.stop()

        # ── Store for dashboard ────────────────────────────────────────────────
        st.session_state.last_result = result

        # ── Uncertain prediction warning ──────────────────────────────────────
        if result.get("uncertain"):
            st.warning(t("low_confidence_warning"))

        # ═══════════════════════════════════════════════════════════════════════
        # TOP RESULT CARDS
        # ═══════════════════════════════════════════════════════════════════════
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="card card1"><h3>{result["label"]}</h3><p>{t("disease")}</p></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card card2"><h3>{result["confidence"]:.2f}</h3><p>{t("confidence")}</p></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card card3"><h3>{result["level"]}</h3><p>{t("severity")}</p></div>',
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="card card4"><h3>{result["quality"]}%</h3><p>{t("image_quality_score")}</p></div>',
                        unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="card card1"><h3>{result["coverage_pct"]}%</h3><p>{t("leaf_coverage_pct")}</p></div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence bar + gauge ─────────────────────────────────────────────
        st.subheader(t("confidence_level"))
        st.progress(min(max(result["confidence"], 0.0), 1.0))
        show_gauge(result["confidence"])

        # ── Weather + risk ─────────────────────────────────────────────────────
        show_weather_ui(result["temp"], result["humidity"], result["weather_source"])
        show_risk_alert(result["level"], result["confidence"])

        # ═══════════════════════════════════════════════════════════════════════
        # TABS
        # ═══════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            t("prediction"),
            t("advice"),
            t("analytics"),
            t("farm_tools"),
            t("gradcam_tab"),
        ])

        # ─── Tab 1: Prediction ─────────────────────────────────────────────────
        with tab1:
            st.markdown('<div class="glass">', unsafe_allow_html=True)

            # Image quality card
            st.subheader(t("image_quality"))
            qcol1, qcol2, qcol3 = st.columns(3)
            with qcol1: st.metric("Blur Score",   result["blur_score"])
            with qcol2: st.metric("Brightness",   result["brightness"])
            with qcol3: st.metric("Contrast",     result["contrast"])
            for w in result.get("quality_warnings", []):
                st.warning(w)

            st.divider()

            # Severity card
            show_severity_card(result["level"], result["color"], result["message"])
            st.markdown(f"**Severity Detail:** {result['sev_desc']}")
            st.markdown(f"**Estimated infected area:** {result['sev_infected']}%")

            for note in result["notes"]:
                st.info(note)

            st.write(f"{t('spray_interval')}: **{result['spray']} {t('days')}**")

            # Top-5 bar chart
            st.plotly_chart(result["fig_top5"], use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ─── Tab 2: Advice ─────────────────────────────────────────────────────
        with tab2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            with st.expander(t("detailed_advice"), expanded=True):
                st.markdown(result["advice"])
            if farmer_mode:
                st.success(t("farmer_enabled"))
                st.info(t("spray_tip"))
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── Tab 3: Analytics ──────────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            if result["fig_trend"]:
                st.plotly_chart(result["fig_trend"], use_container_width=True)
            st.subheader(t("history"))
            if st.session_state.session_conf:
                st.line_chart(st.session_state.session_conf)
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── Tab 4: Farm Tools ─────────────────────────────────────────────────
        with tab4:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(result["farm"])
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── Tab 5: Grad-CAM ───────────────────────────────────────────────────
        with tab5:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader(t("hotspot_analysis"))

            if result.get("gradcam_error"):
                st.warning(f"Grad-CAM unavailable: {result['gradcam_error']}")
                st.image(result["cleaned_image"], caption="Processed Leaf",
                         use_container_width=True)
            else:
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.image(result["gradcam_overlay"],
                             caption="🔥 Disease Hotspot Overlay",
                             use_container_width=True)
                with gcol2:
                    st.image(result["gradcam_heatmap"],
                             caption="🌡 Activation Heatmap",
                             use_container_width=True)
                st.metric(t("hotspot_pct"), f"{result['hotspot_pct']}%")

            st.markdown('</div>', unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════════
        # EXPORT REPORTS (Phase 16)
        # ═══════════════════════════════════════════════════════════════════════
        st.subheader(t("exports"))
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.download_button(
                label    = t("export_txt"),
                data     = generate_txt_report(result),
                file_name= "agrovision_report.txt",
                mime     = "text/plain",
            )
        with ecol2:
            st.download_button(
                label    = t("export_csv"),
                data     = generate_csv_report(result),
                file_name= "agrovision_report.csv",
                mime     = "text/csv",
            )
        with ecol3:
            st.download_button(
                label    = t("export_json"),
                data     = generate_json_report(result),
                file_name= "agrovision_report.json",
                mime     = "application/json",
            )

else:
    st.warning(t("upload_warning"))
