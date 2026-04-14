import streamlit as st
import numpy as np
from PIL import Image
import json
from tflite_runtime.interpreter import Interpreter

# -------------------------
# Load model
# -------------------------
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# Load class names
# -------------------------
with open("class_names.json") as f:
    class_names = json.load(f)

# -------------------------
# STATIC KNOWLEDGE BASE
# -------------------------
knowledge_base = {
    "Tomato___Late_blight": {
        "advice": "Remove infected leaves and apply fungicide.",
        "actions": ["Remove infected parts", "Avoid overwatering", "Use fungicide"]
    },
    "Potato___Early_blight": {
        "advice": "Ensure proper spacing and apply treatment.",
        "actions": ["Improve airflow", "Use resistant seeds", "Apply fungicide"]
    }
}

# -------------------------
# Severity Logic (STATIC)
# -------------------------
def get_severity(conf):
    if conf > 0.7:
        return "🔴 High Risk"
    elif conf > 0.4:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

# -------------------------
# Prediction Function
# -------------------------
def predict(image):
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output)
    confidence = float(np.max(output))

    return class_names[pred_index], confidence, output

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🌱 AgroVision - Plant Disease Detector")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    label, conf, output = predict(image)

    # Prediction
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {conf*100:.2f}%")

    # Severity
    severity = get_severity(conf)
    st.warning(f"Severity: {severity}")

    # Advice
    if label in knowledge_base:
        st.subheader("👨‍🌾 Farmer Advice")
        st.write(knowledge_base[label]["advice"])

        st.subheader("✅ Quick Actions")
        for act in knowledge_base[label]["actions"]:
            st.write("•", act)
    else:
        st.write("No specific advice available")

    # Top 3 Predictions
    st.subheader("📊 Top Predictions")
    top3 = np.argsort(output[0])[-3:][::-1]

    for i in top3:
        st.write(f"{class_names[i]} → {output[0][i]*100:.2f}%")
