import streamlit as st
import numpy as np
from PIL import Image
import json
import random

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🌱 AgroVision - Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Fake prediction (for deployment demo)
    predicted_class = random.choice(class_names)
    confidence = random.uniform(85, 99)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
