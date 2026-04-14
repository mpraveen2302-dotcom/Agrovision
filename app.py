import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# Load model
@st.cache_resource
def load_my_model():
    return load_model("best_model.h5")

model = load_my_model()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🌱 AgroVision - Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
