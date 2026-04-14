import streamlit as st
import numpy as np
from PIL import Image
import json
from keras.models import load_model

# Load model
model = load_model("agrovision_model_final.h5")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Title
st.title("🌱 AgroVision - Plant Disease Detector")

st.write("Upload a plant leaf image to detect disease")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
