import streamlit as st
import numpy as np
from PIL import Image
import json
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Title
st.title("🌱 AgroVision - Plant Disease Detector")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Preprocess
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_class = class_names[np.argmax(output)]
    confidence = np.max(output) * 100

    st.success(f"Prediction: {pred_class}")
    st.info(f"Confidence: {confidence:.2f}%")
