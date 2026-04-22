import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load class names from dataset folders
class_names = sorted(os.listdir("PlantVillage"))

# -----------------------------
# Treatment recommendation
# -----------------------------
treatments = {
    "Pepper__bell___Bacterial_spot": "Remove infected leaves and apply copper fungicide.",
    "Pepper__bell___healthy": "Plant is healthy. No treatment needed.",
    "Potato___Early_blight": "Apply Mancozeb fungicide and remove infected leaves.",
    "Potato___Late_blight": "Use metalaxyl-based fungicide.",
    "Potato___healthy": "Plant is healthy.",
    "Tomato___Early_blight": "Apply chlorothalonil fungicide.",
    "Tomato___Late_blight": "Apply copper fungicide.",
    "Tomato___Leaf_Mold": "Improve air circulation and apply fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and spray fungicide.",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Use neem oil or miticide.",
    "Tomato___Target_Spot": "Apply fungicide spray.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato___Tomato_YellowLeaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato___healthy": "Plant is healthy."
}

# -----------------------------
# UI
# -----------------------------
st.title("🌿 AI Plant Disease Detection")
st.write("Upload or capture a leaf image to detect plant diseases using AI.")

# Camera input
camera_photo = st.camera_input("📷 Take a photo")

# File upload
uploaded_file = st.file_uploader("OR Upload a leaf image", type=["jpg", "jpeg", "png"])

image = None

# Check camera
if camera_photo is not None:
    image = Image.open(camera_photo)

# Check file upload
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# -----------------------------
# Prediction
# -----------------------------
if image is not None:

    image = image.convert("RGB")

    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    with st.spinner("Analyzing leaf image..."):
        time.sleep(1)

        # Preprocess image
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)

        predicted_class = class_names[class_idx]

    # Clean class name
    clean_name = predicted_class.replace("_", " ")

    st.success(f"🌱 Disease Detected: {clean_name}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")

    if predicted_class in treatments:
        st.info(f"💊 Recommendation: {treatments[predicted_class]}")