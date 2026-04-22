import streamlit as st
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# Dummy class names
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Healthy Leaf"
]

# Treatment
treatments = {
    "Tomato Early Blight": "Apply fungicide spray.",
    "Tomato Late Blight": "Use copper fungicide.",
    "Healthy Leaf": "No treatment needed."
}

st.title("🌿 AI Plant Disease Detection")
st.write("Upload or capture a leaf image to detect plant diseases.")

camera_photo = st.camera_input("📷 Take a photo")
uploaded_file = st.file_uploader("OR Upload a leaf image", type=["jpg", "jpeg", "png"])

image = None

if camera_photo:
    image = Image.open(camera_photo)
elif uploaded_file:
    image = Image.open(uploaded_file)

if image is not None:
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    with st.spinner("Analyzing leaf image..."):
        time.sleep(1)

        # Dummy prediction
        predicted_class = np.random.choice(class_names)
        confidence = np.random.uniform(0.85, 0.99)

    st.success(f"🌱 Disease Detected: {predicted_class}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")
    st.info(f"💊 Recommendation: {treatments[predicted_class]}")