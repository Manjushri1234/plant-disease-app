import streamlit as st
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Healthy Leaf"
]

# -----------------------------
# Recommendations
# -----------------------------
recommendations = {
    "Tomato Early Blight": {
        "Low": "Apply fungicide spray.",
        "Medium": "Apply fungicide regularly and remove affected leaves.",
        "High": "Use strong fungicide and isolate plant."
    },
    "Tomato Late Blight": {
        "Low": "Monitor plant condition.",
        "Medium": "Use copper fungicide.",
        "High": "Remove infected plant immediately."
    },
    "Healthy Leaf": {
        "Low": "No treatment needed.",
        "Medium": "Maintain plant properly.",
        "High": "Recheck image."
    }
}

# -----------------------------
# Simple Leaf Check
# -----------------------------
def is_leaf(image):
    img = np.array(image)

    red = img[:, :, 0].astype(float)
    green = img[:, :, 1].astype(float)
    blue = img[:, :, 2].astype(float)

    green_mask = (green > red) & (green > blue)
    green_ratio = np.sum(green_mask) / green_mask.size

    return green_ratio > 0.15


# -----------------------------
# Simple Human Rejection (Color Heuristic)
# -----------------------------
def looks_like_human(image):
    img = np.array(image)

    red = img[:, :, 0].astype(float)
    green = img[:, :, 1].astype(float)
    blue = img[:, :, 2].astype(float)

    # skin-like color detection (basic)
    skin_mask = (red > 95) & (green > 40) & (blue > 20) & \
                (red > green) & (red > blue)

    skin_ratio = np.sum(skin_mask) / skin_mask.size

    return skin_ratio > 0.2


# -----------------------------
# UI
# -----------------------------
st.title("🌿 AI Plant Disease Detection")
st.write("Upload or capture a plant leaf image to detect diseases.")

camera_photo = st.camera_input("📷 Take a photo")
uploaded_file = st.file_uploader("OR Upload a leaf image", type=["jpg", "jpeg", "png"])

image = None

if camera_photo:
    image = Image.open(camera_photo)
elif uploaded_file:
    image = Image.open(uploaded_file)

# -----------------------------
# PROCESS
# -----------------------------
if image is not None:
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        time.sleep(1)

        # STEP 1: Reject human-like images
        if looks_like_human(image):
            st.error("❌ Human detected. Please upload a plant leaf image.")
            st.stop()

        # STEP 2: Leaf check
        if not is_leaf(image):
            st.error("❌ Not a plant leaf. Please upload a leaf image.")
            st.stop()

        # STEP 3: Dummy prediction (replace later with real model)
        predicted_class = np.random.choice(class_names)
        confidence = np.random.uniform(0.85, 0.99)

        # STEP 4: Severity
        if confidence > 0.95:
            severity = "Low"
        elif confidence > 0.90:
            severity = "Medium"
        else:
            severity = "High"

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.success(f"🌱 Disease Detected: {predicted_class}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")
    st.warning(f"🌡️ Severity: {severity}")
    st.info(f"💊 Recommendation: {recommendations[predicted_class][severity]}")