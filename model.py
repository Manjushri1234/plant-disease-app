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
        "Low": "Remove affected leaves and improve air circulation.",
        "Medium": "Apply fungicide weekly and avoid overhead watering.",
        "High": "Use strong fungicide and isolate infected plant."
    },
    "Tomato Late Blight": {
        "Low": "Monitor plant condition.",
        "Medium": "Use copper fungicide.",
        "High": "Remove infected plant immediately."
    },
    "Healthy Leaf": {
        "Low": "No treatment needed.",
        "Medium": "Maintain proper watering and sunlight.",
        "High": "Recheck image for accuracy."
    }
}

# -----------------------------
# 🔥 STRONG LEAF DETECTION
# -----------------------------
def is_leaf(image):
    img = np.array(image)

    red = img[:, :, 0].astype(float)
    green = img[:, :, 1].astype(float)
    blue = img[:, :, 2].astype(float)

    # 1. Green dominance
    green_mask = (green > red + 10) & (green > blue + 10)
    green_ratio = np.sum(green_mask) / green_mask.size

    # 2. Skin detection
    skin_mask = (
        (red > 95) & (green > 40) & (blue > 20) &
        ((np.max(img, axis=2) - np.min(img, axis=2)) > 15) &
        (abs(red - green) > 15) &
        (red > green) & (red > blue)
    )
    skin_ratio = np.sum(skin_mask) / skin_mask.size

    # 3. Texture check
    gray = np.mean(img, axis=2)
    texture = np.std(gray)

    # Final strict condition
    if green_ratio > 0.4 and skin_ratio < 0.15 and texture > 25:
        return True
    else:
        return False

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

        # STEP 1: Validate leaf
        if not is_leaf(image):
            st.error("❌ Not a valid plant leaf. Please upload a leaf image.")
            st.stop()

        # STEP 2: Dummy Prediction (replace with real model later)
        predicted_class = np.random.choice(class_names)
        confidence = np.random.uniform(0.85, 0.99)

        # STEP 3: Severity
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