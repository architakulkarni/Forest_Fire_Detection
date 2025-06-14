import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(page_title="Fire Detection from Satellite", layout="centered")

st.title("🔥 Fire Detection from Satellite Images")
st.write("Upload a satellite image to detect whether there is **Fire** or **No Fire**.")

# Load model (cached)
@st.cache_resource
def load_fire_model():
    model = load_model(r"fire_detection_model.h5")
    return model

model = load_fire_model()

# Class labels (update if needed)
class_names = ["No Fire", "Fire 🔥"]

# Upload section
uploaded_file = st.file_uploader("📷 Upload a Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_shape = (64, 64)  # Update if your model uses different dimensions
    image_resized = image.resize(input_shape)
    img_array = np.array(image_resized) / 255.0
    img_array = img_array.reshape(1, *input_shape, 3)

    if st.button("🧠 Detect Fire"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[0][predicted_class]) * 100

            # Display results
            st.subheader("🔎 Detection Result")
            st.success(f"Prediction: *{class_names[predicted_class]}*")

            if confidence < 0.01:
                st.info("Confidence: *< 0.01%*")
            else:
                st.info(f"Confidence: *{confidence:.2f}%*")
