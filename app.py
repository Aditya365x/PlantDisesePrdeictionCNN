import os
import json

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ⬇️ your file is named llm_rag.py
from llm_rag import get_preventive_measures


# ==========================================================
# 1. PATHS & MODEL LOADING
# ==========================================================
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

# ⬇️ model is in the same folder as app.py (see your screenshot)
MODEL_PATH = os.path.join(WORKING_DIR, "plant_disease_prediction_model.h5")

# ⬇️ class_indices.json is also in the same folder
CLASS_INDICES_PATH = os.path.join(WORKING_DIR, "class_indices.json")

# Load CNN model once
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load class indices once
@st.cache_resource
def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return class_indices


# ==========================================================
# 2. IMAGE PREPROCESSING & PREDICTION
# ==========================================================
def load_and_preprocess_image(file, target_size=(224, 224)):
    """
    Takes an uploaded file (BytesIO) and returns a preprocessed numpy array
    ready for model.predict().
    """
    img = Image.open(file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, 3)
    return img_array


def predict_image_class(model, file, class_indices):
    """
    Runs the CNN model on the uploaded image and returns the class name.
    """
    img_array = load_and_preprocess_image(file)
    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    # class_indices keys are expected to be strings like "0", "1", ...
    pred_class_name = class_indices[str(pred_idx)]
    return pred_class_name


# ==========================================================
# 3. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(page_title="Plant Disease Cure Assistant", page_icon="🌱", layout="centered")

    st.title("🌱 Plant Disease Detection & Cure Assistant")
    st.write(
        "Upload a plant leaf image. The app will:\n"
        "1. Predict the disease using your CNN model.\n"
        "2. Use RAG + OpenAI LLM over your disease PDFs to suggest preventive measures & cure."
    )

    # Load model & class mapping
    model = load_cnn_model()
    class_indices = load_class_indices()

    uploaded_image = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Show preview
        st.subheader("Uploaded Image")
        st.image(uploaded_image, caption="Input Image", use_column_width=True)

        st.markdown("---")

        # Button to run full pipeline: predict + cure
        if st.button("🔮 Predict Disease & Suggest Cure"):
            # Step 1: CNN Prediction
            with st.spinner("Running CNN model to detect disease..."):
                try:
                    predicted_disease = predict_image_class(model, uploaded_image, class_indices)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    return

            st.success(f"🧬 Predicted Disease: **{predicted_disease}**")

            # Step 2: RAG LLM for preventive measures & cure
            with st.spinner("Querying knowledge base & generating preventive measures..."):
                try:
                    preventive_text = get_preventive_measures(predicted_disease)
                except Exception as e:
                    st.error(f"Error while generating cure suggestions from LLM: {e}")
                    return

            st.markdown("### 🩺 Suggested Preventive Measures & Cure")
            st.write(preventive_text)

    else:
        st.info("Please upload an image to begin.")


if __name__ == "__main__":
    main()
