# -*- coding: utf-8 -*-

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import easyocr
import os
import urllib.request
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️",
    layout="wide"
)


# ======================================
# HEADER
# ======================================
st.title("🕵️ Fake Job Posting Detection")
st.divider()


# ======================================
# LOAD MODEL + TOKENIZER + OCR
# ======================================
import gdown

@st.cache_resource
def load_assets():
    MODEL_FILE_ID = "16W7VOvJrXEQTOXje8a7xdl_T42OC5ZMq"
    
    model_path = "fake_job_multimodal_model.keras"
    tokenizer_path = "tokenizer.pkl"

    if not os.path.exists(model_path):
        with st.spinner("Downloading Multimodal Model from Google Drive..."):
            
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            
            gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    reader = easyocr.Reader(['en'])

    return model, tokenizer, reader


try:
    model, tokenizer, reader = load_assets()

except Exception as e:
    st.error(f"Loading Error:\n{e}")
    st.stop()
# ======================================
# MAIN LAYOUT
# ======================================
col1, col2 = st.columns([1, 1.2])


# ======================================
# LEFT COLUMN
# ======================================
with col1:
    st.subheader("Upload Job Image")

    uploaded_file = st.file_uploader(
        "Choose Job Advertisement",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Uploaded Advertisement Preview",
            use_container_width=True
        )


# ======================================
# RIGHT COLUMN
# ======================================
with col2:
    st.subheader(" AI Detection Control")

    detect = st.button(
        "Run Auto-Detection",
        use_container_width=True,
        type="primary"
    )


# ======================================
# PREDICTION & RESULTS
# ======================================
if detect:
    if uploaded_file is None:
        st.warning(
            "Please upload an image first."
        )
    else:
        with st.spinner(
            "Analyzing Job Posting components..."
        ):
            try:
                # =========================
                # IMAGE READING
                # =========================
                file_bytes = np.frombuffer(
                    uploaded_file.read(),
                    np.uint8
                )

                img = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

                img_rgb = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )


                # =========================
                # OCR TEXT EXTRACTION
                # =========================
                ocr_results = reader.readtext(img)

                combined_text = " ".join(
                    [res[1] for res in ocr_results]
                )


                # =========================
                # TEXT PREPROCESSING
                # =========================
                sequences = tokenizer.texts_to_sequences(
                    [combined_text]
                )

                X_text = pad_sequences(
                    sequences,
                    maxlen=200,
                    padding='post'
                )


                # =========================
                # IMAGE PREPROCESSING
                # =========================
                img_resized = cv2.resize(
                    img_rgb,
                    (128, 128)
                )

                X_img = np.array(
                    [img_resized]
                ) / 255.0


                # =========================
                # MODEL PREDICTION
                # =========================
                prediction = model.predict(
                    [X_img, X_text]
                )

                score = prediction[0][0]
                confidence = score * 100


                # ======================================
                # RESULTS CONTAINER (Inside col2 for clean look)
                # ======================================
                with col2:
                    st.divider()
                    st.subheader("📊 System Analytics Output")
                    

                    if score > 0.5:
                        st.error(
                            "Result: Fake / Fraudulent Job Posting"
                        )
                        st.metric(
                            label="Fraud Probability Score",
                            value=f"{confidence:.2f}%"
                        )
                    else:
                        st.success(
                            "Result: Safe & Real Job Posting"
                        )
                        st.metric(
                            label="Fraud Probability Score",
                            value=f"{confidence:.2f}%"
                        )

                    # ======================================
                    # OCR DETAILS
                    # ======================================
                    st.write("") 
                    st.subheader("Extracted OCR Text")
                    st.text_area(
                             "Detected Text",
                             combined_text,
                             height=200
                         )

            except Exception as e:
                st.error(
                    f"Prediction Error:\n{e}"
                )
