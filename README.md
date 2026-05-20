# Fake-Job-Posting-Detection
# 🕵️ Fake Job Posting Detection using Multimodal Deep Learning

A multimodal deep learning system for detecting fraudulent job advertisements using both:

* 🖼️ Image Analysis (CNN + Transfer Learning)
* 📝 Text Analysis (NLP + Embedding)
* 🔍 OCR Text Extraction

The system automatically extracts text from uploaded job posters using OCR, then analyzes both the image and extracted text to classify the job posting as:

* ✅ Real
* 🚨 Fake / Fraudulent

---

# 📌 Project Features

* Multimodal Deep Learning Architecture
* OCR-based automatic text extraction
* CNN for image feature extraction
* NLP pipeline for textual analysis
* Transfer Learning using MobileNetV2
* Streamlit interactive web application
* Real-time fraud prediction

---

# 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* EasyOCR
* Streamlit
* NumPy
* Pandas
* Scikit-learn

---

# 📂 Dataset

Dataset used:

**Multimodal Real / Fake Job Posting Prediction**

The dataset contains:

* Job advertisement text
* Job poster images
* Fraud labels

Target column:

```python
fraudulent
```

Where:

* `0` → Real Job Posting
* `1` → Fake / Fraudulent Job Posting

---

# 🏗️ Model Architecture

## 🔹 Image Branch

* MobileNetV2 (Transfer Learning)
* Custom CNN layers
* GlobalAveragePooling2D

## 🔹 Text Branch

* Tokenizer
* Embedding Layer
* GlobalAveragePooling1D

## 🔹 Fusion Layer

The extracted image features and text features are concatenated into a single multimodal representation before classification.

---

# 🔄 System Pipeline

```text
Uploaded Job Image
        ↓
OCR Text Extraction
        ↓
Text Preprocessing
        ↓
CNN Image Processing
        ↓
Multimodal Fusion
        ↓
Fraud Prediction
```

---

# 📊 Model Performance

The model achieved high classification accuracy on the testing dataset.

Evaluation metrics used:

* Accuracy
* Loss
* Confusion Matrix
* Classification Report

---

# 💻 Streamlit Application

The application allows users to:

1. Upload a job advertisement image
2. Automatically extract text using OCR
3. Analyze image + text simultaneously
4. Detect whether the advertisement is fake or real

---

# ▶️ Run the Application

Install dependencies:

```bash
pip install tensorflow streamlit easyocr opencv-python pandas numpy scikit-learn
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# 📁 Saved Files

* `fake_job_multimodal_model.keras` → trained model
* `tokenizer.pkl` → saved tokenizer
* `accuracy_multimodal.png` → accuracy graph
* `loss.png` → loss graph

---

# 👩‍💻 Author

Developed as a Deep Learning multimodal project for fraudulent job posting detection.
