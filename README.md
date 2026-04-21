# 🎯 Efficient and Accurate Face Recognition System Using Deep Learning

A fully local, beginner-friendly face recognition application built with **DeepFace**, **OpenCV**, and **Streamlit**.  
No C++ build tools, no dlib, no cloud required.

---

## 🗂 Project Structure

```
project/
├── app.py                  # Streamlit GUI
├── train.py                # Build face embeddings DB from dataset/
├── train_custom_model.py   # (Optional) Fine-tune MobileNetV2 classifier
├── recognize.py            # Core recognition logic
├── utils.py                # Drawing helpers
├── dataset/                # Your face images go here
├── models/                 # Saved embeddings & summaries
├── outputs/                # Processed videos saved here
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create a Python 3.11 virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** DeepFace will automatically download the Facenet512 model weights (~92 MB) on the first run.

---

## 📂 Dataset Format

Create a `dataset/` folder in the project root with one sub-folder per person:

```
dataset/
├── alice/
│   ├── alice_001.jpg
│   ├── alice_002.jpg
│   └── ...          ← 10–20 clear frontal photos recommended
├── bob/
│   ├── bob_001.jpg
│   └── ...
└── carol/
    └── ...
```

**Tips for best accuracy:**
- Use clear, well-lit frontal face photos.
- Include varied lighting / slight angles.
- JPEG or PNG format.
- Minimum **5 images per person**; 15–20 is ideal.

---

## 🚀 Training

### Step 1 – Build DeepFace embeddings (required for the app)

```bash
python train.py
```

This scans `dataset/`, computes Facenet512 embeddings for every image, and saves them to `models/face_db.pkl`.  
It also runs a quick validation split and prints approximate accuracy.

### Step 2 – (Optional) Train a custom MobileNetV2 classifier

```bash
# Use your own dataset/ folder
python train_custom_model.py

# Auto-download Olivetti Faces (no manual dataset needed – great for testing!)
python train_custom_model.py --olivetti

# Use LFW (extract lfw-funneled.tgz first, then):
python train_custom_model.py --dataset lfw --min_images 15 --epochs 30
```

---

## 🖥 Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧭 App Modes

| Mode | Description |
|------|-------------|
| 🏠 Home / Train | Scan dataset, train, view accuracy summary |
| 📷 Webcam Live | Real-time recognition with FPS counter |
| 🖼 Upload Image | Recognise faces in a still image |
| 🎬 Upload Video | Process video frame-by-frame, save result to `outputs/` |

---

## 📦 Open-Source Datasets

| Dataset | People | Images | Notes |
|---------|--------|--------|-------|
| **Olivetti Faces** | 40 | 400 | Auto-downloaded via sklearn, great for quick testing |
| **LFW (funneled)** | 5,749 | 13,233 | Industry standard, ~200 MB |
| **MS-Celeb-1M subset** | 100k+ | 10M+ | Large-scale, requires manual download |

For training the custom MobileNetV2 model we recommend **LFW** (filter to people with ≥ 15 images for best results).

---

## 🔧 Example Commands

```bash
# Full workflow
python train.py                          # Build embeddings
streamlit run app.py                     # Launch GUI

# Custom model (optional)
python train_custom_model.py --olivetti  # Quick demo with Olivetti
python train_custom_model.py --dataset lfw --min_images 15

# Specify epochs / image size
python train_custom_model.py --epochs 30 --img_size 112
```

---

## 🛠 Tech Stack

- **Python 3.11**
- **DeepFace 0.0.93** – Facenet512 embeddings for recognition
- **OpenCV 4.9** – Face detection & bounding boxes
- **Streamlit 1.35** – GUI
- **TensorFlow / Keras** – Custom MobileNetV2 classifier
- **scikit-learn** – Train/val split, accuracy metrics
- **NumPy, Pillow, Pandas**

---

## ❓ FAQ

**Q: Does this work without a GPU?**  
A: Yes. Facenet512 inference runs on CPU in ~200–400 ms per frame. For real-time webcam, every 3rd frame is processed to maintain ~10 FPS.

**Q: How do I add a new person?**  
A: Add a new folder under `dataset/` with their photos, then re-run `python train.py`.

**Q: What is the confidence score?**  
A: It's the cosine similarity between the query embedding and the best matching stored embedding, scaled to 0–100%.
