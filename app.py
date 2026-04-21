"""
app.py - Streamlit GUI for the Face Recognition System.
Run: streamlit run app.py
"""

import os
import json
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import recognize as rec
import utils

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    color: #e2e8f0;
  }
  .metric-card h2 { color: #00e676; margin: 4px 0; font-size: 2rem; }
  .metric-card p  { margin: 0; font-size: 0.85rem; color: #94a3b8; }
  .status-ok   { color: #00e676; font-weight: 600; }
  .status-warn { color: #ff6b35; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/face-id.png", width=72)
    st.title("FaceRecog AI")
    st.caption("Local Deep Learning Face Recognition")
    st.divider()
    mode = st.radio(
        "Mode",
        ["🏠 Home / Train", "📷 Webcam Live", "🖼 Upload Image", "🎬 Upload Video"],
        label_visibility="collapsed",
    )
    st.divider()
    db_records = rec.load_db()
    if db_records:
        st.markdown(f'<p class="status-ok">✅ DB loaded – {len(db_records)} embeddings</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-warn">⚠ No DB – train first</p>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME / TRAIN
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in mode:
    st.title("🎯 Face Recognition System")
    st.markdown("**A local Deep Learning system using DeepFace + Facenet512**")
    st.divider()

    # ── Training controls ──────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📂 Dataset & Training")
        dataset_ok = os.path.isdir("dataset") and any(
            os.path.isdir(os.path.join("dataset", p)) for p in os.listdir("dataset")
        ) if os.path.isdir("dataset") else False

        if dataset_ok:
            people = [p for p in os.listdir("dataset")
                      if os.path.isdir(os.path.join("dataset", p))]
            st.success(f"Dataset found: **{len(people)} people** → {', '.join(people)}")
        else:
            st.warning("No dataset found. Create `dataset/<person_name>/` folders with face images.")

        if st.button("🚀 Scan & Train Dataset", type="primary", use_container_width=True):
            if not dataset_ok:
                st.error("Please create the dataset first.")
            else:
                with st.spinner("Building face embeddings … this may take a minute"):
                    ret = os.system("python train.py")
                if ret == 0:
                    st.success("✅ Training complete! Reload the page to refresh the DB.")
                    st.balloons()
                else:
                    st.error("Training failed. Check the terminal for errors.")

    with col2:
        st.subheader("📊 Summary")
        summary_path = "models/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            st.markdown(f"""
            <div class="metric-card"><h2>{s['num_people']}</h2><p>People</p></div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card"><h2>{s['num_images']}</h2><p>Training Images</p></div>
            """, unsafe_allow_html=True)
            st.markdown("")
            acc = s.get("accuracy_pct")
            acc_str = f"{acc}%" if acc is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card"><h2>{acc_str}</h2><p>Val Accuracy</p></div>
            """, unsafe_allow_html=True)
        else:
            st.info("Train the model to see summary.")

    st.divider()
    st.subheader("📋 Dataset Instructions")
    st.code("""
dataset/
├── alice/
│   ├── alice_001.jpg
│   ├── alice_002.jpg
│   └── ...          (10–20 photos recommended)
├── bob/
│   ├── bob_001.jpg
│   └── ...
└── carol/
    └── ...
    """, language="text")
    st.info("Use clear frontal face photos, varied lighting. **Minimum 5 images per person.** "
            "JPEG or PNG format.")


# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM LIVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Webcam" in mode:
    st.title("📷 Live Webcam Recognition")

    if not db_records:
        st.error("No face database found. Please train the model first (Home tab).")
        st.stop()

    run       = st.toggle("▶ Start Webcam", value=False)
    frame_ph  = st.empty()
    info_ph   = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Make sure it is connected.")
            st.stop()

        prev_time = time.time()
        # Process every Nth frame for speed
        SKIP = 3
        frame_count = 0
        last_faces  = []

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % SKIP == 0:
                last_faces = rec.recognize_faces(frame, db_records)

            annotated = utils.draw_faces(frame, last_faces)

            # FPS
            now  = time.time()
            fps  = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            utils.overlay_fps(annotated, fps)

            frame_ph.image(utils.bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

            # Info bar
            names = [f["name"] for f in last_faces]
            info_ph.caption(f"Detected: {names or 'none'}   |   FPS: {fps:.1f}")

            # Allow Streamlit to check the toggle
            time.sleep(0.01)

        cap.release()
    else:
        st.info("Toggle **Start Webcam** above to begin live recognition.")


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════
elif "Image" in mode:
    st.title("🖼 Image Recognition")

    if not db_records:
        st.error("No face database found. Please train the model first.")
        st.stop()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Recognising faces …"):
            faces = rec.recognize_faces(img_bgr, db_records)

        annotated = utils.draw_faces(img_bgr, faces)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_pil, use_container_width=True)
        with col2:
            st.subheader(f"Result – {len(faces)} face(s) found")
            st.image(utils.bgr_to_rgb(annotated), use_container_width=True)

        if faces:
            st.subheader("Detections")
            for i, f in enumerate(faces, 1):
                st.markdown(f"**{i}.** `{f['name']}` — confidence: **{f['confidence']}%**")
        else:
            st.warning("No faces detected.")


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD VIDEO
# ══════════════════════════════════════════════════════════════════════════════
elif "Video" in mode:
    st.title("🎬 Video Recognition")

    if not db_records:
        st.error("No face database found. Please train the model first.")
        st.stop()

    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_vid:
        os.makedirs("outputs", exist_ok=True)

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded_vid.read())
        tmp.close()

        cap    = cv2.VideoCapture(tmp.name)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path  = os.path.join("outputs", "processed_" + uploaded_vid.name)
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        writer    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress  = st.progress(0, text="Processing video …")
        SKIP      = 5   # recognise every 5th frame; copy others unchanged
        last_faces = []

        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if idx % SKIP == 0:
                last_faces = rec.recognize_faces(frame, db_records)
            annotated = utils.draw_faces(frame, last_faces)
            writer.write(annotated)
            progress.progress(min((idx + 1) / max(total, 1), 1.0),
                              text=f"Frame {idx+1}/{total}")

        cap.release()
        writer.release()
        os.unlink(tmp.name)
        progress.empty()

        st.success(f"✅ Saved → `{out_path}`")
        st.video(out_path)
