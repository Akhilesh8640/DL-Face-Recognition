"""
recognize.py - Core recognition logic shared by app.py and webcam mode.
"""

import pickle
import numpy as np
from deepface import DeepFace

DB_PATH    = "models/face_db.pkl"
MODEL_NAME = "Facenet512"
THRESHOLD  = 0.40   # cosine distance threshold (lower = stricter)


def load_db():
    """Load embeddings database from disk."""
    try:
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_embedding(img_bgr):
    """
    Get face embedding from a BGR numpy image (OpenCV frame).
    Returns list of dicts: [{embedding, region}]
    """
    try:
        results = DeepFace.represent(
            img_path         = img_bgr,
            model_name       = MODEL_NAME,
            enforce_detection= False,
            detector_backend = "opencv",
        )
        return results
    except Exception:
        return []


def recognize_faces(img_bgr, db_records):
    """
    Detect and recognise all faces in a BGR image.

    Returns list of dicts:
        name, confidence (0-100), x, y, w, h
    """
    if not db_records:
        return []

    results = get_embedding(img_bgr)
    faces   = []

    for res in results:
        emb    = np.array(res["embedding"])
        region = res.get("facial_area", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        # Compare against every stored embedding
        best_name, best_sim = "Unknown", -1
        for record in db_records:
            sim = cosine_similarity(emb, record["embedding"])
            if sim > best_sim:
                best_sim, best_name = sim, record["name"]

        # Convert cosine similarity → confidence percentage
        # Similarity 1.0 = perfect match, 0.0 = no match
        # We gate at THRESHOLD; below that → Unknown
        if best_sim < (1 - THRESHOLD):
            best_name = "Unknown"

        confidence = round(max(0, best_sim) * 100, 1)

        faces.append({
            "name"      : best_name,
            "confidence": confidence,
            "x": x, "y": y, "w": w, "h": h,
        })

    return faces
