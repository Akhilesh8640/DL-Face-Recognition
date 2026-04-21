"""
train.py - Build face embeddings database from dataset/ folder.
Run: python train.py
"""

import os
import json
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
MODELS_DIR  = "models"
DB_PATH     = os.path.join(MODELS_DIR, "face_db.pkl")
SUMMARY_PATH = os.path.join(MODELS_DIR, "summary.json")
MODEL_NAME  = "Facenet512"   # accurate & no C++ deps

os.makedirs(MODELS_DIR, exist_ok=True)


def build_embeddings():
    """Walk dataset/, compute embeddings for every image, return list of dicts."""
    records = []
    people  = [p for p in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, p))]

    if not people:
        print("⚠  No person folders found in dataset/.")
        return [], []

    print(f"Found {len(people)} people: {people}\n")

    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        images = [f for f in os.listdir(person_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"  Processing '{person}' – {len(images)} images …")
        for img_file in images:
            img_path = os.path.join(person_dir, img_file)
            try:
                result = DeepFace.represent(
                    img_path  = img_path,
                    model_name= MODEL_NAME,
                    enforce_detection=False,
                )
                embedding = np.array(result[0]["embedding"])
                records.append({"name": person, "embedding": embedding, "path": img_path})
            except Exception as e:
                print(f"    ⚠  Skipping {img_file}: {e}")

    return records, people


def compute_accuracy(records):
    """Simple cosine-similarity leave-out accuracy estimate."""
    if len(records) < 4:
        return None

    # Split into train / val
    train_recs, val_recs = train_test_split(records, test_size=0.2, random_state=42,
                                            stratify=[r["name"] for r in records])

    correct = 0
    for v in val_recs:
        best_name, best_sim = "Unknown", -1
        for t in train_recs:
            # Cosine similarity
            sim = np.dot(v["embedding"], t["embedding"]) / (
                np.linalg.norm(v["embedding"]) * np.linalg.norm(t["embedding"]) + 1e-9)
            if sim > best_sim:
                best_sim, best_name = sim, t["name"]
        if best_name == v["name"]:
            correct += 1

    return round(correct / len(val_recs) * 100, 2)


def main():
    print("=" * 50)
    print("  Face Recognition – Training Step")
    print("=" * 50)

    records, people = build_embeddings()
    if not records:
        return

    # Save embeddings DB
    with open(DB_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"\n✅  Saved {len(records)} embeddings → {DB_PATH}")

    # Accuracy
    accuracy = compute_accuracy(records)

    # Summary
    total_images = len(records)
    summary = {
        "num_people"  : len(people),
        "num_images"  : total_images,
        "people"      : people,
        "model"       : MODEL_NAME,
        "accuracy_pct": accuracy,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  People   : {len(people)}")
    print(f"  Images   : {total_images}")
    print(f"  Approx accuracy (val split): {accuracy}%")
    print(f"✅  Summary → {SUMMARY_PATH}")
    print("\nTraining complete! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
