"""
train_custom_model.py
─────────────────────
Fine-tunes a MobileNetV2 classifier on your face dataset (LFW or your own).
The resulting model is saved to models/custom_face_model.h5 and used as an
optional alternative recogniser.

Recommended open-source dataset
────────────────────────────────
  LFW (Labeled Faces in the Wild) – cropped & funneled version
  URL  : http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz  (~200 MB)
  Pages: http://vis-www.cs.umass.edu/lfw/
  
  After download, extract so that:
      lfw/
        Aaron_Eckhart/
          Aaron_Eckhart_0001.jpg
        ...
  Then point DATASET_DIR below to "lfw" (or copy a subset into "dataset/").
  For quick testing, use the 62-class "Olivetti Faces" dataset which can be
  downloaded automatically (see --olivetti flag below).

Usage
─────
  # Use your own dataset/ folder (already populated):
  python train_custom_model.py

  # Use the Olivetti faces dataset (auto-downloaded, no extra files needed):
  python train_custom_model.py --olivetti

  # Use LFW (already extracted to lfw/ folder, keep only people with ≥15 imgs):
  python train_custom_model.py --dataset lfw --min_images 15
"""

import os, argparse, json
import numpy as np

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    default="dataset",
                    help="Path to dataset folder (default: dataset/)")
parser.add_argument("--min_images", type=int, default=5,
                    help="Min images per person (default: 5)")
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--img_size",   type=int, default=96)
parser.add_argument("--olivetti",   action="store_true",
                    help="Auto-download & use Olivetti Faces dataset")
args = parser.parse_args()

# ── Imports (heavy, after arg parse) ─────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE   = (args.img_size, args.img_size)
BATCH      = 32
EPOCHS     = args.epochs
MODEL_OUT  = os.path.join(MODELS_DIR, "custom_face_model.keras")
LABELS_OUT = os.path.join(MODELS_DIR, "custom_labels.json")

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_olivetti():
    """Download Olivetti Faces via sklearn and save as dataset/."""
    from sklearn.datasets import fetch_olivetti_faces
    print("Downloading Olivetti Faces (sklearn) …")
    data  = fetch_olivetti_faces(shuffle=True, random_state=42)
    images, labels = data.images, data.target
    out_dir = "dataset_olivetti"

    for i, (img, lbl) in enumerate(zip(images, labels)):
        person_dir = os.path.join(out_dir, f"person_{lbl:02d}")
        os.makedirs(person_dir, exist_ok=True)
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil.save(os.path.join(person_dir, f"{i:04d}.png"))

    print(f"Saved Olivetti dataset → {out_dir}/")
    return out_dir


def load_dataset(dataset_dir, min_images=5):
    X, y = [], []
    people = sorted([
        p for p in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, p))
    ])
    kept = []
    for person in people:
        person_dir = os.path.join(dataset_dir, person)
        imgs = [f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        if len(imgs) < min_images:
            print(f"  Skipping '{person}' (only {len(imgs)} images < {min_images})")
            continue
        kept.append(person)
        for img_f in imgs:
            try:
                img = Image.open(os.path.join(person_dir, img_f)).convert("RGB")
                img = img.resize(IMG_SIZE)
                X.append(np.array(img) / 255.0)
                y.append(person)
            except Exception as e:
                print(f"  ⚠ {img_f}: {e}")

    print(f"Loaded {len(X)} images for {len(kept)} people.")
    return np.array(X, dtype=np.float32), np.array(y), kept


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes):
    base = keras.applications.MobileNetV2(
        input_shape = (*IMG_SIZE, 3),
        include_top = False,
        weights     = "imagenet",
    )
    # Fine-tune last 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def augment_ds(ds):
    """Apply random flips + brightness to dataset."""
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ])
    return ds.map(lambda x, y: (aug(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dataset_dir = load_olivetti() if args.olivetti else args.dataset

    if not os.path.isdir(dataset_dir):
        print(f"Dataset folder '{dataset_dir}' not found.")
        return

    X, y_raw, people = load_dataset(dataset_dir, args.min_images)
    if len(people) < 2:
        print("Need at least 2 people with enough images.")
        return

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # Train / val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    num_classes = len(people)
    print(f"\nClasses: {num_classes}  |  Train: {len(X_tr)}  |  Val: {len(X_val)}")

    # tf.data pipelines
    tr_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
             .shuffle(1000).batch(BATCH))
    tr_ds = augment_ds(tr_ds).prefetch(tf.data.AUTOTUNE)

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH).prefetch(tf.data.AUTOTUNE))

    model = build_model(num_classes)
    model.compile(
        optimizer = keras.optimizers.Adam(1e-4),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs …\n")
    history = model.fit(tr_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks)

    # Evaluate
    _, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✅  Validation accuracy: {val_acc*100:.2f}%")

    # Save
    model.save(MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(list(le.classes_), f)

    print(f"✅  Model  → {MODEL_OUT}")
    print(f"✅  Labels → {LABELS_OUT}")
    print("\nThe model is ready. DeepFace embeddings (train.py) are used for")
    print("real-time recognition in the app; this model is your custom classifier.")


if __name__ == "__main__":
    main()
