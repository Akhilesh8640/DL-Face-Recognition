"""
utils.py - Drawing helpers and shared utilities.
"""

import cv2
import numpy as np


# Color palette (BGR)
BOX_COLOR     = (0, 230, 118)   # green
UNKNOWN_COLOR = (0, 60, 255)    # red
TEXT_BG_COLOR = (30, 30, 30)
TEXT_COLOR    = (255, 255, 255)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


def draw_faces(img_bgr, faces):
    """
    Draw bounding boxes + labels on a copy of the image.

    faces: list of dicts from recognize.recognize_faces()
    """
    out = img_bgr.copy()

    for face in faces:
        x, y, w, h   = face["x"], face["y"], face["w"], face["h"]
        name         = face["name"]
        confidence   = face["confidence"]
        color        = BOX_COLOR if name != "Unknown" else UNKNOWN_COLOR

        # Bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{name}  {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(out, (x, y - th - 10), (x + tw + 6, y), color, -1)

        # Label text
        cv2.putText(out, label, (x + 3, y - 5), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    return out


def bgr_to_rgb(img_bgr):
    """OpenCV BGR → PIL/Streamlit-friendly RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def overlay_fps(img_bgr, fps):
    """Stamp FPS onto top-left corner."""
    label = f"FPS: {fps:.1f}"
    cv2.putText(img_bgr, label, (10, 28), FONT, 0.8, (0, 255, 200), 2, cv2.LINE_AA)
    return img_bgr
