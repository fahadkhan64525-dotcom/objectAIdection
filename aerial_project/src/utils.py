"""
utils.py
────────
General-purpose helper functions for the Aerial Object Classification project.
"""

import os
import json
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from PIL import Image


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    """Set seeds for Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🌱 Seeds set to {seed}")


# ─── Hardware Info ────────────────────────────────────────────────────────────

def print_hardware_info():
    """Print available GPU / CPU information."""
    gpus = tf.config.list_physical_devices("GPU")
    print("\n💻 Hardware Info")
    print("─" * 40)
    print(f"  TensorFlow : {tf.__version__}")
    if gpus:
        for g in gpus:
            print(f"  GPU        : {g.name}")
        # Enable memory growth to avoid OOM on small GPUs
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
    else:
        print("  GPU        : None (running on CPU)")
    print("─" * 40)


# ─── Image Utilities ──────────────────────────────────────────────────────────

def load_and_preprocess_image(
    img_path: str,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Load a JPEG/PNG image, resize, normalize to [0,1].

    Returns:
        np.ndarray of shape (H, W, 3), float32, values in [0,1].
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def pil_to_array(pil_img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Convert a PIL image to normalized numpy array."""
    pil_img = pil_img.convert("RGB").resize(target_size, Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32) / 255.0


def draw_bounding_boxes(
    image: np.ndarray,
    detections: list,
    font_scale: float = 0.7,
) -> np.ndarray:
    """
    Draw YOLO bounding boxes on a numpy RGB image.

    Args:
        image:      H×W×3 RGB float or uint8 image.
        detections: List of dicts from yolo_pipeline.predict_image_yolo().
        font_scale: OpenCV font scale.

    Returns:
        Annotated uint8 RGB image.
    """
    COLORS = {"bird": (63, 185, 80), "drone": (248, 81, 73)}
    DEFAULT_COLOR = (88, 166, 255)

    img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name  = det["class_name"]
        conf      = det["confidence"]
        color     = COLORS.get(cls_name, DEFAULT_COLOR)
        label     = f"{cls_name}: {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return img


# ─── Model Utilities ──────────────────────────────────────────────────────────

def load_keras_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model (.keras or .h5)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded: {model_path}")
    return model


def get_model_size_mb(model_path: str) -> float:
    """Return file size of a saved model in MB."""
    size_bytes = os.path.getsize(model_path)
    return round(size_bytes / (1024 ** 2), 2)


# ─── GradCAM ──────────────────────────────────────────────────────────────────

def gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer: str = None,
) -> np.ndarray:
    """
    Compute GradCAM heatmap for a given image.

    Args:
        model:           Trained Keras model.
        img_array:       Preprocessed image (1, H, W, 3).
        last_conv_layer: Name of the last convolutional layer.
                         Auto-detected if None.

    Returns:
        Heatmap array (H, W) in [0, 1].
    """
    if last_conv_layer is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads  = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Superimpose GradCAM heatmap onto the original image.

    Returns:
        Overlaid uint8 RGB image.
    """
    img = (original_img * 255).astype(np.uint8) if original_img.max() <= 1 else original_img
    h, w = img.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)
    return overlay


# ─── Timing ───────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""
    def __enter__(self):
        self._start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ─── Dataset stats ────────────────────────────────────────────────────────────

def count_dataset_images(dataset_root: str) -> dict:
    """Count images per class per split."""
    root = Path(dataset_root)
    report = {}
    for split in ["TRAIN", "VALID", "TEST"]:
        split_path = root / split
        if not split_path.exists():
            continue
        report[split] = {}
        for cls in split_path.iterdir():
            if cls.is_dir():
                count = len(list(cls.glob("*.jpg")) + list(cls.glob("*.png")))
                report[split][cls.name] = count
    return report
