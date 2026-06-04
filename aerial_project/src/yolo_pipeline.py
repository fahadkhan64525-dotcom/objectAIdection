"""
yolo_pipeline.py
────────────────
YOLOv8 training, validation, and inference pipeline
for Aerial Object Detection (Bird / Drone).

Requirements:
  pip install ultralytics

Dataset structure (already in YOLOv8 format):
  detection_dataset/
    images/
      train/  valid/  test/
    labels/
      train/  valid/  test/
"""

import os
import shutil
import yaml
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─── YAML Config ──────────────────────────────────────────────────────────────

def create_data_yaml(
    detection_root: str,
    save_path: str = "configs/data.yaml",
) -> str:
    """
    Generate the YOLOv8 data.yaml configuration file.

    Args:
        detection_root: Root folder of the detection dataset.
        save_path:      Where to save the YAML file.

    Returns:
        Absolute path to the saved YAML file.
    """
    detection_root = str(Path(detection_root).resolve())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cfg = {
        "path"  : detection_root,
        "train" : "images/train",
        "val"   : "images/valid",
        "test"  : "images/test",
        "nc"    : 2,
        "names" : ["bird", "drone"],
    }

    with open(save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"✅ data.yaml written → {save_path}")
    print(f"   Dataset root : {detection_root}")
    print(f"   Classes      : {cfg['names']}")
    return os.path.abspath(save_path)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_yolo(
    data_yaml:  str,
    model_size: str = "yolov8n",     # n / s / m / l / x
    epochs:     int = 50,
    imgsz:      int = 640,
    batch:      int = 16,
    project:    str = "models",
    name:       str = "yolo_aerial",
    device:     str = "0",           # '0' for GPU, 'cpu' for CPU
) -> object:
    """
    Train a YOLOv8 detection model.

    Args:
        data_yaml:  Path to data.yaml.
        model_size: YOLOv8 variant (default: yolov8n — nano, fastest).
        epochs:     Number of training epochs.
        imgsz:      Input image size.
        batch:      Batch size (-1 for auto).
        project:    Output directory.
        name:       Run name (sub-folder inside project).
        device:     CUDA device id or 'cpu'.

    Returns:
        Ultralytics Results object.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    print(f"\n🎯 Starting YOLOv8 Training")
    print(f"   Model    : {model_size}.pt")
    print(f"   Data     : {data_yaml}")
    print(f"   Epochs   : {epochs}")
    print(f"   Image sz : {imgsz}")
    print(f"   Batch    : {batch}")
    print(f"   Device   : {device}\n")

    model = YOLO(f"{model_size}.pt")

    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        project = project,
        name    = name,
        device  = device,
        patience= 15,
        save    = True,
        plots   = True,
        verbose = True,
        workers = 4,
        augment = True,
        lr0     = 0.01,
        lrf     = 0.001,
        warmup_epochs = 3,
        cos_lr  = True,
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"   Best weights → {best_path}")
    return results, str(best_path)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_yolo(
    weights_path: str,
    data_yaml:    str,
    imgsz:        int = 640,
    split:        str = "test",
) -> dict:
    """
    Validate YOLOv8 model on a dataset split.

    Returns:
        Dict with mAP50, mAP50-95, precision, recall.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split)

    results = {
        "mAP50"    : float(metrics.box.map50),
        "mAP50_95" : float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall"   : float(metrics.box.mr),
    }

    print(f"\n📊 YOLOv8 Validation Results ({split} split)")
    print("─" * 40)
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")
    print("─" * 40)
    return results


# ─── Inference ────────────────────────────────────────────────────────────────

def run_yolo_inference(
    weights_path: str,
    source: str,          # image path, folder, or 0 for webcam
    conf: float   = 0.25,
    iou:  float   = 0.45,
    imgsz: int    = 640,
    save:  bool   = True,
    project: str  = "models",
    name:    str  = "yolo_inference",
) -> list:
    """
    Run YOLOv8 inference on an image / folder / video stream.

    Returns:
        List of ultralytics Results objects.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.predict(
        source  = source,
        conf    = conf,
        iou     = iou,
        imgsz   = imgsz,
        save    = save,
        project = project,
        name    = name,
        verbose = True,
    )
    print(f"✅ Inference complete — {len(results)} images processed")
    return results


# ─── Visualise detections ─────────────────────────────────────────────────────

def visualize_detections(
    weights_path: str,
    image_path:   str,
    conf:         float = 0.25,
    save_path:    str   = None,
):
    """
    Plot a single image with YOLO bounding boxes using matplotlib.
    """
    from ultralytics import YOLO

    CLASSES = ["bird", "drone"]
    COLORS  = {"bird": "#3fb950", "drone": "#f85149"}

    model = YOLO(weights_path)
    results = model.predict(image_path, conf=conf, verbose=False)[0]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.imshow(img)
    ax.axis("off")

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            color    = COLORS.get(cls_name, "#58a6ff")

            rect = patches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
                boxstyle="round,pad=2",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 8, f"{cls_name} {conf_val:.2f}",
                    color="white", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor="none"))

    n_det = len(results.boxes) if results.boxes else 0
    ax.set_title(f"YOLOv8 Detection — {n_det} object(s) found",
                 color="white", fontsize=13, pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✅ Detection visualisation saved → {save_path}")
    plt.show()


# ─── Result boxes for Streamlit ───────────────────────────────────────────────

def predict_image_yolo(
    weights_path: str,
    image_array:  np.ndarray,
    conf:         float = 0.25,
) -> list:
    """
    Run YOLO on a numpy image (H, W, 3) RGB.

    Returns:
        List of dicts: {class_name, confidence, bbox: [x1, y1, x2, y2]}
    """
    from ultralytics import YOLO
    import tempfile, uuid

    CLASSES = ["bird", "drone"]
    model = YOLO(weights_path)

    # Save temp image for YOLO
    tmp = os.path.join(tempfile.gettempdir(), f"yolo_tmp_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(tmp, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    results = model.predict(tmp, conf=conf, verbose=False)[0]
    os.remove(tmp)

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_name": CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id),
                "confidence": round(float(box.conf[0]), 4),
                "bbox":       [round(v) for v in box.xyxy[0].tolist()],
            })
    return detections
