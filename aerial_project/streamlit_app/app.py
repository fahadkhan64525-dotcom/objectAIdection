"""
app.py — Streamlit Deployment
─────────────────────────────
Aerial Object Classification & Detection UI.

Run:
    .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app/app.py

Features:
  - Upload image → classify as Bird / Drone
  - Choose model (Custom CNN / Transfer Learning variants)
  - Show confidence score & probability bar
  - GradCAM visualisation overlay
  - Optional YOLOv8 object detection with bounding boxes
  - Model comparison metrics dashboard
"""

import os
import sys
import json
import glob
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from PIL import Image

# ── Allow imports from proje
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Lazy-load 
@st.cache_resource
def load_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow import failed. Launch Streamlit using the repository venv:\n"
            "  .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app/app.py\n"
            "If you are using a different Python environment, install compatible TensorFlow "
            "and rerun from the project virtual environment."
        ) from exc

@st.cache_resource
def load_keras_model_cached(path: str):
    tf = load_tf()
    return tf.keras.models.load_model(path)


# ─── Page
st.set_page_config(
    page_title="AerialVision AI",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e17;
    color: #e0e6f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e2a3a;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #111827, #1a2236);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 10px;
  }
  .metric-card .label {
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #38bdf8;
  }

  /* Prediction badge */
  .pred-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 10px 0;
  }
  .pred-bird  { background: rgba(63,185,80,0.15); color: #3fb950; border: 2px solid #3fb950; }
  .pred-drone { background: rgba(248,81,73,0.15);  color: #f85149; border: 2px solid #f85149; }

  /* Section headings */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.15rem;
    color: #93c5fd;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 20px 0 12px;
  }

  /* Hero */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .hero-sub {
    color: #64748b;
    font-size: 1rem;
    margin-top: 4px;
  }

  /* Confidence bar */
  .conf-bar-wrap { background: #1e293b; border-radius: 8px; overflow: hidden; height: 14px; margin: 8px 0; }
  .conf-bar-fill { height: 100%; border-radius: 8px; transition: width 0.5s; }

  /* Info pill */
  .info-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 2px;
  }

  /* Stacked metrics row */
  div[data-testid="column"] > div { height: 100%; }

  /* Streamlit override */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  .stSelectbox > div > div { background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; }
  .stFileUploader { border: 2px dashed #1e3a5f; border-radius: 12px; padding: 16px; }
  .stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8); }
</style>
""", unsafe_allow_html=True)


# ─── Co
CLASSES      = ["bird", "drone"]
IMG_SIZE     = (224, 224)
MODELS_DIR   = str(ROOT / "models")
HISTORY_DIR  = str(ROOT / "models")
YOLO_WEIGHTS_PATH = str((ROOT.parent / "yolov8n.pt").resolve())

CLASS_EMOJI  = {"bird": "🐦", "drone": "🚁"}
CLASS_INFO   = {
    "bird" : "Organic aerial object — likely wildlife or a bird species.",
    "drone": "Unmanned aerial vehicle — requires attention in restricted zones.",
}


# ─── Helpers────────────

def list_classification_models() -> dict:
    """Scan models/ for .keras files and return {display_name: path}."""
    models = {}
    if not os.path.isdir(MODELS_DIR):
        return models
    for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras"))):
        name = Path(p).stem.replace("_", " ").replace("best", "").strip().upper()
        models[name] = p
    return models


def find_yolo_weights() -> str | None:
    """Find best YOLO weights if available."""
    if os.path.exists(YOLO_WEIGHTS_PATH):
        return YOLO_WEIGHTS_PATH

    patterns = [
        os.path.join(MODELS_DIR, "**/weights/best.pt"),
        os.path.join(MODELS_DIR, "**/*.pt"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    matches = [p for p in sorted(matches) if os.path.isfile(p)]
    return matches[0] if matches else None


def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """PIL → (1, 224, 224, 3) float32 array."""
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def get_gradcam(model, img_array: np.ndarray) -> np.ndarray | None:
    """Return GradCAM heatmap or None if unavailable."""
    try:
        import tensorflow as tf
        last_conv = None
        for layer in reversed(model.layers):
            if hasattr(layer, "filters") or "conv" in layer.name.lower():
                last_conv = layer.name
                break
        if not last_conv:
            return None

        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            top_class = tf.argmax(preds[0])
            loss = preds[:, top_class]
        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception:
        return None


def heatmap_to_overlay(original_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_rgb   = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay    = cv2.addWeighted(original_rgb, 0.55, heat_rgb, 0.45, 0)
    return overlay


def draw_yolo_boxes(img_rgb: np.ndarray, detections: list) -> np.ndarray:
    COLORS = {"bird": (63, 185, 80), "drone": (248, 81, 73)}
    img = img_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls   = det["class_name"]
        conf  = det["confidence"]
        color = COLORS.get(cls, (88, 166, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{CLASS_EMOJI.get(cls,'')} {cls.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


def load_comparison_csv():
    path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        import pandas as pd
        return pd.read_csv(path, index_col=0)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
      <span style='font-size:2.4rem;'>🦅</span>
      <div style='font-family:Syne; font-weight:800; font-size:1.2rem;
                  background:linear-gradient(135deg,#38bdf8,#818cf8);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        AerialVision AI
      </div>
      <div style='color:#475569; font-size:0.75rem; margin-top:4px;'>
        Bird vs Drone Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Classify Image", "🎯 Detect Objects (YOLO)", "📊 Model Dashboard", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()

    # Model selector
    avail_models = list_classification_models()
    if avail_models:
        st.markdown("<div class='section-title'>Classification Model</div>", unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Model", list(avail_models.keys()), label_visibility="collapsed"
        )
        selected_model_path = avail_models[selected_model_name]
    else:
        st.warning("⚠️ No trained models found in `models/`.\nRun `scripts/run_training.py` first.")
        selected_model_path = None
        selected_model_name = None

    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)
    threshold  = st.slider("Classification threshold", 0.3, 0.9, 0.5, 0.01)
    show_cam   = st.toggle("Show GradCAM overlay", value=True)

    yolo_path  = find_yolo_weights()
    yolo_conf  = st.slider("YOLO confidence", 0.1, 0.9, 0.25, 0.05)

    if yolo_path:
        st.success(f"✅ YOLO weights found: {yolo_path}")
    else:
        st.info("ℹ️ No YOLO weights found. Train YOLOv8 to enable detection.")
        st.markdown(
            "\n" +
            "Use the dataset in YOLO format and run the training helper. " +
            "If your detection dataset is ready, update `scripts/run_training.py` and set `RUN_YOLO = True`."
        )


# ─── Page: Classify ───────────────────────────────────────────────────────────

if "Classify" in page:
    st.markdown("<div class='hero-title'>🦅...🚁 Aerial Object Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload an aerial image — get an instant Bird or Drone prediction.</div>", unsafe_allow_html=True)

    st.divider()

    uploaded = st.file_uploader(
        "Drop an image here (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded and selected_model_path:
        pil_img = Image.open(uploaded)
        img_rgb = np.array(pil_img.convert("RGB"))

        with st.spinner("Loading model..."):
            try:
                model = load_keras_model_cached(selected_model_path)
            except Exception as exc:
                st.error(f"Model load failed: {exc}")
                st.stop()

        img_input = preprocess_pil(pil_img)

        with st.spinner("Analysing image..."):
            prob = float(model.predict(img_input, verbose=0)[0][0])

        pred_idx    = int(prob >= threshold)
        label       = CLASSES[pred_idx]
        confidence  = prob if pred_idx == 1 else 1.0 - prob
        emoji       = CLASS_EMOJI[label]
        badge_cls   = f"pred-{label}"

        # ── Layout ───────────────────────────────────────────────────────────
        col_img, col_result = st.columns([1.1, 1], gap="large")

        with col_img:
            st.markdown("<div class='section-title'>Input Image</div>", unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True, caption="Uploaded image")

            if show_cam:
                heatmap = get_gradcam(model, img_input)
                if heatmap is not None:
                    st.markdown("<div class='section-title'>GradCAM Attention</div>", unsafe_allow_html=True)
                    overlay = heatmap_to_overlay(img_rgb, heatmap)
                    st.image(overlay, use_container_width=True, caption="Model attention overlay")
                else:
                    st.caption("GradCAM not available for this model architecture.")

        with col_result:
            st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='text-align:center; padding: 30px 20px;
                        background: linear-gradient(135deg, #111827, #1a2236);
                        border: 1px solid #1e3a5f; border-radius: 16px;
                        margin-bottom: 16px;'>
              <div style='font-size: 4rem; margin-bottom: 8px;'>{emoji}</div>
              <div class='pred-badge {badge_cls}'>{label.upper()}</div>
              <div style='color:#64748b; margin-top: 10px; font-size:0.9rem;'>
                {CLASS_INFO[label]}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            bar_color = "#3fb950" if label == "bird" else "#f85149"
            st.markdown(f"""
            <div style='margin: 16px 0 4px; color:#93c5fd; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em;'>
              Confidence
            </div>
            <div class='conf-bar-wrap'>
              <div class='conf-bar-fill' style='width:{confidence*100:.1f}%; background:{bar_color};'></div>
            </div>
            <div style='font-family:Space Mono; font-size:1.5rem; color:white; text-align:right;'>
              {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("<div class='section-title'>Probability Breakdown</div>", unsafe_allow_html=True)

            bird_prob  = 1.0 - prob
            drone_prob = prob

            for cls, p, color in [("🐦 Bird", bird_prob, "#3fb950"), ("🚁 Drone", drone_prob, "#f85149")]:
                is_pred = (cls.split()[1].lower() == label)
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:12px; margin:6px 0;'>
                  <span style='min-width:80px; color:#94a3b8; font-size:0.9rem;'>{cls}</span>
                  <div style='flex:1; background:#1e293b; border-radius:6px; overflow:hidden; height:12px;'>
                    <div style='width:{p*100:.1f}%; height:100%; background:{color}; opacity:{"1" if is_pred else "0.4"};'></div>
                  </div>
                  <span style='min-width:50px; font-family:Space Mono; font-size:0.85rem; color:white; text-align:right;'>{p:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            # Metadata
            st.markdown("<div class='section-title'>Image Info</div>", unsafe_allow_html=True)
            w, h = pil_img.size
            st.markdown(f"""
            <span class='info-pill'>📐 {w}×{h} px</span>
            <span class='info-pill'>🧠 {selected_model_name}</span>
            <span class='info-pill'>⚙️ threshold={threshold:.2f}</span>
            """, unsafe_allow_html=True)

    elif uploaded and not selected_model_path:
        st.error("No trained model found. Please train a model first.")
    else:
        st.markdown("""
        <div style='text-align:center; padding: 60px 40px;
                    background: linear-gradient(135deg, #111827, #1a2236);
                    border: 2px dashed #1e3a5f; border-radius: 16px;
                    margin-top: 20px;'>
          <div style='font-size:3rem; margin-bottom:12px;'>📸</div>
          <div style='color:#93c5fd; font-size:1.1rem; font-weight:600;'>Upload an aerial image to begin</div>
          <div style='color:#475569; font-size:0.85rem; margin-top:8px;'>Supports JPG and PNG formats</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Page: YOLO Detection ─────────────────────────────────────────────────────

elif "Detect" in page:
    st.markdown("<div class='hero-title'>Object Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>YOLOv8 — real-time bounding box detection for birds and drones.</div>", unsafe_allow_html=True)
    st.divider()

    if not yolo_path:
        st.warning("""
        **YOLOv8 weights not found.**

        Train the detection model first:
        ```python
        from src.yolo_pipeline import create_data_yaml, train_yolo
        yaml = create_data_yaml("path/to/detection_dataset")
        train_yolo(yaml, epochs=50)
        ```
        """)
    else:
        uploaded = st.file_uploader(
            "Upload image for detection",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            img_rgb = np.array(pil_img.convert("RGB"))

            with st.spinner("Running YOLOv8 detection..."):
                try:
                    from src.yolo_pipeline import predict_image_yolo
                    detections = predict_image_yolo(yolo_path, img_rgb, conf=yolo_conf)
                except Exception as e:
                    st.error(f"Detection error: {e}")
                    detections = []

            col_orig, col_det = st.columns(2, gap="large")

            with col_orig:
                st.markdown("<div class='section-title'>Original</div>", unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)

            with col_det:
                st.markdown("<div class='section-title'>Detections</div>", unsafe_allow_html=True)
                if detections:
                    annotated = draw_yolo_boxes(img_rgb, detections)
                    st.image(annotated, use_container_width=True)
                else:
                    st.image(pil_img, use_container_width=True)
                    st.caption("No objects detected above confidence threshold.")

            # Detection table
            if detections:
                st.markdown("<div class='section-title'>Detection Results</div>", unsafe_allow_html=True)
                for i, det in enumerate(detections, 1):
                    cls   = det["class_name"]
                    emoji = CLASS_EMOJI.get(cls, "")
                    conf  = det["confidence"]
                    bbox  = det["bbox"]
                    col_color = "#3fb950" if cls == "bird" else "#f85149"
                    st.markdown(f"""
                    <div style='background:#111827; border:1px solid {col_color}33;
                                border-left: 3px solid {col_color};
                                border-radius: 8px; padding: 10px 16px; margin: 6px 0;
                                display:flex; align-items:center; gap:16px;'>
                      <span style='font-size:1.4rem;'>{emoji}</span>
                      <span style='color:{col_color}; font-weight:700; min-width:60px;'>{cls.upper()}</span>
                      <span style='font-family:Space Mono; color:#93c5fd;'>{conf:.1%}</span>
                      <span style='color:#475569; font-size:0.8rem; margin-left:auto;'>
                        [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Summary metrics
                n_birds  = sum(1 for d in detections if d["class_name"] == "bird")
                n_drones = sum(1 for d in detections if d["class_name"] == "drone")
                c1, c2, c3 = st.columns(3)
                for col, label, val in [
                    (c1, "Total Detected", len(detections)),
                    (c2, "Birds 🐦",        n_birds),
                    (c3, "Drones 🚁",       n_drones),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class='metric-card'>
                          <div class='label'>{label}</div>
                          <div class='value'>{val}</div>
                        </div>
                        """, unsafe_allow_html=True)


# ─── Page: Dashboard ──────────────────────────────────────────────────────────

elif "Dashboard" in page:
    st.markdown("<div class='hero-title'>Model Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Performance metrics and training statistics across all trained models.</div>", unsafe_allow_html=True)
    st.divider()

    df = load_comparison_csv()
    if df is not None:
        import pandas as pd

        # Highlight best per column
        st.markdown("<div class='section-title'>Performance Comparison</div>", unsafe_allow_html=True)
        st.dataframe(
            df.style.highlight_max(axis=0, color="#1e3a5f")
                    .format("{:.4f}"),
            use_container_width=True,
        )

        # Bar chart
        st.markdown("<div class='section-title'>Metric Visualization</div>", unsafe_allow_html=True)
        metric = st.selectbox("Select metric", df.columns.tolist())

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#111827")

        colors = ["#38bdf8", "#3fb950", "#d29922", "#f85149", "#bc8cff"]
        bars = ax.barh(df.index, df[metric], color=colors[:len(df)], height=0.55)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color="white", fontsize=10,
                    fontfamily="monospace")

        ax.set_xlim(0, 1.1)
        ax.set_xlabel(metric, color="#8b949e")
        ax.set_title(f"Model Comparison — {metric}", color="white", fontsize=13)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.xaxis.grid(True, color="#21262d", lw=0.8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    else:
        st.info("No comparison data found. Run `scripts/run_training.py` to generate metrics.")

    # Training curves
    st.markdown("<div class='section-title'>Training Curves</div>", unsafe_allow_html=True)
    history_files = glob.glob(os.path.join(HISTORY_DIR, "*_history.json"))
    if history_files:
        hist_names = {Path(f).stem.replace("_history", ""): f for f in history_files}
        selected_hist = st.selectbox("Select model history", list(hist_names.keys()))
        with open(hist_names[selected_hist]) as f:
            hist = json.load(f)

        epochs = list(range(1, len(hist.get("accuracy", hist.get("loss", []))) + 1))

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        fig2.patch.set_facecolor("#0d1117")

        for ax in (ax1, ax2):
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#8b949e")
            ax.spines[:].set_color("#30363d")

        if "accuracy" in hist:
            ax1.plot(epochs, hist["accuracy"],     color="#38bdf8", lw=2, label="Train")
            ax1.plot(epochs, hist["val_accuracy"], color="#3fb950", lw=2, label="Val", ls="--")
            ax1.set_title("Accuracy", color="white")
            ax1.legend(facecolor="#21262d", labelcolor="white")

        if "loss" in hist:
            ax2.plot(epochs, hist["loss"],     color="#f85149", lw=2, label="Train")
            ax2.plot(epochs, hist["val_loss"], color="#d29922", lw=2, label="Val", ls="--")
            ax2.set_title("Loss", color="white")
            ax2.legend(facecolor="#21262d", labelcolor="white")

        for ax in (ax1, ax2):
            ax.set_xlabel("Epoch", color="#8b949e")
            ax.yaxis.grid(True, color="#21262d", lw=0.8)

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
    else:
        st.caption("No training history files found yet.")


# ─── Page: About ──────────────────────────────────────────────────────────────

elif "About" in page:
    st.markdown("<div class='hero-title'>About This Project</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div style='background:linear-gradient(135deg,#111827,#1a2236);
                border:1px solid #1e3a5f; border-radius:16px; padding:28px 32px;
                line-height:1.8;'>

    <p style='color:#93c5fd; font-size:1.05rem; font-weight:600;'>🎯 Problem Statement</p>
    <p>This project develops a deep learning solution to <strong>classify aerial images</strong> as
    <strong>Bird</strong> or <strong>Drone</strong> and optionally <strong>detect and localize</strong>
    objects in real-world scenes using YOLOv8.</p>

    <p style='color:#93c5fd; font-size:1.05rem; font-weight:600; margin-top:20px;'>🛠 Models Trained</p>
    <ul>
      <li><strong>Custom CNN</strong> — Built from scratch with Conv, BN, Dropout layers</li>
      <li><strong>ResNet50</strong> — Deep residual network, ImageNet pretrained</li>
      <li><strong>MobileNetV2</strong> — Lightweight, optimised for speed</li>
      <li><strong>EfficientNetB0</strong> — Best accuracy / parameter efficiency tradeoff</li>
      <li><strong>YOLOv8n</strong> — Real-time object detection with bounding boxes</li>
    </ul>

    <p style='color:#93c5fd; font-size:1.05rem; font-weight:600; margin-top:20px;'>📦 Dataset</p>
    <ul>
      <li>Classification: 2,662 train / 442 val / 215 test (Bird & Drone JPEGs)</li>
      <li>Detection: 3,319 images with YOLOv8-format bounding box annotations</li>
    </ul>

    <p style='color:#93c5fd; font-size:1.05rem; font-weight:600; margin-top:20px;'>🏗 Tech Stack</p>
    <p>
      <span class='info-pill'>Python 3.10+</span>
      <span class='info-pill'>TensorFlow / Keras</span>
      <span class='info-pill'>Ultralytics YOLOv8</span>
      <span class='info-pill'>OpenCV</span>
      <span class='info-pill'>scikit-learn</span>
      <span class='info-pill'>Streamlit</span>
      <span class='info-pill'>Matplotlib</span>
    </p>

    <p style='color:#93c5fd; font-size:1.05rem; font-weight:600; margin-top:20px;'>🌐 Use Cases</p>
    <ul>
      <li>Airport bird-strike prevention</li>
      <li>Restricted airspace drone monitoring</li>
      <li>Wildlife population tracking</li>
      <li>Security & defence surveillance</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)
