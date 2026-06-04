"""
train.py
────────
Training pipeline for Custom CNN and Transfer Learning models.

Features:
  - EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
  - Class weight support for imbalanced datasets
  - Two-phase training for transfer learning (freeze → fine-tune)
  - Training history plotting
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

from src.custom_cnn       import build_custom_cnn
from src.transfer_learning import build_transfer_model, unfreeze_and_finetune
from src.preprocess        import get_data_generators, compute_class_weights


# ─── Default hyperparameters ──────────────────────────────────────────────────
DEFAULT_CFG = {
    "epochs_phase1": 30,      # frozen backbone / CNN full training
    "epochs_phase2": 20,      # fine-tuning unfrozen layers
    "batch_size"   : 32,
    "cnn_lr"       : 1e-3,
    "tl_lr_phase1" : 1e-3,
    "tl_lr_phase2" : 1e-5,
    "patience"     : 7,       # early stopping patience
    "models_dir"   : "models",
    "logs_dir"     : "logs",
}


# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(
    model_name: str,
    models_dir: str,
    logs_dir: str,
    patience: int = 7,
) -> list:
    """
    Return a list of Keras callbacks for training.

    Includes:
      - EarlyStopping        (monitors val_accuracy)
      - ModelCheckpoint      (saves best model as .keras file)
      - ReduceLROnPlateau    (halves LR on val_loss plateau)
      - TensorBoard          (optional; logs to logs_dir)
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    ckpt_path = os.path.join(models_dir, f"{model_name}_best.keras")

    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(logs_dir, model_name),
            histogram_freq=1,
        ),
    ]


# ─── Train Custom CNN ─────────────────────────────────────────────────────────

def train_custom_cnn(
    dataset_root: str,
    cfg: dict = None,
    use_class_weights: bool = True,
) -> tuple:
    """
    Train the custom CNN model from scratch.

    Returns:
        (model, history, elapsed_seconds)
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    print("\n" + "═" * 60)
    print("  🧠  Training Custom CNN")
    print("═" * 60)

    train_gen, valid_gen, _ = get_data_generators(dataset_root, cfg["batch_size"])
    class_weights = compute_class_weights(train_gen) if use_class_weights else None

    model = build_custom_cnn(learning_rate=cfg["cnn_lr"])
    cbs   = get_callbacks("custom_cnn", cfg["models_dir"], cfg["logs_dir"], cfg["patience"])

    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=cfg["epochs_phase1"],
        callbacks=cbs,
        class_weight=class_weights,
        verbose=1,
    )
    elapsed = time.time() - t0

    _save_history(history.history, "custom_cnn", cfg["models_dir"])
    print(f"\n  ⏱  Training time: {elapsed/60:.1f} min")
    return model, history, elapsed


# ─── Train Transfer Learning (two-phase) ─────────────────────────────────────

def train_transfer_model(
    dataset_root: str,
    backbone: str  = "efficientnetb0",
    cfg: dict      = None,
    use_class_weights: bool = True,
) -> tuple:
    """
    Two-phase training for a transfer learning model.

    Phase 1: Train head only (backbone frozen).
    Phase 2: Fine-tune last 30 backbone layers.

    Returns:
        (model, history_phase1, history_phase2, elapsed_seconds)
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    name = f"tl_{backbone}"
    print("\n" + "═" * 60)
    print(f"  🔁  Training Transfer Model: {backbone.upper()}")
    print("═" * 60)

    train_gen, valid_gen, _ = get_data_generators(dataset_root, cfg["batch_size"])
    class_weights = compute_class_weights(train_gen) if use_class_weights else None

    # ── Phase 1: Frozen backbone ─────────────────────────────────────────────
    print("\n📌 Phase 1 — Training head (backbone frozen)")
    model = build_transfer_model(backbone, learning_rate=cfg["tl_lr_phase1"], freeze_base=True)
    cbs1  = get_callbacks(f"{name}_p1", cfg["models_dir"], cfg["logs_dir"], cfg["patience"])

    t0 = time.time()
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=cfg["epochs_phase1"],
        callbacks=cbs1,
        class_weight=class_weights,
        verbose=1,
    )

    # ── Phase 2: Fine-tuning ──────────────────────────────────────────────────
    print("\n📌 Phase 2 — Fine-tuning unfrozen layers")
    model = unfreeze_and_finetune(model, fine_tune_lr=cfg["tl_lr_phase2"])
    cbs2  = get_callbacks(f"{name}_p2", cfg["models_dir"], cfg["logs_dir"], cfg["patience"])

    history2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=cfg["epochs_phase2"],
        callbacks=cbs2,
        class_weight=class_weights,
        verbose=1,
    )

    elapsed = time.time() - t0
    _save_history(history1.history, f"{name}_p1", cfg["models_dir"])
    _save_history(history2.history, f"{name}_p2", cfg["models_dir"])

    # Save final fine-tuned model
    final_path = os.path.join(cfg["models_dir"], f"{name}_finetuned.keras")
    model.save(final_path)
    print(f"\n  💾 Saved fine-tuned model → {final_path}")
    print(f"  ⏱  Total training time: {elapsed/60:.1f} min")
    return model, history1, history2, elapsed


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(history, model_name: str = "Model", save_path: str = None):
    """
    Plot training & validation accuracy and loss curves side by side.
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    # Accuracy
    ax1.plot(epochs, hist["accuracy"],     color="#58a6ff", lw=2, label="Train Accuracy")
    ax1.plot(epochs, hist["val_accuracy"], color="#3fb950", lw=2, label="Val Accuracy", ls="--")
    ax1.set_title(f"{model_name} — Accuracy", color="white", fontsize=13)
    ax1.set_xlabel("Epoch", color="#8b949e")
    ax1.set_ylabel("Accuracy", color="#8b949e")
    ax1.legend(facecolor="#21262d", labelcolor="white")

    # Loss
    ax2.plot(epochs, hist["loss"],     color="#f85149", lw=2, label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], color="#d29922", lw=2, label="Val Loss", ls="--")
    ax2.set_title(f"{model_name} — Loss", color="white", fontsize=13)
    ax2.set_xlabel("Epoch", color="#8b949e")
    ax2.set_ylabel("Loss", color="#8b949e")
    ax2.legend(facecolor="#21262d", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  ✅ Saved training plot → {save_path}")
    plt.show()


def plot_combined_histories(h1, h2, name1="Phase 1", name2="Phase 2",
                            model_name="Model", save_path=None):
    """
    Plot two training histories (e.g., Phase 1 + Phase 2) on the same axes.
    """
    def _get(h):
        return h.history if hasattr(h, "history") else h

    h1, h2 = _get(h1), _get(h2)
    e1 = list(range(1, len(h1["accuracy"]) + 1))
    e2 = list(range(e1[-1] + 1, e1[-1] + 1 + len(h2["accuracy"])))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.axvline(e1[-1], color="#30363d", ls=":", lw=1.5)

    for ax, key in zip(axes, ["accuracy", "loss"]):
        ax.plot(e1, h1[key],     color="#58a6ff", lw=2, label=f"{name1} Train")
        ax.plot(e1, h1[f"val_{key}"], color="#3fb950", lw=2, label=f"{name1} Val", ls="--")
        ax.plot(e2, h2[key],     color="#58a6ff", lw=2, alpha=0.6)
        ax.plot(e2, h2[f"val_{key}"], color="#3fb950", lw=2, alpha=0.6, ls="--")
        ax.set_title(f"{model_name} — {key.capitalize()}", color="white", fontsize=13)
        ax.set_xlabel("Epoch", color="#8b949e")
        ax.set_ylabel(key.capitalize(), color="#8b949e")
        ax.legend(facecolor="#21262d", labelcolor="white")
        ax.text((e1[-1] + e2[0]) / 2, ax.get_ylim()[1] * 0.95,
                "Fine-tune →", color="#d29922", fontsize=8, ha="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_history(hist: dict, name: str, models_dir: str):
    """Persist training history as JSON for later analysis."""
    path = os.path.join(models_dir, f"{name}_history.json")
    # Convert numpy floats to Python floats
    hist_serializable = {k: [float(v) for v in vals] for k, vals in hist.items()}
    with open(path, "w") as f:
        json.dump(hist_serializable, f, indent=2)
    print(f"  💾 History saved → {path}")


def load_history(name: str, models_dir: str = "models") -> dict:
    """Load a previously saved training history."""
    path = os.path.join(models_dir, f"{name}_history.json")
    with open(path) as f:
        return json.load(f)
