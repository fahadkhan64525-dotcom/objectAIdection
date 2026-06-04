"""
evaluate.py
───────────
Model evaluation, metrics computation, and comparison utilities.

Outputs:
  - Classification report (Precision, Recall, F1, Accuracy)
  - Confusion matrix heatmap
  - ROC curve
  - Model comparison table & bar chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


CLASSES = ["bird", "drone"]


# ─── Core Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    model: tf.keras.Model,
    test_gen,
    model_name: str  = "Model",
    threshold: float = 0.5,
    save_dir: str    = "models",
) -> dict:
    """
    Run full evaluation on the test generator.

    Returns:
        dict with keys: accuracy, precision, recall, f1, auc_score,
                        y_true, y_pred, y_prob
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📊 Evaluating: {model_name}")
    print("─" * 50)

    # ── Predictions ──────────────────────────────────────────────────────────
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = test_gen.classes

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score   = auc(fpr, tpr)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc_score:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES))

    results = dict(
        model_name=model_name,
        accuracy=acc, precision=prec, recall=rec,
        f1=f1, auc=auc_score,
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        fpr=fpr, tpr=tpr,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, model_name,
                          save_path=os.path.join(save_dir, f"{model_name}_cm.png"))
    plot_roc_curve(fpr, tpr, auc_score, model_name,
                   save_path=os.path.join(save_dir, f"{model_name}_roc.png"))

    return results


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred,
    model_name: str = "Model",
    save_path: str  = None,
):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm, annot=False, fmt="d", ax=ax,
        cmap="Blues", linewidths=1.5, linecolor="#30363d",
        cbar_kws={"shrink": 0.8},
    )
    # Annotate with count + percentage
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    ax.set_xticklabels(CLASSES, color="#8b949e", fontsize=11)
    ax.set_yticklabels(CLASSES, color="#8b949e", fontsize=11, rotation=0)
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#8b949e")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  ✅ Confusion matrix saved → {save_path}")
    plt.show()


def plot_roc_curve(
    fpr, tpr, auc_score: float,
    model_name: str = "Model",
    save_path: str  = None,
):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(fpr, tpr, color="#58a6ff", lw=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#30363d", lw=1.5, ls="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#58a6ff")

    ax.set_xlabel("False Positive Rate", color="#8b949e", fontsize=12)
    ax.set_ylabel("True Positive Rate",  color="#8b949e", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", color="white", fontsize=14)
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=11)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  ✅ ROC curve saved → {save_path}")
    plt.show()


# ─── Model Comparison ─────────────────────────────────────────────────────────

def compare_models(results_list: list, save_dir: str = "models"):
    """
    Build a comparison table and bar chart across all trained models.

    Args:
        results_list: List of dicts returned by evaluate_model().
        save_dir:     Where to save the comparison chart.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── DataFrame ────────────────────────────────────────────────────────────
    rows = []
    for r in results_list:
        rows.append({
            "Model"    : r["model_name"],
            "Accuracy" : round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall"   : round(r["recall"],    4),
            "F1-Score" : round(r["f1"],        4),
            "AUC-ROC"  : round(r["auc"],       4),
        })
    df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "═" * 60)
    print("  📊  Model Comparison Report")
    print("═" * 60)
    print(df.to_string())
    print("═" * 60)

    # Save CSV
    csv_path = os.path.join(save_dir, "model_comparison.csv")
    df.to_csv(csv_path)
    print(f"\n  💾 Comparison CSV → {csv_path}")

    # ── Bar Chart ─────────────────────────────────────────────────────────────
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    n_models  = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i, (model_name, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2,
                      vals, width * 0.9,
                      label=model_name,
                      color=colors[i % len(colors)],
                      alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="#8b949e", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color="#8b949e", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", color="white", fontsize=15, pad=15)
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=10)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")
    ax.yaxis.grid(True, color="#21262d", lw=0.8)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✅ Comparison chart → {chart_path}")
    plt.show()

    # ── Best model ───────────────────────────────────────────────────────────
    best = df["F1-Score"].idxmax()
    print(f"\n  🏆  Best Model (by F1): {best}  →  F1 = {df.loc[best, 'F1-Score']:.4f}")
    return df, best


# ─── Single image prediction ──────────────────────────────────────────────────

def predict_single_image(
    model: tf.keras.Model,
    img_array: np.ndarray,
    threshold: float = 0.5,
) -> tuple:
    """
    Predict class for a single preprocessed image (H, W, 3) or (1, H, W, 3).

    Returns:
        (label: str, confidence: float)
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, 0)
    prob = model.predict(img_array, verbose=0)[0][0]
    label = CLASSES[int(prob >= threshold)]
    confidence = prob if prob >= threshold else 1 - prob
    return label, float(confidence)
