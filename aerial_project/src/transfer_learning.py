"""
transfer_learning.py
────────────────────
Transfer learning models for Bird vs Drone classification.

Supported backbones:
  - ResNet50
  - MobileNetV2
  - EfficientNetB0

Strategy:
  Phase 1 — Freeze backbone, train custom head only  (fast convergence)
  Phase 2 — Unfreeze last N layers, fine-tune at low LR (accuracy boost)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224, 3)
L2_REG      = 1e-4
SUPPORTED   = ["resnet50", "mobilenetv2", "efficientnetb0"]


# ─── Generic builder ──────────────────────────────────────────────────────────

def build_transfer_model(
    backbone_name: str   = "efficientnetb0",
    dropout_rate: float  = 0.4,
    dense_units: int     = 256,
    learning_rate: float = 1e-3,
    freeze_base: bool    = True,
) -> tf.keras.Model:
    """
    Build a binary classifier on top of a pretrained ImageNet backbone.

    Args:
        backbone_name:  One of 'resnet50', 'mobilenetv2', 'efficientnetb0'.
        dropout_rate:   Dropout applied before output layer.
        dense_units:    Units in the intermediate dense layer.
        learning_rate:  Initial Adam LR.
        freeze_base:    If True, freeze all backbone layers (Phase 1).

    Returns:
        Compiled Keras model ready for training.
    """
    backbone_name = backbone_name.lower()
    assert backbone_name in SUPPORTED, f"Backbone must be one of {SUPPORTED}"

    reg = regularizers.l2(L2_REG)

    # ── Load pretrained backbone ─────────────────────────────────────────────
    base = _load_backbone(backbone_name)
    base.trainable = not freeze_base

    # ── Build head ───────────────────────────────────────────────────────────
    inp = layers.Input(shape=IMG_SIZE, name="input_image")

    # EfficientNet & MobileNet include their own rescaling; ResNet needs it
    if backbone_name == "resnet50":
        x = tf.keras.applications.resnet50.preprocess_input(inp)
    elif backbone_name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    else:  # efficientnetb0
        x = tf.keras.applications.efficientnet.preprocess_input(inp)

    x = base(x, training=False)              # feature extraction
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)

    x = layers.Dense(dense_units, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_d1")(x)
    x = layers.Activation("relu", name="relu_d1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out,
                         name=f"TL_{backbone_name.upper()}")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )

    _print_model_info(model, backbone_name, freeze_base)
    return model


def _load_backbone(name: str):
    kwargs = dict(include_top=False, weights="imagenet", input_shape=IMG_SIZE)
    if name == "resnet50":
        return ResNet50(**kwargs)
    elif name == "mobilenetv2":
        return MobileNetV2(**kwargs)
    else:
        return EfficientNetB0(**kwargs)


def _print_model_info(model, name, frozen):
    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    print(f"\n🔁 Transfer Model: {name.upper()}")
    print(f"   Backbone frozen  : {frozen}")
    print(f"   Trainable layers : {trainable} / {total}")
    print(f"   Total params     : {model.count_params():,}")


# ─── Phase 2: Unfreeze for fine-tuning ────────────────────────────────────────

def unfreeze_and_finetune(
    model: tf.keras.Model,
    unfreeze_from: int   = -30,
    fine_tune_lr: float  = 1e-5,
) -> tf.keras.Model:
    """
    Unfreeze the last `unfreeze_from` layers of the backbone and recompile
    at a lower learning rate for fine-tuning (Phase 2).

    Args:
        model:          Model returned from build_transfer_model().
        unfreeze_from:  Negative int — how many tail layers to unfreeze.
        fine_tune_lr:   Fine-tuning learning rate (keep small to avoid forgetting).

    Returns:
        Recompiled model ready for Phase 2 training.
    """
    # Find the backbone sub-model (second layer after Input + preprocess)
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is None:
        print("⚠️  Could not locate backbone — skipping unfreeze")
        return model

    backbone.trainable = True
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False

    unfrozen = sum(1 for l in backbone.layers if l.trainable)
    print(f"\n🔓 Fine-tuning: {unfrozen} backbone layers unfrozen")

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name in SUPPORTED:
        m = build_transfer_model(name, freeze_base=True)
        dummy = tf.random.normal((2, 224, 224, 3))
        out = m(dummy, training=False)
        print(f"  ✅ {name}: output shape {out.shape}\n")
