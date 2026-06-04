"""
custom_cnn.py
─────────────
Custom Convolutional Neural Network for Bird vs Drone binary classification.

Architecture:
  Input (224×224×3)
  → [Conv2D → BN → ReLU → MaxPool] × 4 blocks (increasing filters)
  → GlobalAveragePooling
  → Dense(512) + BN + Dropout(0.5)
  → Dense(256) + BN + Dropout(0.3)
  → Dense(1, sigmoid)       ← binary output
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam


# ─── Architecture ─────────────────────────────────────────────────────────────

def build_custom_cnn(
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.5,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build and compile a custom CNN for binary image classification.

    Args:
        input_shape:    Image dimensions (H, W, C).
        dropout_rate:   Dropout fraction for first dense layer.
        l2_reg:         L2 regularization weight for Conv & Dense layers.
        learning_rate:  Adam optimizer learning rate.

    Returns:
        Compiled Keras model.
    """
    reg = regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape, name="input_image")

    # ── Block 1: 32 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv1_1")(inp)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # ── Block 2: 64 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # ── Block 3: 128 filters ─────────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.SpatialDropout2D(0.2, name="spatial_drop3")(x)

    # ── Block 4: 256 filters ─────────────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)
    x = layers.SpatialDropout2D(0.3, name="spatial_drop4")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(512, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_d1")(x)
    x = layers.Activation("relu", name="relu_d1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(256, kernel_regularizer=reg, name="dense2")(x)
    x = layers.BatchNormalization(name="bn_d2")(x)
    x = layers.Activation("relu", name="relu_d2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CustomCNN")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )

    return model


def get_cnn_summary(model: tf.keras.Model):
    """Print a clean model summary."""
    print("\n🧠 Custom CNN Architecture")
    print("─" * 60)
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters : {total_params:,}")
    print(f"  Trainable params : {sum(w.numpy().size for w in model.trainable_weights):,}")
    print("─" * 60)


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_custom_cnn()
    get_cnn_summary(model)

    # Dummy forward pass
    dummy = tf.random.normal((2, 224, 224, 3))
    out = model(dummy, training=False)
    print(f"\n✅ Forward pass OK — output shape: {out.shape}  values: {out.numpy().flatten()}")
