"""
preprocess.py
─────────────
Data loading, normalization, and augmentation pipeline
for the Aerial Object Classification project.

Dataset structure expected:
    dataset/
        TRAIN/bird/, TRAIN/drone/
        VALID/bird/, VALID/drone/
        TEST/bird/,  TEST/drone/
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
CLASSES      = ["bird", "drone"]
NUM_CLASSES  = len(CLASSES)


# ─── Data Generators ──────────────────────────────────────────────────────────

def get_data_generators(dataset_root: str, batch_size: int = BATCH_SIZE):
    """
    Build train / validation / test generators with augmentation on train set.

    Args:
        dataset_root: Root folder containing TRAIN/, VALID/, TEST/ sub-folders.
        batch_size:   Mini-batch size (default 32).

    Returns:
        train_gen, valid_gen, test_gen  (Keras DirectoryIterator objects)
    """
    dataset_root = Path(dataset_root)

    # ── Train: heavy augmentation ──────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest",
    )

    # ── Validation & Test: only rescale ───────────────────────────────────
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        dataset_root / "TRAIN",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )

    valid_gen = val_test_datagen.flow_from_directory(
        dataset_root / "VALID",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        dataset_root / "TEST",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    _print_generator_info(train_gen, valid_gen, test_gen)
    return train_gen, valid_gen, test_gen


def _print_generator_info(train_gen, valid_gen, test_gen):
    print("\n📦 Dataset Summary")
    print("─" * 40)
    print(f"  Classes   : {train_gen.class_indices}")
    print(f"  Train     : {train_gen.samples} images")
    print(f"  Validation: {valid_gen.samples} images")
    print(f"  Test      : {test_gen.samples} images")
    print(f"  Batch size: {train_gen.batch_size}")
    print(f"  Image size: {IMG_SIZE}")
    print("─" * 40)


# ─── Class Imbalance Check ────────────────────────────────────────────────────

def check_class_balance(dataset_root: str):
    """
    Print & return per-class image counts for each split.
    """
    dataset_root = Path(dataset_root)
    splits = ["TRAIN", "VALID", "TEST"]
    report = {}

    print("\n📊 Class Distribution")
    print("─" * 40)
    for split in splits:
        split_counts = {}
        for cls in CLASSES:
            folder = dataset_root / split / cls
            if folder.exists():
                count = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
                split_counts[cls] = count
        report[split] = split_counts
        print(f"  {split:6s}: {split_counts}")
    print("─" * 40)
    return report


def compute_class_weights(train_gen):
    """
    Compute class weights to handle mild class imbalance.
    Returns a dict {0: w0, 1: w1} for use in model.fit().
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))
    print(f"\n⚖️  Class Weights: {class_weights}")
    return class_weights


# ─── Visualizations ───────────────────────────────────────────────────────────

def visualize_samples(dataset_root: str, n: int = 12, save_path: str = None):
    """
    Display a grid of sample images from each class in the TRAIN split.
    """
    dataset_root = Path(dataset_root)
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(2, n // 2, figure=fig, hspace=0.35, wspace=0.1)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(n // 2)]

    idx = 0
    for cls in CLASSES:
        folder = dataset_root / "TRAIN" / cls
        images = sorted(folder.glob("*.jpg"))[:n // 2]
        for img_path in images:
            img = plt.imread(str(img_path))
            axes[idx].imshow(img)
            axes[idx].set_title(cls.upper(), color="#58a6ff", fontsize=9, pad=4)
            axes[idx].axis("off")
            idx += 1

    fig.suptitle("Sample Aerial Images — Bird vs Drone", color="white",
                 fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        print(f"  ✅ Saved sample grid → {save_path}")
    plt.show()


def visualize_augmentations(dataset_root: str, save_path: str = None):
    """
    Show one image before and after various augmentations.
    """
    dataset_root = Path(dataset_root)
    sample_path = next((dataset_root / "TRAIN" / "bird").glob("*.jpg"))
    img = tf.keras.preprocessing.image.load_img(sample_path, target_size=IMG_SIZE)
    img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, 0)

    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.6, 1.4],
        fill_mode="nearest",
    )

    titles = ["Original"] + [f"Aug #{i}" for i in range(1, 8)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor("#0d1117")
    axes = axes.flatten()

    axes[0].imshow(img_arr[0])
    axes[0].set_title("Original", color="#58a6ff", fontsize=10)
    axes[0].axis("off")

    gen = aug.flow(img_arr, batch_size=1)
    for i in range(1, 8):
        aug_img = next(gen)[0]
        axes[i].imshow(np.clip(aug_img, 0, 1))
        axes[i].set_title(f"Aug #{i}", color="#8b949e", fontsize=10)
        axes[i].axis("off")

    fig.suptitle("Data Augmentation Examples", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        print(f"  ✅ Saved augmentation grid → {save_path}")
    plt.show()


# ─── TF Dataset API alternative (for large datasets) ─────────────────────────

def build_tf_dataset(dataset_root: str, batch_size: int = BATCH_SIZE):
    """
    Alternative: Build tf.data.Dataset pipelines for train/valid/test splits.
    Useful for larger datasets or custom training loops.
    """
    dataset_root = Path(dataset_root)
    AUTOTUNE = tf.data.AUTOTUNE

    def _load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    def _augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    def _build_split(split_name, augment=False):
        paths, labels = [], []
        for label_idx, cls in enumerate(CLASSES):
            folder = dataset_root / split_name / cls
            for p in folder.glob("*.jpg"):
                paths.append(str(p))
                labels.append(float(label_idx))

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(len(paths), seed=42) if augment else ds
        ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds, len(paths)

    train_ds, n_train = _build_split("TRAIN", augment=True)
    valid_ds, n_valid = _build_split("VALID", augment=False)
    test_ds,  n_test  = _build_split("TEST",  augment=False)

    print(f"\n🔥 tf.data.Dataset built: {n_train} train | {n_valid} valid | {n_test} test")
    return train_ds, valid_ds, test_ds
