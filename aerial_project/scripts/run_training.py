"""

End-to-end training script for the Aerial Object Classification project.

"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # add project root

import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if os.getenv("AERIAL_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.utils             import set_seeds, print_hardware_info
from src.preprocess        import get_data_generators, check_class_balance, visualize_samples
from src.train             import (train_custom_cnn, train_transfer_model,
                                   plot_history, plot_combined_histories, DEFAULT_CFG)
from src.evaluate          import evaluate_model, compare_models
from src.yolo_pipeline     import create_data_yaml, train_yolo, validate_yolo



#  ⚙️  CONFIG — Edit these paths before running

CLASSIFICATION_DATASET = os.getenv(
    "AERIAL_CLASSIFICATION_DATASET",
    str(ROOT / "data" / "classification_dataset"),
)
DETECTION_DATASET = os.getenv(
    "AERIAL_DETECTION_DATASET",
    str(ROOT / "data" / "detection_dataset"),
)
MODELS_DIR = os.getenv(
    "AERIAL_MODELS_DIR",
    str(ROOT / "models"),
)
RUN_YOLO = os.getenv("AERIAL_RUN_YOLO", "false").lower() in {"1", "true", "yes"}

TRAIN_CFG = {
    **DEFAULT_CFG,
    "epochs_phase1"  : 10,
    "epochs_phase2"  : 8,
    "batch_size"     : 32,
    "models_dir"     : MODELS_DIR,
    "logs_dir"       : os.path.join(MODELS_DIR, "logs"),
}

BACKBONES = ["efficientnetb0"]



def main():
    set_seeds(42)
    print_hardware_info()

    # Ensure model output directory exists before saving plots or weights
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Sanity check dataset ─────────────────────────────────────────────────
    if not Path(CLASSIFICATION_DATASET).exists():
        print(f"\n❌ Dataset not found: {CLASSIFICATION_DATASET}")
        print("   Set AERIAL_CLASSIFICATION_DATASET or place the dataset under data/classification_dataset")
        return

    check_class_balance(CLASSIFICATION_DATASET)
    visualize_samples(CLASSIFICATION_DATASET,
                      save_path=os.path.join(MODELS_DIR, "sample_images.png"))

    # ── Get data generators (shared across models) ───────────────────────────
    train_gen, valid_gen, test_gen = get_data_generators(
        CLASSIFICATION_DATASET, TRAIN_CFG["batch_size"]
    )

    all_results = []

    
    #  1. Custom CNN
    
    cnn_model, cnn_history, cnn_time = train_custom_cnn(
        CLASSIFICATION_DATASET, cfg=TRAIN_CFG
    )
    plot_history(cnn_history, model_name="Custom CNN",
                 save_path=os.path.join(MODELS_DIR, "custom_cnn_training.png"))

    cnn_results = evaluate_model(
        cnn_model, test_gen,
        model_name="CustomCNN",
        save_dir=MODELS_DIR,
    )
    cnn_results["train_time_min"] = round(cnn_time / 60, 1)
    all_results.append(cnn_results)

    
    #  2. Transfer Learning models
    
    for backbone in BACKBONES:
        tl_model, h1, h2, tl_time = train_transfer_model(
            CLASSIFICATION_DATASET,
            backbone=backbone,
            cfg=TRAIN_CFG,
        )
        plot_combined_histories(
            h1, h2,
            model_name=f"TL_{backbone.upper()}",
            save_path=os.path.join(MODELS_DIR, f"tl_{backbone}_training.png"),
        )
        tl_results = evaluate_model(
            tl_model, test_gen,
            model_name=f"TL_{backbone.upper()}",
            save_dir=MODELS_DIR,
        )
        tl_results["train_time_min"] = round(tl_time / 60, 1)
        all_results.append(tl_results)

    
    #  3. Model Comparison

    df, best_model = compare_models(all_results, save_dir=MODELS_DIR)

    
    #   YOLOv8 (Optional)
    if RUN_YOLO:
        if not Path(DETECTION_DATASET).exists():
            print(f"\n⚠️  Detection dataset not found: {DETECTION_DATASET} — skipping YOLO")
        else:
            yaml_save_path = ROOT / "configs" / "data.yaml"
            yaml_path = create_data_yaml(DETECTION_DATASET, save_path=str(yaml_save_path))
            _, best_weights = train_yolo(
                data_yaml = yaml_path,
                model_size= "yolov8n",
                epochs    = 50,
                project   = MODELS_DIR,
                name      = "yolo_aerial",
            )
            validate_yolo(best_weights, yaml_path, split="test")
            print(f"\n🎯 YOLO best weights → {best_weights}")

    print("\n" + "═" * 60)
    print("  🏁  All training complete!")
    print(f"  🏆  Best classification model: {best_model}")
    print(f"  📁  Models saved in: {MODELS_DIR}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
