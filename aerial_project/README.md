# Aerial Object Classification and Detection

Classify aerial images as `bird` or `drone` with TensorFlow models, and optionally run YOLOv8 object detection through the same project.

## Project Structure

```text
aerial_project/
|-- configs/
|   `-- data.yaml
|-- notebooks/
|   `-- aerial_eda.ipynb
|-- scripts/
|   `-- run_training.py
|-- src/
|   |-- custom_cnn.py
|   |-- evaluate.py
|   |-- preprocess.py
|   |-- train.py
|   |-- transfer_learning.py
|   |-- utils.py
|   `-- yolo_pipeline.py
|-- streamlit_app/
|   `-- app.py
|-- MLcalculator.py
`-- requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

The training script uses repository-relative defaults and supports environment variable overrides.

Defaults:

- classification dataset: `aerial_project/data/classification_dataset`
- detection dataset: `aerial_project/data/detection_dataset`
- output models: `aerial_project/models`

Optional environment variables:

```bash
set AERIAL_CLASSIFICATION_DATASET=path\to\classification_dataset
set AERIAL_DETECTION_DATASET=path\to\detection_dataset
set AERIAL_MODELS_DIR=path\to\output\models
set AERIAL_RUN_YOLO=true
set AERIAL_FORCE_CPU=true
```

Run training:

```bash
python scripts/run_training.py
```

## Streamlit App

```bash
streamlit run streamlit_app/app.py
```

The app scans `models/` for saved `.keras` classifiers and looks for YOLO weights if they are available locally.

## Models Included

- Custom CNN
- EfficientNetB0 transfer learning
- YOLOv8 for optional object detection

## Metrics

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- training and validation curves

## Stack

Python, TensorFlow/Keras, Ultralytics YOLOv8, Streamlit, OpenCV, scikit-learn, matplotlib, and plotly.
