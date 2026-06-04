# Aerial Object Classification

This repository contains an aerial image classification and detection project built with TensorFlow, YOLOv8, and Streamlit.

The main application code lives in [aerial_project/](./aerial_project/). That folder includes:

- training and evaluation code in `src/`
- an end-to-end runner in `scripts/run_training.py`
- a Streamlit UI in `streamlit_app/app.py`
- saved plots and JSON training history in `models/`

Quick start:

```bash
cd aerial_project
pip install -r requirements.txt
python scripts/run_training.py
streamlit run streamlit_app/app.py
```

Repository notes:

- virtual environments, logs, and model weight files are ignored in Git
- training paths are configurable through environment variables
- the detailed project README is in [aerial_project/README.md](./aerial_project/README.md)
