# 📈 LightGBM Model Trainer & Predictor

This project provides two main Python scripts to train a LightGBM model using historical sports data and make predictions based on new daily inputs.

- `trainer.py`: trains the model using historical Excel data and performs cross-validation.
- `predict.py`: applies a trained model to daily data and outputs predictions.

---

## ⚙️ Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) for virtual environment and dependency management
- PowerShell (Windows) or your preferred shell for environment activation

### 🔧 Setting Up

1. Install `uv` if you haven't:
    ```bash
    pip install uv
    ```

2. Initialize UV project
    ```bash
    uv sync
    ```

3. Activate the environment
    ```bash
    . .venv/Scripts/Activate.ps1
    ```
--- 

## 🏋️‍♂️ Training the Model
The training script loads historical Excel data, extracts features, tunes hyperparameters with cross-validation, and saves the best model.

### 🔢 Input
- Historical Excel file (e.g., `aHistorical_01012024_02142025.xlsx`) containing:

- A percent_over feature column

- A label column with binary targets (0/1)

### 💾 Output
Trained model (.pkl) saved under models/

Best hyperparameters saved to data/best_params.json

Console output showing cross-validation progress and test accuracy

--- 

## ▶️ Usage
