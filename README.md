# 📈 LightGBM Model Trainer & Predictor

This project provides two main Python scripts to train a LightGBM model using historical sports data and make predictions based on new daily inputs.

- `trainer.py`: trains the model using historical Excel data and performs cross-validation.
- `predict.py`: applies a trained model to daily data and outputs predictions.

---

## ⚙️ Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) for virtual environment and dependency management
- PowerShell (Windows) or your preferred shell for environment activation

### 🔧 Setting Up

1. Install `uv` if you haven't:
    ```bash
    pip install uv
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
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
- Trained model (.pkl) saved under models/

- Best hyperparameters saved to data/best_params.json

- Console output showing cross-validation progress and test accuracy

### ▶️ Usage
```bash
python -m src.scripts.trainer \
--input data/training/aHistorical_01012024_02142025.xlsx \
--output history.pkl
```

**This will:**
- Load and preprocess the historical data.

- Run GridSearchCV on an LGBMClassifier (using only percent_over).

- Save the best model to models/history.pkl.

- Save the grid search’s best parameters to data/best_params.json.

- Automatically evaluate the model on all .xlsx files in data/tests/.

---

## 🔮 Making Predictions
The prediction script loads a trained model and applies it to a new daily Excel file.

### 🔢 Input
- `--input`: Daily Excel file (e.g., Daily__02152025.xlsx)

- `--model`: Path to the trained .pkl model file

- `--history`: Historical Excel file to compute feature lookup (same as used for training)

### 💾 Output
- **CSV file with predictions saved to `data/predictions/`**

- Columns in the output CSV:
    - player
    - planets_intensity
    - predicted_result (OVER or UNDER)

### ▶️ Usage
```bash
python -m src.scripts.predict \
  --input tests/Daily__02152025.xlsx \
  --model models/history.pkl \
  --history data/training/aHistorical_01012024_02142025.xlsx
```
**This will:**

- Load the trained model.

- Load daily and historical data, extract/merge percent_over.

- Predict each row as OVER (1) or UNDER (0).

- Save results to data/predictions/Daily__02152025_predictions.csv.

---

## 📁 Project Structure


```plaintext
+---data
|   |   best_params.json
|   |
|   +---predictions
|   |       Daily__02152025_predictions.csv
|   |
|   +---tests
|   |       Daily__02152025.xlsx
|   |       Daily__02162025.xlsx
|   |       Daily__02172025.xlsx
|   |       Daily__02182025.xlsx
|   |       Daily__02192025.xlsx
|   |       Daily__02202025.xlsx
|   |       Daily__02212025.xlsx
|   |       Daily__02222025.xlsx
|   |       Daily__02232025.xlsx
|   |       Daily__02242025.xlsx
|   |
|   +---training
|           aHistorical_01012024_02142025.xlsx
|
+---models
|       history.pkl
|
+---src
|   |   __init__.py
|   |
|   +---notebooks
|   |       visualization.ipynb
|   |
|   +---scripts
|   |       predict.py
|   |       trainer.py
|   |    
|   +---utils
|           load.py
|           processing.py
|
|--- .gitignore
|--- .python-version
|--- pyproject.toml
|--- README.ms
|--- uv.lock
```

---

## ✅ Notes
- Ensure `models/` and `data/predictions/` directories exist before running the scripts.

- The model currently uses only a single feature (percent_over). You can extend prepare_training_data and extract_features to add more.

- Adjust PARAM_GRID in trainer.py to explore additional hyperparameter combinations.