# 📈 LightGBM Model Trainer & Predictor

Train a LightGBM model using historical sports data and make predictions based on new daily inputs. 

- `find_best_params.py`: By cross validation with multiple combinations, it finds the best parameters for optimizing accuracy.
- `train_model.py`: trains the model using historical Excel data.
- `test_model_performance.py`: Tests the accuracy of a trained model over a set of tests.
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
The training script loads historical Excel data, extracts features and trains the model.

### 🔢 Input
- `--input`: Historical Excel file (e.g., `data/training/aHistorical_01012024_02142025.xlsx`).

- `--output`: Name of the trained model.

- `--params` (Optional): Params json file path.

### 💾 Output
- Trained model (.pkl) saved under `models/`

### ▶️ Usage
```bash
python -m src.scripts.train_model --input data/training/aHistorical_01012024_02142025.xlsx --output history.pkl
```
*Optionally, you can add precomputed hyperparameters.*

```bash
python -m src.scripts.train_model --input data/training/aHistorical_01012024_02142025.xlsx --output history.pkl --params data/best_params.json
```
---

## 🔍 Find best parameters

Iterate through different combinations of hyperparameters, and see which combination results in the higest performance.

```bash
pyton -m src.scripts.find_best_params --input data/training/aHistorical_01012024_02142025.xlsx
```

---

## 📈 Testing performance
Test trained model performance on some tests.
```bash
python -m src.scripts.test_model_performance --model models/history.pkl --tests_folder data/tests
```
- **Output**
<pre><code>
-------------------- Testing Model Performance -------------------- 

Model loaded from models/history.pkl

Testing Daily__02152025.xlsx... Accuracy: 0.87
Testing Daily__02162025.xlsx... Accuracy: 0.90
Testing Daily__02172025.xlsx... Accuracy: 0.90
Testing Daily__02182025.xlsx... Accuracy: 0.85
Testing Daily__02192025.xlsx... Accuracy: 0.94
Testing Daily__02202025.xlsx... Accuracy: 0.94
Testing Daily__02212025.xlsx... Accuracy: 0.88
Testing Daily__02222025.xlsx... Accuracy: 0.82
Testing Daily__02232025.xlsx... Accuracy: 0.94
Testing Daily__02242025.xlsx... Accuracy: 0.90
</code></pre>

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
python -m src.scripts.predict --input tests/Daily__02152025.xlsx --model models/history.pkl --history data/training/aHistorical_01012024_02142025.xlsx
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