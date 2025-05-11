import os
import json
import joblib
import argparse

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from src.utils.load import load_to_df
from src.utils.processing import prepare_training_data

PARAM_GRID = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
TEST_DIR = "data/tests"
PARAMS_PATH = "data/best_params.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LightGBM model")
    parser.add_argument("--input", required=True, help="Path to historical Excel file")
    parser.add_argument("--output", default="model.pkl", help="Output model file")
    return parser.parse_args()

def main():
    print("Preparing data...", end=" ")
    df = load_to_df(args.input)
    data = prepare_training_data(df)
    X = data[["percent_over"]]
    y = data["label"]
    print("Done")

    print("Training...", end=" ")
    grid = GridSearchCV(LGBMClassifier(verbose=-1), PARAM_GRID, cv=5, scoring='accuracy', verbose=2)
    grid.fit(X, y)
    print("Done")

    # Save the best model and parameters
    model = grid.best_estimator_
    best_params = grid.best_params_
    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)

    joblib.dump(model, f"models/{args.output}")
    print(f"Model saved in models/{args.output}")

    print("\n\n")
    print("-"*20, "Testing", "-"*20)
    print("\n")

    for test in os.listdir(TEST_DIR):
        if test.endswith(".xlsx"):
            print(f"Testing {test}...", end=" ")
            test_df = load_to_df(os.path.join(TEST_DIR, test))
            test_data = prepare_training_data(test_df)
            X_test = test_data[["percent_over"]]
            y_test = test_data["label"]
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    args = parse_args()
    main()
