import os
import json
import joblib
import argparse

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from src.utils.load import load_to_df
from src.utils.load import prepare_training_data
from src.globals import PARAM_GRID, PARAMS_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LightGBM model")
    parser.add_argument("--input", required=True, help="Path to historical Excel file")
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

    # Save the best parameters
    best_params = grid.best_params_
    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved in {PARAMS_PATH}")


if __name__ == "__main__":
    args = parse_args()
    main()
