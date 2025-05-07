import json
import joblib
import argparse

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from src.utils.load import load_to_df
from src.utils.processing import extract_features

PARAM_GRID = {
    'num_leaves': [15],
    'max_depth': [-1, 5],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [50, 100],
    'subsample': [0.6],
    'colsample_bytree': [0.6, 0.8]
}

TEST_DIR = "tests"

def prepare_training_data(df):
    df = df.copy()
    features = extract_features(df)
    df = df.merge(features[["planets_intensity", "percent_over"]], on="planets_intensity", how="left")
    df["label"] = df["result"].map({"OVER": 1, "UNDER": 0})
    return df[["percent_over", "label"]]

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
    grid = GridSearchCV(LGBMClassifier(verbose=-1), PARAM_GRID, cv=5, scoring='accuracy', verbose=3)
    grid.fit(X, y)
    print("Done")

    # Save the best model and parameters
    model = grid.best_estimator_
    best_params = grid.best_params_
    with open("data/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    joblib.dump(model, f"models/{args.output}")
    print(f"Model saved in models/{args.output}")

    print("-"*20, "Testing", "-"*20)

    for test in os.listdir(TEST_DIR)


if __name__ == "__main__":
    args = parse_args()
    main()
