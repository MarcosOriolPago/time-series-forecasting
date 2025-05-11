import json
import joblib
import argparse

from lightgbm import LGBMClassifier

from src.utils.load import load_to_df
from src.utils.load import prepare_training_data
from src.globals import DEFAULT_PARAMS


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LightGBM model")
    parser.add_argument("--input", required=True, help="Path to historical Excel file")
    parser.add_argument("--output", default="model.pkl", help="Output model file")
    parser.add_argument("--params", default=None, help="Hyperparameters json file")
    return parser.parse_args()

def main():
    print("Preparing data...", end=" ")
    df = load_to_df(args.input)
    data = prepare_training_data(df)
    X = data[["percent_over"]]
    y = data["label"]
    # Load hyperparameters if provided
    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
        print(f"Using hyperparameters from {args.params}")
    else:
        params = DEFAULT_PARAMS
        print("Using default hyperparameters")
    print("Done")

    print("Training...", end=" ")
    model = LGBMClassifier(
        **params,
        verbose=-1
    )
    model.fit(X, y)
    print("Done")

    # Save the model
    joblib.dump(model, f"models/{args.output}")
    print(f"Model saved in models/{args.output}")


if __name__ == "__main__":
    args = parse_args()
    main()
