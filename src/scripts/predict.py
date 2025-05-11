import os
import joblib
import argparse

from src.utils.load import load_to_df
from src.utils.processing import extract_features

OUTPUT_DIR = "data/predictions"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to daily Excel file")
    parser.add_argument("--model", required=True, help="Path to trained model file (pkl)")
    parser.add_argument("--history", required=True, help="Path to historical Excel file")
    return parser.parse_args()

def main():
    model = joblib.load(args.model)

    # Load the daily and historical data
    daily_df = load_to_df(args.input)
    history_df = load_to_df(args.history)

    # Preprocess the daily data
    feature_lookup = extract_features(history_df)
    # Merge the daily data with the feature lookup table
    daily_df = daily_df.merge(feature_lookup[["planets_intensity", "percent_over"]],
                              on="planets_intensity", how="left").fillna(0.5)  # valor neutro

    # Predict the results using the model
    X = daily_df[["percent_over"]]
    predictions = model.predict(X)
    daily_df["predicted_result"] = ["OVER" if p == 1 else "UNDER" for p in predictions]

    # Save the predictions to a CSV file
    output_path = f"{OUTPUT_DIR}/{os.path.basename(args.input).replace('.xlsx', '_predictions.csv')}"
    daily_df[["player", "planets_intensity", "predicted_result"]].to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main()
