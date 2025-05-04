import pandas as pd
import joblib
import argparse

def extract_features_from_history(history_df):
    feature_df = history_df.groupby("planets_intensity")["result"].value_counts().unstack().fillna(0)
    feature_df["percent_over"] = feature_df.get("OVER", 0) / (feature_df.get("OVER", 0) + feature_df.get("UNDER", 0) + 1e-5)
    return feature_df.reset_index()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to daily Excel file")
    parser.add_argument("--model", required=True, help="Path to trained model file (pkl)")
    parser.add_argument("--history", required=True, help="Path to historical Excel file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV with predictions")

    return parser.parse_args()

def main():
    model = joblib.load(args.model)

    daily_df = pd.read_excel(args.input, header=None, usecols=[0, 7, 11, 12, 13],
                             names=["player", "result", "planets_combo", "intensity_degree", "planets_intensity"])

    history_df = pd.read_excel(args.history, header=None, usecols=[0, 7, 11, 12, 13],
                               names=["player", "result", "planets_combo", "intensity_degree", "planets_intensity"])

    feature_lookup = extract_features_from_history(history_df)

    daily_df = daily_df.merge(feature_lookup[["planets_intensity", "percent_over"]],
                              on="planets_intensity", how="left").fillna(0.5)  # valor neutro

    X = daily_df[["percent_over"]]
    print(X)
    predictions = model.predict(X)
    print(predictions)
    daily_df["predicted_result"] = ["OVER" if p == 1 else "UNDER" for p in predictions]

    daily_df[["player", "planets_intensity", "predicted_result"]].to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main()
