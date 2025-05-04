import pandas as pd
import joblib
import argparse
from lightgbm import LGBMClassifier

from src.utils.load import load_to_df

def extract_features(df):
    # Conteo de OVER por símbolo
    feature_df = df.groupby("planets_intensity")["result"].value_counts().unstack().fillna(0)
    feature_df["percent_over"] = feature_df.get("OVER", 0) / (feature_df.get("OVER", 0) + feature_df.get("UNDER", 0) + 1e-5)
    return feature_df.reset_index()

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
    df = load_to_df(args.input)
    
    data = prepare_training_data(df)
    X = data[["percent_over"]]
    y = data["label"]

    model = LGBMClassifier()
    model.fit(X, y)

    joblib.dump(model, f"models/{args.output}")
    print(f"Modelo guardado en models/{args.output}")

if __name__ == "__main__":
    args = parse_args()
    main()
