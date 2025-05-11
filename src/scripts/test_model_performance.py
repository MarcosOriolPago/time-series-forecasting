import os
import argparse
import joblib

from src.utils.load import load_to_df, prepare_training_data

def parse_args():
    parser = argparse.ArgumentParser(description="Test model performance")
    parser.add_argument("--model", required=True, help="Path to trained model file (pkl)")
    parser.add_argument("--tests_folder", required=True, help="Folder containing test files")
    return parser.parse_args()

def main():
    # Load the model
    model = joblib.load(args.model)
    # Test the model on each test file
    print("\n","-"*20, "Testing Model Performance", "-"*20, "\n")
    print(f"Model loaded from {args.model}\n")
    for test in os.listdir(args.tests_folder):
        if test.endswith(".xlsx"):
            print(f"Testing {test}...", end=" ")
            test_df = load_to_df(os.path.join(args.tests_folder, test))
            test_data = prepare_training_data(test_df)
            X_test = test_data[["percent_over"]]
            y_test = test_data["label"]
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    args = parse_args()
    main()