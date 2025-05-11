import pandas as pd
from src.utils.processing import extract_features


def load_to_df(file_path: str) -> pd.DataFrame:
    data = pd.read_excel(
        file_path, 
        header=None, 
        usecols=[0, 7, 11, 12, 13], 
        names=["player", "result", "planets_combo", "intensity_degree", "planets_intensity"],
        engine='openpyxl'
    )

    return data


def prepare_training_data(df):
    """
    Prepare the training data by extracting features and creating labels. 
    Args:
        - df: DataFrame containing the data
    Returns:
        - DataFrame with features and labels
    """
    df = df.copy()
    features = extract_features(df)
    df = df.merge(features[["planets_intensity", "percent_over"]], on="planets_intensity", how="left")
    df["label"] = df["result"].map({"OVER": 1, "UNDER": 0})
    
    return df[["percent_over", "label"]]