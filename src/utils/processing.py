
def extract_features(df):
    """
    Extract features from the historical data. 
    Calculates the percentage of OVER results for each intensity level.
    Args:
        - df (pd.DataFrame)
    Returns:
        - pd.DataFrame: DataFrame with the percentage of OVER results for each intensity level.
    """
    feature_df = df.groupby("planets_intensity")["result"].value_counts().unstack().fillna(0)
    feature_df["percent_over"] = feature_df.get("OVER", 0) / (feature_df.get("OVER", 0) + feature_df.get("UNDER", 0) + 1e-5)
    return feature_df.reset_index()