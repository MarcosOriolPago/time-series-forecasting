
from src.utils.load import load_to_df


def measure_performance(results, predictions):
    """
    Measure the performance of the model by calculating the accuracy, precision, recall, and F1 score.
    Args:
        - results (pd.Series): Actual results (ground truth).
        - predictions (pd.Series): Predicted results by the model.
    Returns:
        - dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    performance = {}
    performance["accuracy"] = (results == predictions).mean()
    performance["precision"] = (results & predictions).sum() / predictions.sum()
    performance["recall"] = (results & predictions).sum() / results.sum()
    performance["f1"] = 2 * (performance["precision"] * performance["recall"]) / (performance["precision"] + performance["recall"] + 1e-5)

    return performance