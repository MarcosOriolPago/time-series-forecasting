import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_pred, y_test, test_name):
    """
    Plot the confusion matrix for the model predictions.
    Args:
        - y_pred: Predicted labels
        - y_test: True labels
        - test_name: Name of the test for labeling the plot
    Returns:
        - None
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["UNDER", "OVER"], yticklabels=["UNDER", "OVER"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {test_name}")
    plt.tight_layout()
    plt.show()