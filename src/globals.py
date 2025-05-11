# DECISION TREE PARAMETERS
PARAM_GRID = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
DEFAULT_PARAMS = {
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Paths
TEST_DIR = "data/tests"
PARAMS_PATH = "data/best_params.json"
PREDICTIONS_OUTPUT_DIR = "data/predictions"