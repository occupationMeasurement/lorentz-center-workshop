import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from .config import default_config

def train_model(X_train: pd.DataFrame, Y_train: pd.Series, config: dict = default_config):
    if (config["model"] == "gradient_boosting"):
        model = GradientBoostingClassifier()
        model.fit(X_train, Y_train)
    if (config["model"] == "knn"):
        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
    else:
        raise ValueError("Unknown model type")

    return model

def predict(model, X_test: pd.DataFrame, config: dict = default_config) -> pd.Series:
    if (config["model"] == "gradient_boosting" or config["model"] == "knn"):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Unknown model type")

    return y_pred

def correct_at_digit(y_pred: pd.Series, y_true: pd.Series, digit: int) -> pd.Series:
    return y_pred.str.slice(0, digit) == y_true.str.slice(0, digit)
