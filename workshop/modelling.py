import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .config import default_config

def train_model(X_train: pd.DataFrame, Y_train: pd.Series, config: dict = default_config):
    mlflow.autolog(log_models=False)
    mlflow.set_tags(default_config)

    if (config["model"] == "gradient_boosting"):
        model = GradientBoostingClassifier()
        model.fit(X_train, Y_train)
    elif (config["model"] == "xgboost"):
        model = XGBClassifier(objective='multi:softprob') # , num_class=Y_train.unique().size
        model.fit(X_train, Y_train)
    elif (config["model"] == "knn"):
        model_type = KNeighborsClassifier()
        param_distributions = {'n_neighbors': range(1, 10)}
        cv = RandomizedSearchCV(
            model_type,
            param_distributions=param_distributions,
            cv=5,
            n_jobs=-1,
        )
        cv.fit(X_train, Y_train)
        model = cv.best_estimator_
    else:
        raise ValueError("Unknown model type")

    return model

def predict(model, X_test: pd.DataFrame, config: dict = default_config) -> pd.Series:
    if (config["model"] == "gradient_boosting" or config["model"] == "knn" or config["model"] == "xgboost"):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Unknown model type")

    return y_pred

def correct_at_digit(y_pred: pd.Series, y_true: pd.Series, digit: int) -> pd.Series:
    return y_pred.str.slice(0, digit) == y_true.str.slice(0, digit)
