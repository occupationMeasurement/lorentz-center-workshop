import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from xgboost import XGBClassifier

from .config import default_config

def train_model(X_train: pd.DataFrame, Y_train: pd.Series, config: dict = default_config):
    mlflow.autolog(log_models=False)
    mlflow.set_tags(default_config)

    train_output = dict()

    if (config["model"] == "gradient_boosting"):
        model = GradientBoostingClassifier()
        model.fit(X_train, Y_train)
    elif (config["model"] == "xgboost"):
        encoder = preprocessing.LabelEncoder().fit(Y_train)
        model = XGBClassifier(
            objective='multi:softmax',
        )
        model.fit(X_train, encoder.transform(Y_train))
        train_output["encoder"] = encoder
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

    train_output["model"] = model

    return train_output

def predict(train_output, X_test: pd.DataFrame, config: dict = default_config) -> pd.Series:
    model = train_output["model"]

    if (config["model"] == "gradient_boosting" or config["model"] == "knn"):
        y_pred = model.predict(X_test)
    elif (config["model"] == "xgboost"):
        encoder = train_output["encoder"]
        y_pred = encoder.inverse_transform(model.predict(X_test))
    else:
        raise ValueError("Unknown model type")

    return y_pred

def correct_at_digit(y_pred: pd.Series, y_true: pd.Series, digit: int) -> pd.Series:
    return y_pred.str.slice(0, digit) == y_true.str.slice(0, digit)
