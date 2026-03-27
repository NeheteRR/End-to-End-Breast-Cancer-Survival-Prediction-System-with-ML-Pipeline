from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS


def train_all(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "RandomForest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "XGBoost":            XGBClassifier(**XGBOOST_PARAMS),
    }
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
    return models
