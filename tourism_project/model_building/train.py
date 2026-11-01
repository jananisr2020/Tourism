"""Tune an XGBoost classifier, track runs in MLflow, and save the best model."""
import json
import os
import urllib.request

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = f"{MODEL_DIR}/best_model.joblib"
THRESHOLD = 0.45          # decision threshold used by the app and the evaluation below
SEED = 42

# ---------------------------------------------------------------- data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
           "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
           "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
               "MaritalStatus", "Designation"]

# ---------------------------------------------------------------- model
preprocessor = make_column_transformer(
    (StandardScaler(), numeric),
    (OneHotEncoder(handle_unknown="ignore"), categorical),
)
class_weight = float(ytrain.value_counts()[0] / ytrain.value_counts()[1])
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=SEED,
                              eval_metric="logloss", n_jobs=-1)
pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [200, 400],
    "xgbclassifier__max_depth": [4, 6, 8],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
    "xgbclassifier__reg_lambda": [1, 5],
}

# ---------------------------------------------------------------- MLflow
def server_up(uri):
    try:
        urllib.request.urlopen(uri, timeout=3)
        return True
    except Exception:
        return False

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
if not server_up(tracking_uri):
    tracking_uri = "sqlite:///mlflow.db"      # fall back to a local SQLite store
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("tourism-wellness-package")
print("MLflow tracking URI:", tracking_uri)

# ---------------------------------------------------------------- tune + log
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)

with mlflow.start_run(run_name="xgboost-grid-search"):
    grid.fit(Xtrain, ytrain)

    # log every parameter combination as a nested run
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True, run_name=f"trial-{i}"):
            mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
            mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])
            mlflow.log_metric("cv_std_f1", results["std_test_score"][i])

    best_model = grid.best_estimator_
    mlflow.log_params({k.split("__")[1]: v for k, v in grid.best_params_.items()})
    mlflow.log_param("decision_threshold", THRESHOLD)

    def evaluate(X, y, prefix):
        proba = best_model.predict_proba(X)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        return {
            f"{prefix}_accuracy": accuracy_score(y, pred),
            f"{prefix}_precision": precision_score(y, pred),
            f"{prefix}_recall": recall_score(y, pred),
            f"{prefix}_f1": f1_score(y, pred),
            f"{prefix}_roc_auc": roc_auc_score(y, proba),
        }, pred

    train_metrics, _ = evaluate(Xtrain, ytrain, "train")
    test_metrics, test_pred = evaluate(Xtest, ytest, "test")
    mlflow.log_metrics({**train_metrics, "cv_best_f1": grid.best_score_, **test_metrics})

    print("\nBest parameters:", grid.best_params_)
    print(f"Best CV F1: {grid.best_score_:.4f}")
    print("\nTrain metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test metrics :", {k: round(v, 4) for k, v in test_metrics.items()})
    print(f"\nClassification report (test, threshold={THRESHOLD}):")
    print(classification_report(ytest, test_pred, target_names=["No purchase", "Purchase"]))

    # ------------------------------------------------------------ save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump({"threshold": THRESHOLD, "best_params": {k.split("__")[1]: v for k, v in grid.best_params_.items()},
                   "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()}}, f, indent=2)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(f"{MODEL_DIR}/model_meta.json")
    print("Saved model ->", MODEL_PATH)
