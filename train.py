
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from datasets import load_dataset
from huggingface_hub import login, HfApi, upload_file

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HF_DATASET_ID = os.getenv("HF_DATASET_REPO_ID")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO_ID")
TARGET_COL = "ProdTaken"

MLFLOW_EXPERIMENT = "tourism_prodTaken_model_building"
RANDOM_STATE = 42
N_ITER = 20
CV_FOLDS = 5
SCORING = "roc_auc"

HF_TOKEN = os.getenv("HF_TOKEN")

try:
    train_ds = load_dataset(HF_DATASET_ID, split="train", token=HF_TOKEN)
    test_ds = load_dataset(HF_DATASET_ID, split="test", token=HF_TOKEN)
except Exception as e:
    raise ValueError(f"Failed to load dataset from Hugging Face Hub: {HF_DATASET_ID}. Error: {e}
Please ensure the dataset exists and your HF_TOKEN is correctly set with read access.")

train_df = train_ds.to_pandas()
test_df = test_ds.to_pandas()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

for df_temp in [train_df, test_df]:
    if "Gender" in df_temp.columns:
        df_temp["Gender"] = df_temp["Gender"].replace({"Fe Male": "Female"})
    if "Unnamed: 0" in df_temp.columns:
        df_temp.drop(columns=["Unnamed: 0"], inplace=True)
    if "CustomerID" in df_temp.columns:
        df_temp.drop(columns=["CustomerID"], inplace=True)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL].astype(int)

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL].astype(int)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

models = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {
            "model__max_depth": [3, 5, 7, 10, 15, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__criterion": ["gini", "entropy"]
        }
    },
    "Bagging": {
        "model": BaggingClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_samples": [0.6, 0.8, 1.0],
            "model__max_features": [0.6, 0.8, 1.0]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "model__max_depth": [3, 5, 7]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
        ),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2]
        }
    }
}

mlflow.set_experiment(MLFLOW_EXPERIMENT)

results = []

for model_name, config in models.items():
    print(f"
Training {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", config["model"])
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config["params"],
        n_iter=N_ITER,
        scoring=SCORING,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=model_name) as run:
        search.fit(X_train, y_train)

        tuned_model = search.best_estimator_
        y_pred = tuned_model.predict(X_test)
        y_prob = tuned_model.predict_proba(X_test)[:, 1] if hasattr(tuned_model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cm = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        os.makedirs("artifacts", exist_ok=True)
        with open(f"artifacts/{model_name}_classification_report.json", "w") as f:
            json.dump(clf_report, f, indent=4)
        np.savetxt(f"artifacts/{model_name}_confusion_matrix.txt", cm, fmt="%d")

        mlflow.log_artifact(local_path=f"artifacts/{model_name}_classification_report.json")
        mlflow.log_artifact(local_path=f"artifacts/{model_name}_confusion_matrix.txt")

        mlflow.sklearn.log_model(tuned_model, artifact_path=f"{model_name}_model")

        results.append({
            "model_name": model_name,
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            **metrics,
            "run_id": run.info.run_id,
            "model": tuned_model # Store the actual best model for later use
        })

results_df = pd.DataFrame(results)
best_model_row = results_df.sort_values(by="roc_auc", ascending=False).iloc[0]
best_model_name = best_model_row["model_name"]
best_model = best_model_row["model"]
best_run_id = best_model_row["run_id"]
best_score = best_model_row["roc_auc"]

results_df.to_csv("model_comparison_results.csv", index=False)

print(f"
Best model: {best_model_name}")
print(f"Best ROC-AUC: {best_score:.4f}")

output_dir = "best_model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))

with open(os.path.join(output_dir, "_metadata.json"), "w") as f:
    json.dump({
        "model_name": best_model_name,
        "best_run_id": best_run_id,
        "best_score": best_score,
        "feature_names": X_train.columns.tolist()
    }, f, indent=4)

with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write("---\ntags:\n- travel\n- classification\n- mlflow\n- xgboost\n---\n# {{best_model_name}} Model for Travel Package Prediction\nThis model is the best performing model from an MLOps pipeline designed to predict customer purchases of a 'Wellness Tourism Package'.\n## Model Details\n*   **Model Type:** {{best_model_name}}\n*   **Best ROC-AUC Score:** {{best_score:.4f}}\n*   **MLflow Run ID:** {{best_run_id}}\n## Usage\nThis model can be loaded and used for inference on new customer data to predict their likelihood of purchasing the 'Wellness Tourism Package'.\n## Pipeline Overview\nThe MLOps pipeline involved:\n1.  **Data Preprocessing:** Handling missing values, encoding categorical features.\n2.  **Model Training:** Experimenting with various classical ML algorithms for tourism package prediction, including:\n    *   Decision Tree\n    *   Bagging\n    *   Random Forest\n    *   AdaBoost\n    *   Gradient Boosting\n    *   XGBoost\n3.  **Model Selection:** The {{best_model_name}} was selected based on its performance during Randomized Search Cross-Validation.[2]\n4.  **MLflow Tracking:** Logged parameters, metrics, and artifacts using MLflow for reproducibility and comparison.\n5.  **Hugging Face Hub Deployment:** The best model, along with its metadata and a README, is pushed to the Hugging Face Hub.\n## Metrics Captured\nThe following metrics were tracked for each model:\n*   Accuracy\n*   Precision\n*   Recall\n*   F1-score\n*   ROC-AUC\n".format(
        best_model_name=best_model_name,
        best_score=best_score,
        best_run_id=best_run_id
    ))

api = HfApi(token=HF_TOKEN)

api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, exist_ok=True)

upload_file(
    path_or_fileobj=os.path.join(output_dir, "model.joblib"),
    path_in_repo="model.joblib",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN
)

upload_file(
    path_or_fileobj=os.path.join(output_dir, "_metadata.json"),
    path_in_repo="_metadata.json",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN
)

upload_file(
    path_or_fileobj=os.path.join(output_dir, "README.md"),
    path_in_repo="README.md",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN
)

print(f"
Best model uploaded to Hugging Face Hub: https://huggingface.co/{HF_MODEL_REPO}")
