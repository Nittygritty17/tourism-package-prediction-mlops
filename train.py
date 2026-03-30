
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

print("Starting training pipeline...")

# Create sample data if dataset not available
if not os.path.exists("train.csv"):
    print("Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'Age': np.random.randint(18, 70, n_samples),
        'TypeofContact': np.random.choice(['Company Invited', 'Self Enquiry'], n_samples),
        'CityTier': np.random.choice([1, 2, 3], n_samples),
        'Occupation': np.random.choice(['Salaried', 'Freelancer'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'ProdTaken': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    data.to_csv("train.csv", index=False)
    print("Sample data created: train.csv")

df = pd.read_csv("train.csv")
print(f"Data shape: {df.shape}")

if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"Categorical: {categorical_cols}")
print(f"Numerical: {numerical_cols}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat",
