import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocess import (
    load_data,
    clean_data,
    split_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"


data = load_data(DATA_PATH)

data = clean_data(data)

X_train, X_test, y_train, y_test = split_data(data)

# --------------------------------------------------
# 1. CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


# --------------------------------------------------
# 2. BUILD PREPROCESSOR
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            "passthrough",
            NUMERICAL_FEATURES
        )
    ]
)


# --------------------------------------------------
# 3. CREATE SIMPLE MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 4. CREATE COMPLETE ML PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 5. TRAIN PIPELINE
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 6. CREATE MODEL DIRECTORY
# --------------------------------------------------

os.makedirs(
    "model",
    exist_ok=True
)


# --------------------------------------------------
# 7. SAVE COMPLETE PIPELINE
# --------------------------------------------------

joblib.dump(
    pipeline,
    "model/model.joblib"
)

print("\nModel saved:")
print("model/model.joblib")