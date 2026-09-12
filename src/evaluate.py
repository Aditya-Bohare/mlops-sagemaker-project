import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import (
    load_data,
    clean_data,
    split_data
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
MODEL_PATH = "model/model.joblib"

MIN_F1_SCORE = 0.50


# --------------------------------------------------
# LOAD AND PREPARE TEST DATA
# --------------------------------------------------

data = load_data(DATA_PATH)

data = clean_data(data)

X_train, X_test, y_train, y_test = split_data(data)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

matrix = confusion_matrix(y_test, predictions)


# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------

print("\nModel Evaluation")
print("------------------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(matrix)


# --------------------------------------------------
# MODEL QUALITY GATE
# --------------------------------------------------

print("\nQuality Gate")
print("------------------------")

if f1 >= MIN_F1_SCORE:

    print(
        f"PASS: F1 score {f1:.4f} "
        f">= {MIN_F1_SCORE}"
    )

else:

    raise ValueError(
        f"FAIL: F1 score {f1:.4f} "
        f"< {MIN_F1_SCORE}"
    )