import pandas as pd

from sklearn.model_selection import train_test_split


NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def load_data(file_path):

    print("Loading dataset...")

    data = pd.read_csv(file_path)

    print(f"Dataset loaded: {data.shape}")

    return data


def clean_data(data):

    print("Cleaning dataset...")

    # Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    # New customers have tenure=0,
    # so missing TotalCharges is treated as 0
    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    # customerID is not useful for model training
    data = data.drop(columns=["customerID"])

    # Convert target into numerical form
    data["Churn"] = data["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    print("Dataset cleaned.")

    return data


def split_data(data):

    print("Splitting dataset...")

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    return X_train, X_test, y_train, y_test