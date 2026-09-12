import pandas as pd


# Load the raw dataset
data = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")


# Number of rows and columns
print("\nDataset Shape:")
print(data.shape)


# Show first 5 records
print("\nFirst 5 Rows:")
print(data.head())


# Show column names
print("\nColumns:")
print(data.columns.tolist())


# Understand data types
print("\nData Types:")
print(data.dtypes)


# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Check duplicate records
print("\nDuplicate Rows:")
print(data.duplicated().sum())


# Basic statistics
print("\nNumerical Statistics:")
print(data.describe())


# Check our target
print("\nChurn Distribution:")
print(data["Churn"].value_counts())


# Percentage distribution
print("\nChurn Percentage:")
print(data["Churn"].value_counts(normalize=True) * 100)

print("\nTotalCharges Sample:")
print(data["TotalCharges"].head(20))


print("\nRows where TotalCharges is blank:")
print(
    data[data["TotalCharges"].str.strip() == ""]
)


print("\nNumber of blank TotalCharges:")
print(
    (data["TotalCharges"].str.strip() == "").sum()
)