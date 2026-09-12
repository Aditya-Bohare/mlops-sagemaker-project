import joblib
import pandas as pd


# --------------------------------------------------
# 1. LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading model...")

model = joblib.load("../model/model.joblib")

print("Model loaded.")


# --------------------------------------------------
# 2. NEW CUSTOMER DATA
# --------------------------------------------------

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.50,
    "TotalCharges": 181.00
}


# --------------------------------------------------
# 3. CONVERT CUSTOMER TO DATAFRAME
# --------------------------------------------------

customer_df = pd.DataFrame([customer])


# --------------------------------------------------
# 4. MAKE PREDICTION
# --------------------------------------------------

prediction = model.predict(customer_df)[0]


# --------------------------------------------------
# 5. GET PROBABILITY
# --------------------------------------------------

probability = model.predict_proba(customer_df)[0][1]


# --------------------------------------------------
# 6. DISPLAY RESULT
# --------------------------------------------------

print("\nPrediction Result")
print("-------------------------")

if prediction == 1:
    print("Prediction: Customer WILL churn")
else:
    print("Prediction: Customer will NOT churn")

print(f"Churn Probability: {probability:.2%}")