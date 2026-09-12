import json
import sys

sys.path.append("src")

from sagemaker_inference import (
    model_fn,
    input_fn,
    predict_fn,
    output_fn
)


model = model_fn("model")


customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}


request_body = json.dumps(customer)


input_data = input_fn(
    request_body,
    "application/json"
)


prediction = predict_fn(
    input_data,
    model
)


response = output_fn(
    prediction,
    "application/json"
)


print("\nSageMaker-style response")
print("-------------------------")
print(response)