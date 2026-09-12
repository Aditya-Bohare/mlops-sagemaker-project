import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    """
    Load the trained model when the SageMaker container starts.
    """

    model_path = os.path.join(
        model_dir,
        "model.joblib"
    )

    print(f"Loading model from: {model_path}")

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    return model


def input_fn(request_body, content_type):
    """
    Convert incoming JSON request into a pandas DataFrame.
    """

    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {content_type}"
        )

    input_data = json.loads(request_body)

    dataframe = pd.DataFrame(
        [input_data]
    )

    return dataframe


def predict_fn(input_data, model):
    """
    Run prediction using the trained sklearn pipeline.
    """

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }


def output_fn(prediction, accept):
    """
    Convert prediction result into JSON.
    """

    if accept != "application/json":
        raise ValueError(
            f"Unsupported accept type: {accept}"
        )

    return json.dumps(prediction)