import os
import time
import json

import boto3
from botocore.exceptions import ClientError
from sagemaker.core import image_uris

REGION = "ap-south-1"

BUCKET = "sagemaker-mlops-project-artifacts"

GIT_SHA = os.environ["GITHUB_SHA"]

EXECUTION_ROLE_ARN = os.environ[
    "SAGEMAKER_EXECUTION_ROLE_ARN"
]


model_name = f"churn-model-{GIT_SHA[:8]}"

endpoint_config_name = f"churn-config-{GIT_SHA[:8]}"

endpoint_name = f"churn-endpoint-{GIT_SHA[:8]}"


model_data_url = (
    f"s3://{BUCKET}/models/churn/"
    f"{GIT_SHA}/model.tar.gz"
)


print("Deployment information")
print("----------------------")
print(f"Model name      : {model_name}")
print(f"Endpoint config : {endpoint_config_name}")
print(f"Endpoint name   : {endpoint_name}")
print(f"Model artifact  : {model_data_url}")


sagemaker = boto3.client(
    "sagemaker",
    region_name=REGION
)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)

image_uri = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.4-2-py312",
    py_version="py3",
    instance_type="ml.m5.large",
    image_scope="inference"
)

print(f"Inference image : {image_uri}")

model_created = False
config_created = False
endpoint_created = False

try:

    # -------------------------------------------------
    # 1. CREATE SAGEMAKER MODEL
    # -------------------------------------------------

    print("\nCreating SageMaker model...")

    sagemaker.create_model(
        ModelName=model_name,

        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,

            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY":
                    "/opt/ml/model/code"
            }
        },

        ExecutionRoleArn=EXECUTION_ROLE_ARN
    )

    model_created = True

    print("SageMaker model created.")


    # -------------------------------------------------
    # 2. CREATE ENDPOINT CONFIGURATION
    # -------------------------------------------------

    print("\nCreating serverless endpoint configuration...")

    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,

        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,

                "ServerlessConfig": {
                    "MemorySizeInMB": 1024,
                    "MaxConcurrency": 1
                }
            }
        ]
    )

    config_created = True

    print("Endpoint configuration created.")


    # -------------------------------------------------
    # 3. CREATE ENDPOINT
    # -------------------------------------------------

    print("\nCreating serverless endpoint...")

    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    endpoint_created = True

    print("Endpoint creation started.")
    print("Waiting for endpoint to become InService...")

    start_time = time.time()
    max_wait_seconds = 15 * 60



    # -------------------------------------------------
    # 4. WAIT FOR ENDPOINT
    # -------------------------------------------------

    while True:

        response = sagemaker.describe_endpoint(
            EndpointName=endpoint_name
        )

        status = response["EndpointStatus"]

        print(f"Endpoint status: {status}")

        elapsed_time = time.time() - start_time

        if elapsed_time > max_wait_seconds:
            raise TimeoutError(
                "Endpoint did not become InService "
                "within 15 minutes."
            )

        if status == "InService":
            break

        if status == "Failed":

            reason = response.get(
                "FailureReason",
                "Unknown reason"
            )

            raise RuntimeError(
                f"Endpoint creation failed: {reason}"
            )

        time.sleep(30)


    # -------------------------------------------------
    # 5. TEST CUSTOMER
    # -------------------------------------------------

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
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 151.65
    }


    # -------------------------------------------------
    # 6. INVOKE ENDPOINT
    # -------------------------------------------------

    print("\nInvoking endpoint...")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(customer)
    )

    result = json.loads(
        response["Body"].read().decode("utf-8")
    )


    print("\nPrediction received!")
    print("--------------------")

    print(
        f"Prediction        : "
        f"{result['prediction']}"
    )

    print(
        f"Churn probability : "
        f"{result['churn_probability']:.2%}"
    )


# -------------------------------------------------
# 7. ALWAYS CLEAN UP
# -------------------------------------------------

finally:

    print("\nStarting cleanup...")


    # Delete endpoint first
    if endpoint_created:

        print("Deleting endpoint...")

        sagemaker.delete_endpoint(
            EndpointName=endpoint_name
        )

        print("Waiting for endpoint deletion...")

        while True:

            try:

                response = sagemaker.describe_endpoint(
                    EndpointName=endpoint_name
                )

                print(
                    f"Endpoint status: "
                    f"{response['EndpointStatus']}"
                )

                time.sleep(15)

            except ClientError as error:

                if (
                    error.response["Error"]["Code"]
                    == "ValidationException"
                ):
                    print("Endpoint deleted.")
                    break

                raise


    # Delete endpoint config second
    if config_created:

        print("Deleting endpoint configuration...")

        sagemaker.delete_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )

        print("Endpoint configuration deleted.")


    # Delete model last
    if model_created:

        print("Deleting SageMaker model...")

        sagemaker.delete_model(
            ModelName=model_name
        )

        print("SageMaker model deleted.")


    print("\nCleanup completed.")
