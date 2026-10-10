
from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_default_pipeline.joblib"

# Load the complete preprocessing + model pipeline
model = joblib.load(MODEL_PATH)


def predict_default(customer_data: dict) -> dict:
    """
    Predict the default class for one customer.

    customer_data must contain the same feature columns
    used during model training, excluding CUSTOMER_ID
    and DEFAULT_STATUS.
    """
    input_df = pd.DataFrame([customer_data])

    prediction = int(model.predict(input_df)[0])

    result = {
        "prediction": prediction,
        "label": (
            "Default predicted"
            if prediction == 1
            else "No default predicted"
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        result["class_probabilities"] = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }

    return result


if __name__ == "__main__":
    # Example values: replace these with valid feature values
    # from your dataset and keep the original column names.
    example_customer = {
        "AGE": 35,
        "GENDER": "Male",
        "INCOME": 60000,
        "EMPLOYMENT_YEARS": 8,
        "CITY": "Pune",
        "LOAN_AMOUNT": 300000,
        "LOAN_TERM": 36,
        "INTEREST_RATE": 10.5,
        "LOAN_TYPE": "Personal",
        "CREDIT_SCORE": 700,
        "PREVIOUS_DEFAULTS": 0,
        "NUMBER_OF_LOANS": 1,
    }

    result = predict_default(example_customer)
    print(result)