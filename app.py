
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask
app = Flask(__name__)

# Load the saved ML pipeline
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_default_pipeline.joblib"

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Loan Default Prediction API is running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read customer data from the JSON request
        customer_data = request.get_json(silent=True)

        if not isinstance(customer_data, dict) or not customer_data:
            return jsonify({
                "error": "Please send customer details as a JSON object."
            }), 400

        # Convert input into a DataFrame
        input_df = pd.DataFrame([customer_data])

        # Make prediction using the saved pipeline
        prediction = int(model.predict(input_df)[0])

        result = {
            "prediction": prediction,
            "label": (
                "Default predicted"
                if prediction == 1
                else "No default predicted"
            )
        }

        # Return class probabilities if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            result["class_probabilities"] = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(model.classes_, probabilities)
            }

        return jsonify(result)

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": "Invalid input data",
            "details": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)