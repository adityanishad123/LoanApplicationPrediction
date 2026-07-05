"""
app.py
------
Flask backend for the Loan Prediction Web Application.

Responsibilities:
    1. Serve the home page containing the loan application form.
    2. Accept POST form data on /predict.
    3. Validate every input field on the server side (defense in depth,
       in addition to the client-side JS validation).
    4. Encode categorical fields using the SAME mapping used at training
       time (loaded from model.pkl), then scale and feed the feature
       vector to the model in the EXACT column order used during training.
    5. Render the result back on the page.

IMPORTANT: model.pkl is only ever LOADED here, never retrained.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ----------------------------------------------------------------------
# Load the trained model bundle once, at startup.
# The bundle contains: model, scaler, feature_order, encodings
# ----------------------------------------------------------------------
model_bundle = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
except FileNotFoundError:
    model_load_error = "model.pkl was not found. Please run generate_and_train.py first."
except Exception as exc:  # noqa: BLE001
    model_load_error = f"Failed to load model.pkl: {exc}"


def validate_form(form):
    """
    Validates every incoming field.
    Returns a dict of {field_name: error_message} -- empty dict means valid.
    """
    errors = {}

    def get_float(name, min_val=None, max_val=None, label=None):
        label = label or name
        raw = form.get(name, "").strip()
        if raw == "":
            errors[name] = f"{label} is required."
            return None
        try:
            value = float(raw)
        except ValueError:
            errors[name] = f"{label} must be a number."
            return None
        if min_val is not None and value < min_val:
            errors[name] = f"{label} must be at least {min_val}."
            return None
        if max_val is not None and value > max_val:
            errors[name] = f"{label} must be at most {max_val}."
            return None
        return value

    def get_choice(name, allowed, label=None):
        label = label or name
        raw = form.get(name, "").strip()
        if raw == "":
            errors[name] = f"{label} is required."
            return None
        if raw not in allowed:
            errors[name] = f"Invalid value for {label}."
            return None
        return raw

    age = get_float("age", min_val=18, max_val=100, label="Age")
    gender = get_choice("gender", ["Male", "Female"], label="Gender")
    income = get_float("income", min_val=1, label="Annual Income")
    employment = get_choice(
        "employment", ["Salaried", "Self-Employed", "Unemployed"], label="Employment Status"
    )
    education = get_choice("education", ["Graduate", "Not Graduate"], label="Education Level")
    marital = get_choice("marital", ["Single", "Married"], label="Marital Status")
    dependents = get_float("dependents", min_val=0, max_val=10, label="Number of Dependents")
    loan_amount = get_float("loan_amount", min_val=1, label="Loan Amount")
    loan_term = get_float("loan_term", min_val=1, label="Loan Term")
    credit_score = get_float("credit_score", min_val=300, max_val=900, label="Credit Score")
    existing_loan = get_choice("existing_loan", ["Yes", "No"], label="Existing Loan")
    property_area = get_choice(
        "property_area", ["Urban", "Semiurban", "Rural"], label="Property Area"
    )

    cleaned = {
        "Age": age,
        "Gender": gender,
        "Annual_Income": income,
        "Employment_Status": employment,
        "Education_Level": education,
        "Marital_Status": marital,
        "Dependents": dependents,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Credit_Score": credit_score,
        "Existing_Loan": existing_loan,
        "Property_Area": property_area,
    }

    return cleaned, errors


@app.route("/", methods=["GET"])
def home():
    """Display the loan prediction form."""
    return render_template("index.html", model_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data, validate, run the model, and show the result."""
    if model_bundle is None:
        return render_template(
            "index.html",
            model_error=model_load_error or "Model is not available.",
        )

    cleaned, errors = validate_form(request.form)

    if errors:
        return render_template(
            "index.html",
            errors=errors,
            form_data=request.form,
            model_error=model_load_error,
        )

    try:
        encodings = model_bundle["encodings"]
        feature_order = model_bundle["feature_order"]
        scaler = model_bundle["scaler"]
        model = model_bundle["model"]

        # Encode categorical fields using the training-time mapping
        encoded = dict(cleaned)
        encoded["Gender"] = encodings["Gender"][cleaned["Gender"]]
        encoded["Employment_Status"] = encodings["Employment_Status"][cleaned["Employment_Status"]]
        encoded["Education_Level"] = encodings["Education_Level"][cleaned["Education_Level"]]
        encoded["Marital_Status"] = encodings["Marital_Status"][cleaned["Marital_Status"]]
        encoded["Existing_Loan"] = encodings["Existing_Loan"][cleaned["Existing_Loan"]]
        encoded["Property_Area"] = encodings["Property_Area"][cleaned["Property_Area"]]

        # Arrange features in the EXACT order used during training
        feature_vector = np.array([[encoded[col] for col in feature_order]])
        feature_vector_scaled = scaler.transform(feature_vector)

        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0][int(prediction)]

        approved = bool(prediction == 1)

        return render_template(
            "index.html",
            result={
                "approved": approved,
                "confidence": round(probability * 100, 1),
            },
            form_data=request.form,
            model_error=model_load_error,
        )

    except KeyError as exc:
        return render_template(
            "index.html",
            errors={"general": f"Unexpected value received: {exc}"},
            form_data=request.form,
            model_error=model_load_error,
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "index.html",
            errors={"general": f"Something went wrong while predicting: {exc}"},
            form_data=request.form,
            model_error=model_load_error,
        )


@app.errorhandler(404)
def not_found(_e):
    return render_template("index.html", model_error=model_load_error), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template(
        "index.html",
        errors={"general": "An internal server error occurred. Please try again."},
        model_error=model_load_error,
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
